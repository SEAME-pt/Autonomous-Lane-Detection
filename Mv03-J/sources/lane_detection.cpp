#include <opencv2/opencv.hpp>
#include "jetracer.hpp"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <vector>
#include <atomic>
#include <csignal>
#include <cmath>
#include <memory>
#include "NvInferPlugin.h"

using namespace nvinfer1;

JetCar jetracer(0x40, 0x60);
std::atomic<bool> running(true);

// Logger simples para TensorRT
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

// Variáveis TensorRT
IRuntime* runtime = nullptr;
ICudaEngine* engine = nullptr;
IExecutionContext* context = nullptr;
void* buffers[2];

const int INPUT_SIZE = 3 * 512 * 512;
const int OUTPUT_SIZE = 2 * 512 * 512;

// Carrega modelo TRT serializado
std::vector<char> loadTRTModel(const std::string& modelPath) {
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("Falha ao abrir arquivo TRT.");
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

// Inicializa TensorRT
void initializeTensorRT() {
    initLibNvInferPlugins(&gLogger, "");
    auto modelData = loadTRTModel("lanenet.trt");
    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
    context = engine->createExecutionContext();
    cudaMalloc(&buffers[0], INPUT_SIZE * sizeof(float));
    cudaMalloc(&buffers[1], OUTPUT_SIZE * sizeof(float));
}

// Libera recursos TensorRT
void destroyTensorRT() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete context;
    delete engine;
    delete runtime;
}

// Pré-processa imagem
cv::Mat preprocess_frame(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(512, 512));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    return resized;
}

// Inferência do modelo TensorRT
std::vector<float> inferLaneNet(const cv::Mat& frame) {
    cv::Mat input_tensor = preprocess_frame(frame);
    float chw_input[INPUT_SIZE]; // [3 * 512 * 512]
    int idx = 0;
for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 512; ++i) {
        for (int j = 0; j < 512; ++j) {
            chw_input[idx++] = input_tensor.at<cv::Vec3f>(i, j)[c];
        }
    }
}
cudaMemcpy(buffers[0], chw_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	
    //cudaMemcpy(buffers[0], input_tensor.data, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    context->executeV2(buffers);
    std::vector<float> output_data(OUTPUT_SIZE);
    cudaMemcpy(output_data.data(), buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    return output_data;
}

// Estrutura para armazenar informações da faixa
struct LaneInfo {
    cv::Point2f center;
    bool detected;
};

// // Detecção simples da faixa a partir da máscara
// LaneInfo detect_lane(const cv::Mat& mask, const cv::Size& original_size) {
//     LaneInfo lane{};
//     cv::Mat resized_mask;
//     cv::resize(mask, resized_mask, original_size);
//     cv::threshold(resized_mask, resized_mask, 0.1, 1, cv::THRESH_BINARY);
//     auto M = cv::moments(resized_mask, true);
//     if (M.m00 > original_size.width * original_size.height * 0.01) {
//         lane.center = {float(M.m10 / M.m00), float(M.m01 / M.m00)};
//         lane.detected = true;
//     } else {
//         lane.detected = false;
//     }
//     return lane;
// }


LaneInfo detect_lane(const cv::Mat& mask, const cv::Size& original_size, const std::string& label = "") {
    LaneInfo lane{};
    cv::Mat resized_mask;
    cv::resize(mask, resized_mask, original_size);

    // Aplicar threshold
    // cv::threshold(resized_mask, resized_mask, 0, 1, cv::THRESH_BINARY);

    // 🔍 Mostrar a máscara binarizada para debug
    cv::Mat debug_vis;
    resized_mask.convertTo(debug_vis, CV_8U, 255.0); // converte para 0–255
    cv::imshow(label + " Threshold", debug_vis);

    cv::imshow(label + " mask", resized_mask);

    auto M = cv::moments(resized_mask, true);
    if (M.m00 > original_size.width * original_size.height * 0.01) {
        lane.center = {float(M.m10 / M.m00), float(M.m01 / M.m00)};
        lane.detected = true;
    } else {
        lane.detected = false;
    }
    return lane;
}


void signal_handler(int /*signal*/) {
    running = false;
    std::cout << "[INFO] Encerrando..." << std::endl;
}

int main() {
    signal(SIGINT, signal_handler);
    initializeTensorRT();

    cv::VideoCapture cap(
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=512, height=512, "
        "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, "
        "format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1",
        cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmera!" << std::endl;
        return -1;
    }

    jetracer.start();

    while (running) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        auto output = inferLaneNet(frame);

        // Recriar as máscaras a partir da saída do modelo
        cv::Mat left_mask(512, 512, CV_32F, output.data());
        cv::Mat right_mask(512, 512, CV_32F, output.data() + 512 * 512);

        LaneInfo left_lane = detect_lane(left_mask, frame.size());
        LaneInfo right_lane = detect_lane(right_mask, frame.size());

        if (left_lane.detected)
            cv::circle(frame, left_lane.center, 5, {0, 255, 0}, -1); // verde
        if (right_lane.detected)
            cv::circle(frame, right_lane.center, 5, {0, 0, 255}, -1); // vermelho

        // --- Cálculo do ângulo de direção ---
        float steering_angle = 0;
        int desired_position_x = frame.cols / 2;
        float offset_ratio = 0.25f;

        if (left_lane.detected && right_lane.detected) {
            desired_position_x = (left_lane.center.x + right_lane.center.x) / 2;
        }
        else if (left_lane.detected) {
            desired_position_x = left_lane.center.x + frame.cols * offset_ratio;
        }
        else if (right_lane.detected) {
            desired_position_x = right_lane.center.x - frame.cols * offset_ratio;
        }

        float offset = desired_position_x - (frame.cols / 2.0f);
        steering_angle = std::clamp((offset / (frame.cols / 2.0f)) * 75.0f, -90.0f, 90.0f);

        jetracer.smooth_steering(steering_angle, 15); // curva mais agressiva

        // --- VISUALIZAÇÃO DA SAÍDA DO MODELO ---
        // Cria uma imagem colorida com: Azul = direita, Verde = esquerda
        cv::Mat color_mask;
        std::vector<cv::Mat> channels = {
            right_mask,  // Azul
            left_mask,   // Verde
            cv::Mat::zeros(512, 512, CV_32F)  // Vermelho
        };
        cv::merge(channels, color_mask);

        // Converte para exibição
        cv::Mat display_mask;
        color_mask.convertTo(display_mask, CV_8UC3, 255.0);

        // Redimensiona para o tamanho do frame original
        cv::resize(display_mask, display_mask, frame.size());

        // Combina lado a lado com a imagem da câmera
        cv::Mat combined;
        cv::hconcat(frame, display_mask, combined);

        // Mostra na tela
        cv::imshow("Original + Model Output", combined);

        if (cv::waitKey(1) == 'q') break;
    }

    destroyTensorRT();
    jetracer.stop();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}


/*
int main() {
    signal(SIGINT, signal_handler);
    initializeTensorRT();

    cv::VideoCapture cap(
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, "
        "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, "
        "format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1",
        cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmera!" << std::endl;
        return -1;
    }

    jetracer.start();
    jetracer.set_speed(90);

    while (running) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        auto output = inferLaneNet(frame);

        cv::Mat left_mask(512, 512, CV_32F, output.data());
        cv::Mat right_mask(512, 512, CV_32F, output.data() + 512 * 512);

        LaneInfo left_lane = detect_lane(left_mask, frame.size());
        LaneInfo right_lane = detect_lane(right_mask, frame.size());

        if (left_lane.detected)
            cv::circle(frame, left_lane.center, 5, {0, 255, 0}, -1); // verde
        if (right_lane.detected)
            cv::circle(frame, right_lane.center, 5, {0, 0, 255}, -1); // vermelho

        // --- NOVA LÓGICA DE DIREÇÃO ---
        float steering_angle = 0;
        int desired_position_x = frame.cols / 2;  // posição desejada padrão = centro da imagem
        float offset_ratio = 0.25f;               // distância para se manter ao lado da faixa (25% da largura)

        if (left_lane.detected && right_lane.detected) {
            // Centralizar entre as duas faixas
            desired_position_x = (left_lane.center.x + right_lane.center.x) / 2;
        }
        else if (left_lane.detected) {
            // Manter-se à direita da faixa esquerda
            desired_position_x = left_lane.center.x + frame.cols * offset_ratio;
        }
        else if (right_lane.detected) {
            // Manter-se à esquerda da faixa direita
            desired_position_x = right_lane.center.x - frame.cols * offset_ratio;
        }

        // Converter posição desejada em ângulo de direção
        float offset = desired_position_x - (frame.cols / 2.0f);
//        steering_angle = std::clamp((offset / (frame.cols / 2.0f)) * 45.0f, -45.0f, 45.0f);
        steering_angle = std::clamp((offset / (frame.cols / 2.0f)) * 75.0f, -45.0f, 45.0f);

        // Aplicar direção e velocidade
        //jetracer.smooth_steering(steering_angle, 5);
        jetracer.smooth_steering(steering_angle, 15);
        if (left_lane.detected || right_lane.detected)
            jetracer.set_speed(120);
        else
            jetracer.set_speed(0);

//        jetracer.set_speed(120);

        cv::imshow("Lane Detection", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    destroyTensorRT();
    jetracer.stop();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
*/

/*
int main() {
    signal(SIGINT, signal_handler);
    initializeTensorRT();

    cv::VideoCapture cap(
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, "
        "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, "
        "format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1",
        cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmera!" << std::endl;
        return -1;
    }

    jetracer.start();

    while (running) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        auto output = inferLaneNet(frame);

        cv::Mat left_mask(512, 512, CV_32F, output.data());
        cv::Mat right_mask(512, 512, CV_32F, output.data() + 512 * 512);

        LaneInfo left_lane = detect_lane(left_mask, frame.size());
        LaneInfo right_lane = detect_lane(right_mask, frame.size());

        if (left_lane.detected)
            cv::circle(frame, left_lane.center, 5, {0, 255, 0}, -1);
        if (right_lane.detected)
            cv::circle(frame, right_lane.center, 5, {0, 0, 255}, -1);

        float steering_angle = 0;
        if (left_lane.detected && right_lane.detected) {
            float lane_center = (left_lane.center.x + right_lane.center.x) / 2;
            steering_angle = (lane_center - frame.cols / 2) / (frame.cols / 2) * 45;
        }

        // Compensação quando apenas uma faixa é detectada
        else if (left_lane.detected) {
            float offset = left_lane.center.x + (frame.cols / 4.0f) - (frame.cols / 2.0f);
            steering_angle = (offset / (frame.cols / 2.0f)) * 45.0f;

    // Compensação extra para manter mais centralizado
            steering_angle -= 10.0f;
        }
        else if (right_lane.detected) {
            float offset = right_lane.center.x - (frame.cols * 3.0f / 4.0f) - (frame.cols / 2.0f);
            steering_angle = (offset / (frame.cols / 2.0f)) * 45.0f;

    // Compensação extra para manter mais centralizado
            steering_angle += 10.0f;
        }


        jetracer.smooth_steering(steering_angle, 5);
        jetracer.set_speed(120);
//        jetracer.set_speed(left_lane.detected || right_lane.detected ? 35 : 0);

        cv::imshow("Lane Detection", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    destroyTensorRT();
    jetracer.stop();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

*/
