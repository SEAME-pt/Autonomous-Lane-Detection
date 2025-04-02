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
const int OUTPUT_SIZE = 1 * 512 * 512;

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
    // Ou alternativamente: cudaMemcpy(buffers[0], input_tensor.data, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    context->executeV2(buffers);
    
    std::vector<float> output_data(OUTPUT_SIZE);
    cudaMemcpy(output_data.data(), buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    return output_data;
}

// Função para converter o vetor de saída em uma imagem para visualização
cv::Mat visualizeOutput(const std::vector<float>& output_data) {
    // Cria uma matriz de 512x512 com os dados da saída (sem copiar os dados)
    cv::Mat outputMat(512, 512, CV_32F, const_cast<float*>(output_data.data()));
    
    // Clona a matriz para ter uma cópia dos dados
    cv::Mat outputClone = outputMat.clone();
    
    // Converte os valores de [0, 1] para [0, 255] e de float para 8 bits para exibição
    cv::Mat display;
    outputClone.convertTo(display, CV_8U, 255.0);
    
    return display;
}



// Estrutura para armazenar informações da faixa
struct LaneInfo {
    cv::Point2f center;
    bool detected;
};

LaneInfo detect_lane(const cv::Mat& mask, const cv::Size& original_size) {
    LaneInfo lane{};
    cv::Mat resized_mask;
    cv::resize(mask, resized_mask, original_size);

    // Aplica threshold binário (ajuste o valor conforme o comportamento desejado)
    cv::threshold(resized_mask, resized_mask, 0.1, 1.0, cv::THRESH_BINARY);

    // 🔍 Visualização da máscara unificada após threshold
    cv::Mat debug_vis;
    resized_mask.convertTo(debug_vis, CV_8U, 255.0);
    cv::imshow("Máscara Unificada - Threshold", debug_vis);

    // Detecta centro da faixa
    auto M = cv::moments(resized_mask, true);
    if (M.m00 > original_size.width * original_size.height * 0.01) {
        lane.center = {float(M.m10 / M.m00), float(M.m01 / M.m00)};
        lane.detected = true;
    } else {
        lane.detected = false;
    }
    return lane;
}


// Função para exibir apenas os pixels classificados como 1
cv::Mat displayClass1(const cv::Mat& modelOutput, float threshold = 0.5) {
    cv::Mat binary;
    // Aplica um threshold: pixels com valor maior que 'threshold' serão 1 e os demais 0
    cv::threshold(modelOutput, binary, threshold, 1.0, cv::THRESH_BINARY);
    
    // Converte a imagem para 8 bits (0 a 255) para facilitar a visualização
    cv::Mat display;
    binary.convertTo(display, CV_8U, 255);
    return display;
}



void signal_handler(int /*signal*/) {
    running = false;
    std::cout << "[INFO] Encerrando..." << std::endl;
}





// --- Função principal ---
int main() {
    // Configura o handler para sinal de encerramento
    signal(SIGINT, signal_handler);
    
    // Inicializa o TensorRT (modelo, buffers, etc.)
    initializeTensorRT();

    // Abre a câmera usando pipeline GStreamer
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

    // Inicia o carrinho (se necessário)
    jetracer.start();

    while (running) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty())
            break;

        // Executa a inferência e obtém o vetor de saída do modelo
        auto output = inferLaneNet(frame);

        // Converte a saída do modelo para uma imagem para visualização
        cv::Mat model_vis = visualizeOutput(output);

        // Redimensiona a imagem da saída para o mesmo tamanho do frame da câmera
        cv::resize(model_vis, model_vis, frame.size());

        // Exibe a imagem da câmera em uma janela
        cv::imshow("Camera", frame);
        // Exibe a saída do modelo em outra janela
        cv::imshow("Model Output", model_vis);

        if (cv::waitKey(1) == 'q')
            break;
    }

    // Libera os recursos do TensorRT, para o carrinho, e fecha a câmera e janelas
    destroyTensorRT();
    jetracer.stop();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}





// int main() {
//     signal(SIGINT, signal_handler);
//     initializeTensorRT();

//     cv::VideoCapture cap(
//         "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=512, height=512, "
//         "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, "
//         "format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! "
//         "appsink drop=true max-buffers=1",
//         cv::CAP_GSTREAMER);

//     if (!cap.isOpened()) {
//         std::cerr << "Erro ao abrir a câmera!" << std::endl;
//         return -1;
//     }

//     jetracer.start();

//     while (running) {
//         cv::Mat frame;
//         if (!cap.read(frame) || frame.empty()) break;

//         auto output = inferLaneNet(frame);

//         visualizeOutput(output);

//         // cv::imshow(" output", output);

//         // Recriar as máscaras a partir da saída do modelo
//         cv::Mat left_mask(512, 512, CV_32F, output.data());
//         cv::Mat right_mask(512, 512, CV_32F, output.data() + 512 * 512);

//         // LaneInfo left_lane = detect_lane(left_mask, frame.size());
//         // LaneInfo right_lane = detect_lane(right_mask, frame.size());

//         // if (left_lane.detected)
//         //     cv::circle(frame, left_lane.center, 5, {0, 255, 0}, -1); // verde
//         // if (right_lane.detected)
//         //     cv::circle(frame, right_lane.center, 5, {0, 0, 255}, -1); // vermelho

//         // // --- Cálculo do ângulo de direção ---
//         // float steering_angle = 0;
//         // int desired_position_x = frame.cols / 2;
//         // float offset_ratio = 0.25f;

//         // if (left_lane.detected && right_lane.detected) {
//         //     desired_position_x = (left_lane.center.x + right_lane.center.x) / 2;
//         // }
//         // else if (left_lane.detected) {
//         //     desired_position_x = left_lane.center.x + frame.cols * offset_ratio;
//         // }
//         // else if (right_lane.detected) {
//         //     desired_position_x = right_lane.center.x - frame.cols * offset_ratio;
//         // }

//         // float offset = desired_position_x - (frame.cols / 2.0f);
//         // steering_angle = std::clamp((offset / (frame.cols / 2.0f)) * 75.0f, -90.0f, 90.0f);

//         jetracer.smooth_steering(steering_angle, 15); // curva mais agressiva

//         // --- VISUALIZAÇÃO DA SAÍDA DO MODELO ---
//         // Cria uma imagem colorida com: Azul = direita, Verde = esquerda
//         cv::Mat color_mask;
//         std::vector<cv::Mat> channels = {
//             right_mask,  // Azul
//             left_mask,   // Verde
//             cv::Mat::zeros(512, 512, CV_32F)  // Vermelho
//         };
//         cv::merge(channels, color_mask);

//         // Converte para exibição
//         cv::Mat display_mask;
//         color_mask.convertTo(display_mask, CV_8UC3, 255.0);

//         // Redimensiona para o tamanho do frame original
//         cv::resize(display_mask, display_mask, frame.size());

//         // Combina lado a lado com a imagem da câmera
//         cv::Mat combined;
//         cv::hconcat(frame, display_mask, combined);

//         // Mostra na tela
//         cv::imshow("Original + Model Output", combined);

//         if (cv::waitKey(1) == 'q') break;
//     }

//     destroyTensorRT();
//     jetracer.stop();
//     cap.release();
//     cv::destroyAllWindows();
//     return 0;
// }