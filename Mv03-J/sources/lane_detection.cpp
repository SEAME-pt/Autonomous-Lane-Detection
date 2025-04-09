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
#include <mosquitto.h>
#include <thread>
#include <mutex>

using namespace nvinfer1;
std::mutex frame_mutex;
cv::Mat latest_frame;
JetCar jetracer(0x40, 0x60);
std::atomic<bool> running(true);

#define LOCAL_BROKER "10.21.221.67"
#define LOCAL_PORT 1883
#define MQTT_TOPIC "jetracer/lane_touch"

struct mosquitto *mosq;
std::thread mqtt_thread;

void mqtt_loop_thread() {
    while (running) {
        int rc = mosquitto_loop(mosq, 1000, 1); // Timeout de 1000ms
        if (rc != MOSQ_ERR_SUCCESS) {
            std::cerr << "[MQTT] Erro no loop: " << mosquitto_strerror(rc) << std::endl;
            // Tenta reconectar
            if (mosquitto_reconnect(mosq) == MOSQ_ERR_SUCCESS) {
                std::cout << "[MQTT] Reconectado ao broker!" << std::endl;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void initMQTT() {
    mosquitto_lib_init();
    mosq = mosquitto_new("lane_publisher", true, NULL);
    if (!mosq) throw std::runtime_error("Erro ao criar cliente MQTT");
    if (mosquitto_connect(mosq, LOCAL_BROKER, LOCAL_PORT, 60) != MOSQ_ERR_SUCCESS) {
        throw std::runtime_error("Erro ao conectar ao broker MQTT");
    }
    std::cout << "[MQTT] Conectado ao broker MQTT!" << std::endl;
    mqtt_thread = std::thread(mqtt_loop_thread); // Inicia a thread do loop
}

void publishLaneTouch(int value) {
    std::string message = std::to_string(value);
    int ret = mosquitto_publish(mosq, NULL, MQTT_TOPIC, message.size(), message.c_str(), 0, false);
    if (ret != MOSQ_ERR_SUCCESS) {
        std::cerr << "[MQTT] Falha ao publicar: " << mosquitto_strerror(ret) << std::endl;
    } else {
        std::cout << "[MQTT] Publicado: " << message << std::endl;
    }
}

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
    auto modelData = loadTRTModel("models/lanenet.trt");
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

// cv::Mat preprocess_frame(const cv::Mat& frame) {
//     cv::Mat resized;
//     cv::resize(frame, resized, cv::Size(512, 512));
//     resized.convertTo(resized, CV_32F, 1.0 / 255.0);
//     cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

//     // Ajuste os pontos para equilibrar os dois lados
//     std::vector<cv::Point2f> src_points = {
//         {0, 512}, {512, 512}, {0, 0}, {512, 0}
//     };
//     std::vector<cv::Point2f> dst_points = {
//         {128, 512}, {384, 512}, {0, 0}, {512, 0} // Centraliza mais a base
//     };
//     cv::Mat transform = cv::getPerspectiveTransform(src_points, dst_points);
//     cv::Mat warped;
//     cv::warpPerspective(resized, warped, transform, cv::Size(512, 512));

//     // Opcional: exibir a imagem transformada para depuração
//     cv::Mat debug_image;
//     cv::cvtColor(warped, debug_image, cv::COLOR_RGB2BGR);
//     debug_image.convertTo(debug_image, CV_8U, 255.0);
//     cv::imshow("Transformed Image", debug_image);

//     return warped;
// }

cv::Mat preprocess_frame(const cv::Mat& frame, cv::Mat& debug_image) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(512, 512));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // Ajuste os pontos para equilibrar os dois lados
    std::vector<cv::Point2f> src_points = {
        {0, 512}, {512, 512}, {0, 0}, {512, 0}
    };
    std::vector<cv::Point2f> dst_points = {
        {128, 512}, {384, 512}, {0, 0}, {512, 0} // Centraliza mais a base
    };
    cv::Mat transform = cv::getPerspectiveTransform(src_points, dst_points);
    cv::Mat warped;
    cv::warpPerspective(resized, warped, transform, cv::Size(512, 512));

    // Gera a debug_image para depuração
    cv::cvtColor(warped, debug_image, cv::COLOR_RGB2BGR);
    debug_image.convertTo(debug_image, CV_8U, 255.0);

    return warped;
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

void signal_handler(int /*signal*/) {
    running = false;
    std::cout << "[INFO] Encerrando..." << std::endl;
}

cv::Mat visualizeOutput(const std::vector<float>& output_data, float threshold) {
    cv::Mat outputMat(512, 512, CV_32F, const_cast<float*>(output_data.data()));
    cv::Mat outputClone = outputMat.clone();

    // Aplica sigmoide com fator de escala k = 0.1
    float k = 0.2;
    outputClone *= k; // Multiplica os logits por k
    cv::exp(-outputClone, outputClone);
    outputClone = 1.0 / (1.0 + outputClone);

    // Aplica o threshold
    cv::Mat binaryOutput;
    cv::threshold(outputClone, binaryOutput, threshold, 1, cv::THRESH_BINARY);

    // Converte para exibição
    cv::Mat display;
    binaryOutput.convertTo(display, CV_8U, 255.0);
    return display;
}

// Thread só para capturar
void capture_thread(cv::VideoCapture& cap) {
    while (running) {
        cv::Mat frame;
        cap.grab();
        cap.read(frame);
        if (!frame.empty()) {
            std::lock_guard<std::mutex> lock(frame_mutex);
            latest_frame = frame.clone();
        }
    }
}

void displayROI(const cv::Mat& debug_image, const cv::Mat& binaryOutput) {
    // Cria uma cópia da debug_image para desenhar a ROI
    cv::Mat roiFrame = debug_image.clone();

    // Parâmetros da ROI (mesmos usados em isTouchingYellowLaneAndPublish)
    const int carLeftEdge = 151;
    const int carRightEdge = 361;
    const int checkHeight = 50;
    const int roiTop = 512 - checkHeight; // Linha inicial da ROI (462)

    // Garante que roiFrame está em BGR (debug_image já deve estar, mas verificamos)
    if (roiFrame.channels() == 1) {
        cv::cvtColor(roiFrame, roiFrame, cv::COLOR_GRAY2BGR);
    }

    // Desenha um retângulo ao redor da ROI inteira
    cv::rectangle(
        roiFrame,
        cv::Point(0, roiTop),           // Canto superior esquerdo (0, 462)
        cv::Point(511, 511),            // Canto inferior direito (511, 511)
        cv::Scalar(0, 255, 0),          // Cor verde em BGR
        2                               // Espessura da linha
    );

    // Desenha linhas verticais nas bordas do carro
    cv::line(
        roiFrame,
        cv::Point(carLeftEdge, roiTop), // Início da linha esquerda
        cv::Point(carLeftEdge, 511),    // Fim da linha esquerda
        cv::Scalar(0, 0, 255),          // Cor vermelha em BGR
        2                               // Espessura da linha
    );
    cv::line(
        roiFrame,
        cv::Point(carRightEdge, roiTop), // Início da linha direita
        cv::Point(carRightEdge, 511),    // Fim da linha direita
        cv::Scalar(0, 0, 255),           // Cor vermelha em BGR
        2                                // Espessura da linha
    );

    // Destaca pixels detectados como faixa na ROI
    for (int row = roiTop; row < 512; ++row) {
        if (binaryOutput.at<uchar>(row, carLeftEdge) == 255) {
            cv::circle(roiFrame, cv::Point(carLeftEdge, row), 3, cv::Scalar(255, 0, 0), -1); // Azul
        }
        if (binaryOutput.at<uchar>(row, carRightEdge) == 255) {
            cv::circle(roiFrame, cv::Point(carRightEdge, row), 3, cv::Scalar(255, 0, 0), -1); // Azul
        }
    }

    // Exibe a ROI em uma janela separada
    cv::imshow("ROI", roiFrame);
}

bool isTouchingYellowLaneAndPublish(const cv::Mat& binaryOutput) {
    const int carLeftEdge = 151;
    const int carRightEdge = 361;
    const int checkHeight = 200;

    for (int row = 512 - checkHeight; row < 512; ++row) {
        if (binaryOutput.at<uchar>(row, carLeftEdge) == 255) {
			// std::cout << "[DEBUG] LEFT lane touched. Publishing 76!" << std::endl;
            std::cout << "[DEBUG] RIGHT lane touched. Publishing 82!" << std::endl;
            publishLaneTouch(82);
            return true;
        }
        if (binaryOutput.at<uchar>(row, carRightEdge) == 255) {
			// std::cout << "[DEBUG] RIGHT lane touched. Publishing 82!" << std::endl;
            std::cout << "[DEBUG] LEFT lane touched. Publishing 76!" << std::endl;
            publishLaneTouch(76);
            return true;
        }
    }
	std::cout << "[DEBUG] Any lane touched. Publishing 0!" << std::endl;
	publishLaneTouch(0);
    return false;
}

// No main():
int main() {
    signal(SIGINT, signal_handler);
    initializeTensorRT();
    initMQTT();

    std::string pipeline = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=512, height=512, format=NV12, framerate=60/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmera!" << std::endl;
        return -1;
    }

    std::thread cam_thread(capture_thread, std::ref(cap));
    std::thread mqtt_thread(mqtt_loop_thread);
    jetracer.start();

    while (running) {
        cv::Mat frame_copy;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            if (!latest_frame.empty())
                frame_copy = latest_frame.clone();
        }

        if (frame_copy.empty())
            continue;

        cv::Mat debug_image; // Variável para armazenar a debug_image
        auto output = inferLaneNet(preprocess_frame(frame_copy, debug_image)); // Passa debug_image por referência
        cv::Mat model_vis_07 = visualizeOutput(output, 0.7);
        isTouchingYellowLaneAndPublish(model_vis_07);
        displayROI(debug_image, model_vis_07); // Usa debug_image para exibir a ROI

        cv::imshow("Model Output 0.7", model_vis_07);

        if (cv::waitKey(1) == 'q')
            break;
    }

    running = false;
    cam_thread.join();
    mqtt_thread.join();
    destroyTensorRT();
    jetracer.stop();
    cap.release();
    cv::destroyAllWindows();

    mosquitto_disconnect(mosq);
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();

    return 0;
}
