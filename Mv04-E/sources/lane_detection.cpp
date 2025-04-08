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
#include <mosquitto.h> // MQTT

using namespace nvinfer1;

// MQTT
#define LOCAL_BROKER "10.21.221.67"
#define LOCAL_PORT 1883
#define MQTT_TOPIC "jetracer/lane_touch"

struct mosquitto *mosq;

void initMQTT() {
    mosquitto_lib_init();
    mosq = mosquitto_new("lane_publisher", true, NULL);
    if (!mosq) throw std::runtime_error("Erro ao criar cliente MQTT");
    if (mosquitto_connect(mosq, LOCAL_BROKER, LOCAL_PORT, 60) != MOSQ_ERR_SUCCESS) {
        throw std::runtime_error("Erro ao conectar ao broker MQTT");
    }
    std::cout << "[MQTT] Conectado ao broker MQTT!" << std::endl;
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

JetCar jetracer(0x40, 0x60);
std::atomic<bool> running(true);

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

IRuntime* runtime = nullptr;
ICudaEngine* engine = nullptr;
IExecutionContext* context = nullptr;
void* buffers[2];

const int INPUT_SIZE = 3 * 512 * 512;
const int OUTPUT_SIZE = 1 * 512 * 512;

std::vector<char> loadTRTModel(const std::string& modelPath) {
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("Falha ao abrir arquivo TRT.");
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

void initializeTensorRT() {
    initLibNvInferPlugins(&gLogger, "");
    auto modelData = loadTRTModel("lanenet.trt");
    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
    context = engine->createExecutionContext();
    cudaMalloc(&buffers[0], INPUT_SIZE * sizeof(float));
    cudaMalloc(&buffers[1], OUTPUT_SIZE * sizeof(float));
}

void destroyTensorRT() {
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete context;
    delete engine;
    delete runtime;
}

cv::Mat preprocess_frame(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(512, 512));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    std::vector<cv::Point2f> src_points = { {0, 512}, {512, 512}, {0, 0}, {512, 0} };
    std::vector<cv::Point2f> dst_points = { {128, 512}, {384, 512}, {0, 0}, {512, 0} };
    cv::Mat transform = cv::getPerspectiveTransform(src_points, dst_points);
    cv::Mat warped;
    cv::warpPerspective(resized, warped, transform, cv::Size(512, 512));
    cv::Mat debug_image;
    cv::cvtColor(warped, debug_image, cv::COLOR_RGB2BGR);
    debug_image.convertTo(debug_image, CV_8U, 255.0);
    cv::imshow("Transformed Image", debug_image);
    return warped;
}

std::vector<float> inferLaneNet(const cv::Mat& frame) {
    cv::Mat input_tensor = preprocess_frame(frame);
    float chw_input[INPUT_SIZE];
    int idx = 0;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 512; ++i)
            for (int j = 0; j < 512; ++j)
                chw_input[idx++] = input_tensor.at<cv::Vec3f>(i, j)[c];
    cudaMemcpy(buffers[0], chw_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
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
    float k = 0.2;
    outputClone *= k;
    cv::exp(-outputClone, outputClone);
    outputClone = 1.0 / (1.0 + outputClone);
    cv::Mat binaryOutput;
    cv::threshold(outputClone, binaryOutput, threshold, 1, cv::THRESH_BINARY);
    cv::Mat display;
    binaryOutput.convertTo(display, CV_8U, 255.0);
    return display;
}

bool isTouchingYellowLaneAndPublish(const cv::Mat& binaryOutput) {
    const int carLeftEdge = 151;
    const int carRightEdge = 361;
    const int checkHeight = 50;

    for (int row = 512 - checkHeight; row < 512; ++row) {
        if (binaryOutput.at<uchar>(row, carLeftEdge) == 255) {
            std::cout << "[DEBUG] Faixa ESQUERDA tocada!" << std::endl;
            publishLaneTouch(76);
            return true;
        }
        if (binaryOutput.at<uchar>(row, carRightEdge) == 255) {
            std::cout << "[DEBUG] Faixa DIREITA tocada!" << std::endl;
            publishLaneTouch(82);
            return true;
        }
		// Printar que o carro não está tocando a faixa
		if (binaryOutput.at<uchar>(row, carLeftEdge) == 0 && binaryOutput.at<uchar>(row, carRightEdge) == 0) {
			std::cout << "[DEBUG] Carro NÃO tocando a faixa!" << std::endl;
			publishLaneTouch(0);
			return false;
		}
    }
    return false;
}

int main() {
    signal(SIGINT, signal_handler);
    initializeTensorRT();
    initMQTT();

    std::string pipeline =
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=512, height=512, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmera!" << std::endl;
        return -1;
    }

    while (running) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty())
            break;

        auto output = inferLaneNet(frame);

        cv::Mat model_vis_04 = visualizeOutput(output, float(0.4));
        isTouchingYellowLaneAndPublish(model_vis_04);

        cv::imshow("Camera", frame);
        cv::imshow("Model Output 0.4", model_vis_04);

        if (cv::waitKey(1) == 'q')
            break;
    }

    destroyTensorRT();
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}




// #include <opencv2/opencv.hpp>
// #include "jetracer.hpp"
// #include "NvInfer.h"
// #include "cuda_runtime_api.h"
// #include <fstream>
// #include <vector>
// #include <atomic>
// #include <csignal>
// #include <cmath>
// #include <memory>
// #include "NvInferPlugin.h"

// using namespace nvinfer1;

// JetCar jetracer(0x40, 0x60);
// std::atomic<bool> running(true);

// // Logger simples para TensorRT
// class Logger : public ILogger {
// public:
//     void log(Severity severity, const char* msg) noexcept override {
//         if (severity <= Severity::kWARNING)
//             std::cout << "[TensorRT] " << msg << std::endl;
//     }
// } gLogger;

// // Variáveis TensorRT
// IRuntime* runtime = nullptr;
// ICudaEngine* engine = nullptr;
// IExecutionContext* context = nullptr;
// void* buffers[2];

// const int INPUT_SIZE = 3 * 512 * 512;
// const int OUTPUT_SIZE = 1 * 512 * 512;

// // Carrega modelo TRT serializado
// std::vector<char> loadTRTModel(const std::string& modelPath) {
//     std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
//     if (!file.is_open()) throw std::runtime_error("Falha ao abrir arquivo TRT.");
//     size_t size = file.tellg();
//     file.seekg(0, std::ios::beg);
//     std::vector<char> buffer(size);
//     file.read(buffer.data(), size);
//     return buffer;
// }

// // Inicializa TensorRT
// void initializeTensorRT() {
//     initLibNvInferPlugins(&gLogger, "");
//     auto modelData = loadTRTModel("lanenet.trt");
//     runtime = createInferRuntime(gLogger);
//     engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
//     context = engine->createExecutionContext();
//     cudaMalloc(&buffers[0], INPUT_SIZE * sizeof(float));
//     cudaMalloc(&buffers[1], OUTPUT_SIZE * sizeof(float));
// }

// // Libera recursos TensorRT
// void destroyTensorRT() {
//     cudaFree(buffers[0]);
//     cudaFree(buffers[1]);
//     delete context;
//     delete engine;
//     delete runtime;
// }

// // // Pré-processa imagem
// // cv::Mat preprocess_frame(const cv::Mat& frame) {
// //     cv::Mat resized;
// //     cv::resize(frame, resized, cv::Size(512, 512));
// //     resized.convertTo(resized, CV_32F, 1.0 / 255.0);
// //     cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
// //     return resized;
// // }

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

// // Inferência do modelo TensorRT
// std::vector<float> inferLaneNet(const cv::Mat& frame) {
//     cv::Mat input_tensor = preprocess_frame(frame);
//     float chw_input[INPUT_SIZE]; // [3 * 512 * 512]
//     int idx = 0;
// for (int c = 0; c < 3; ++c) {
//     for (int i = 0; i < 512; ++i) {
//         for (int j = 0; j < 512; ++j) {
//             chw_input[idx++] = input_tensor.at<cv::Vec3f>(i, j)[c];
//         }
//     }
// }
// cudaMemcpy(buffers[0], chw_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

//     //cudaMemcpy(buffers[0], input_tensor.data, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
//     context->executeV2(buffers);
//     std::vector<float> output_data(OUTPUT_SIZE);
//     cudaMemcpy(output_data.data(), buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
//     return output_data;
// }

// void signal_handler(int /*signal*/) {
//     running = false;
//     std::cout << "[INFO] Encerrando..." << std::endl;
// }

// cv::Mat visualizeOutput(const std::vector<float>& output_data, float threshold) {
//     cv::Mat outputMat(512, 512, CV_32F, const_cast<float*>(output_data.data()));
//     cv::Mat outputClone = outputMat.clone();

//     // Aplica sigmoide com fator de escala k = 0.1
//     float k = 0.2;
//     outputClone *= k; // Multiplica os logits por k
//     cv::exp(-outputClone, outputClone);
//     outputClone = 1.0 / (1.0 + outputClone);

//     // Exibe mínimo e máximo
//     // double min_val, max_val;
//     // cv::minMaxLoc(outputClone, &min_val, &max_val);
//     // std::cout << "Sigmoid Output range (threshold " << threshold << "): "
//     //           << min_val << " to " << max_val << std::endl;

//     // Aplica o threshold
//     cv::Mat binaryOutput;
//     cv::threshold(outputClone, binaryOutput, threshold, 1, cv::THRESH_BINARY);

//     // Converte para exibição
//     cv::Mat display;
//     binaryOutput.convertTo(display, CV_8U, 255.0);
//     return display;
// }

// bool isTouchingYellowLane(const cv::Mat& binaryOutput) {
//     const int carLeftEdge = 151;  // Borda esquerda do carro em pixels
//     const int carRightEdge = 361; // Borda direita do carro em pixels
//     const int checkHeight = 50;   // Verificar os últimos 50 pixels da imagem (parte inferior)

//     for (int row = 512 - checkHeight; row < 512; ++row) {
//         if (binaryOutput.at<uchar>(row, carLeftEdge) == 255) {
//             std::cout << "[DEBUG] Carro tocando a faixa amarela na borda ESQUERDA!" << std::endl;
//             return true;
//         }
//         if (binaryOutput.at<uchar>(row, carRightEdge) == 255) {
//             std::cout << "[DEBUG] Carro tocando a faixa amarela na borda DIREITA!" << std::endl;
//             return true;
//         }
//     }
//     return false;
// }

// // ==================================================================
// // Função principal simplificada
// // ==================================================================
// int main() {
//     signal(SIGINT, signal_handler);
//     initializeTensorRT();

//     std::string pipeline =
//         "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=512, height=512, format=NV12, framerate=30/1 ! "
//         "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink";

//     cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
//     if (!cap.isOpened()) {
//         std::cerr << "Erro ao abrir a câmera!" << std::endl;
//         return -1;
//     }

//     while (running) {
//         cv::Mat frame;
//         if (!cap.read(frame) || frame.empty())
//             break;

//         auto output = inferLaneNet(frame);

//         cv::Mat model_vis_04 = visualizeOutput(output, float(0.4));
//         cv::Mat model_vis_03 = visualizeOutput(output, float(0.3));
//         cv::Mat model_vis_05 = visualizeOutput(output, float(0.5));
//         cv::Mat model_vis_038 = visualizeOutput(output, float(0.38));
//         cv::Mat model_vis_07 = visualizeOutput(output, float(0.7));

//         // Verificar se o carro está tocando a faixa amarela (usando threshold 0.4 como exemplo)
//         isTouchingYellowLane(model_vis_04);

//         cv::imshow("Camera", frame);
//         cv::imshow("Model Output 0.4", model_vis_04);
//         cv::imshow("Model Output 0.3", model_vis_03);
//         cv::imshow("Model Output 0.5", model_vis_05);
//         cv::imshow("Model Output 0.38", model_vis_038);
//         cv::imshow("Model Output 0.7", model_vis_07);

//         if (cv::waitKey(1) == 'q')
//             break;
//     }

//     destroyTensorRT();
//     cap.release();
//     cv::destroyAllWindows();
//     return 0;
// }
