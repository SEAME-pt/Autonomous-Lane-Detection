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

void signal_handler(int /*signal*/) {
    running = false;
    std::cout << "[INFO] Encerrando..." << std::endl;
}

// cv::Mat visualizeOutput(const std::vector<float>& output_data, float threshold) {
//     // Cria um cv::Mat de 512x512 com os dados do vetor (CV_32F)
//     cv::Mat outputMat(512, 512, CV_32F, const_cast<float*>(output_data.data()));
    
//     // Clona a matriz para não modificar os dados originais
//     cv::Mat outputClone = outputMat.clone();
    
//     // Aplica o threshold: pixels com valor > 0.5 se tornam 1, os demais 0
//     cv::Mat binaryOutput;
//     cv::threshold(outputClone, binaryOutput, threshold, 1, cv::THRESH_BINARY);
    
//     // Converte a imagem binarizada para 8 bits e escala para [0, 255] para exibição
//     cv::Mat display;
//     binaryOutput.convertTo(display, CV_8U, 255.0);
    
//     return display;
// }

cv::Mat visualizeOutput(const std::vector<float>& output_data, float threshold) {
    cv::Mat outputMat(512, 512, CV_32F, const_cast<float*>(output_data.data()));
    cv::Mat outputClone = outputMat.clone();
    
    // Aplica sigmoide com fator de escala k = 0.1
    float k = 0.1;
    outputClone *= k; // Multiplica os logits por k
    cv::exp(-outputClone, outputClone);
    outputClone = 1.0 / (1.0 + outputClone);
    
    // Exibe mínimo e máximo
    double min_val, max_val;
    cv::minMaxLoc(outputClone, &min_val, &max_val);
    std::cout << "Sigmoid Output range (threshold " << threshold << "): " 
              << min_val << " to " << max_val << std::endl;
    
    // Aplica o threshold
    cv::Mat binaryOutput;
    cv::threshold(outputClone, binaryOutput, threshold, 1, cv::THRESH_BINARY);
    
    // Converte para exibição
    cv::Mat display;
    binaryOutput.convertTo(display, CV_8U, 255.0);
    return display;
}

// ==================================================================
// Função principal simplificada
// ==================================================================
int main() {
    signal(SIGINT, signal_handler);
    initializeTensorRT();

  
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

        // No main(), após a inferência
        auto output = inferLaneNet(frame);

        // Adiciona depuração para inspecionar o intervalo de valores
        // double min_val, max_val;  // Alterado de float para double
        // cv::minMaxLoc(cv::Mat(512, 512, CV_32F, output.data()), &min_val, &max_val);
        // std::cout << "Output range: " << min_val << " to " << max_val << std::endl;

        cv::Mat model_vis_04 = visualizeOutput(output, float(0.4));
        cv::Mat model_vis_03  = visualizeOutput(output, float(0.3));
        cv::Mat model_vis_06  = visualizeOutput(output, float(0.6));
        cv::Mat model_vis_05  = visualizeOutput(output, float(0.5));
        

        cv::imshow("Camera", frame);
        cv::imshow("Model Output 0.4", model_vis_04);
        cv::imshow("Model Output 0.3", model_vis_03);
        cv::imshow("Model Output 0.6", model_vis_06);
        cv::imshow("Model Output 0.5", model_vis_05);

        if (cv::waitKey(1) == 'q')
            break;
    }

    destroyTensorRT();
    // jetracer.stop();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
