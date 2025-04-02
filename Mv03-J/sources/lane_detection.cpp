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

// // Detecção simples da faixa a partir da máscara
// LaneInfo detect_lane(const cv::Mat& mask, const cv::Size& original_size) {
//     LaneInfo lane{};
//     cv::Mat resized_mask;
//     cv::resize(mask, resized_mask, original_size);
//     cv::threshold(resized_mask, resized_mask, 0.5, 1, cv::THRESH_BINARY);
//     auto M = cv::moments(resized_mask, true);
//     if (M.m00 > original_size.width * original_size.height * 0.01) {
//         lane.center = {float(M.m10 / M.m00), float(M.m01 / M.m00)};
//         lane.detected = true;
//     } else {
//         lane.detected = false;
//     }
//     return lane;
// }

void signal_handler(int /*signal*/) {
    running = false;
    std::cout << "[INFO] Encerrando..." << std::endl;
}

// cv::Mat visualizeOutput(const std::vector<float>& output_data) {
//     // Cria um cv::Mat de 512x512 com os dados do vetor (CV_32F)
//     cv::Mat outputMat(512, 512, CV_32F, const_cast<float*>(output_data.data()));
    
//     // Clona a matriz para ter uma cópia independente dos dados
//     cv::Mat outputClone = outputMat.clone();
    
//     // Converte os valores para a faixa [0, 255] e muda o tipo para 8 bits (CV_8U)
//     cv::Mat display;
//     outputClone.convertTo(display, CV_8U, 255.0);
    
//     return display;
// }

cv::Mat visualizeOutput(const std::vector<float>& output_data) {
    // Cria um cv::Mat de 512x512 com os dados do vetor (CV_32F)
    cv::Mat outputMat(512, 512, CV_32F, const_cast<float*>(output_data.data()));
    
    // Clona a matriz para não modificar os dados originais
    cv::Mat outputClone = outputMat.clone();
    
    // Aplica o threshold: pixels com valor > 0.5 se tornam 1, os demais 0
    cv::Mat binaryOutput;
    cv::threshold(outputClone, binaryOutput, 0.1, 1, cv::THRESH_BINARY);
    
    // Converte a imagem binarizada para 8 bits e escala para [0, 255] para exibição
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

        // Executa a inferência
        auto output = inferLaneNet(frame);
        // Converte a saída em imagem
        cv::Mat model_vis = visualizeOutput(output);
        // Redimensiona a imagem do modelo para o mesmo tamanho da câmera (opcional)
        cv::resize(model_vis, model_vis, frame.size());

        // Exibe janelas separadas: uma para a câmera, outra para a saída do modelo
        cv::imshow("Camera", frame);
        cv::imshow("Model Output", model_vis);

        if (cv::waitKey(1) == 'q')
            break;
    }

    destroyTensorRT();
    // jetracer.stop();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
