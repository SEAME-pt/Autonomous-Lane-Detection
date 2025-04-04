#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <memory>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    std::vector<char> buffer(size);
    file.seekg(0, std::ifstream::beg);
    file.read(buffer.data(), size);
    return buffer;
}

int main() {
    Logger logger;

    // Load engine
    std::vector<char> engine_data = readFile("lanenet.trt");
    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size(), nullptr);
    IExecutionContext* context = engine->createExecutionContext();

    // Allocate buffers
    int inputIndex = engine->getBindingIndex("images");
    int outputIndex = engine->getBindingIndex("output");

    Dims inputDims = engine->getBindingDimensions(inputIndex);  // (1,3,512,512)
    Dims outputDims = engine->getBindingDimensions(outputIndex);

    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; i++) inputSize *= inputDims.d[i];
    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; i++) outputSize *= outputDims.d[i];

    float* input_host = new float[inputSize];
    float* output_host = new float[outputSize];

    void* input_device;
    void* output_device;
    cudaMalloc(&input_device, inputSize * sizeof(float));
    cudaMalloc(&output_device, outputSize * sizeof(float));

    // Prepare input
    cv::Mat img = cv::imread("exemplo.jpg");
    cv::resize(img, img, cv::Size(512, 512));
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    // HWC to CHW
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    for (int c = 0; c < 3; ++c)
        memcpy(input_host + c * 512 * 512, channels[c].data, 512 * 512 * sizeof(float));

    // Copy to device
    cudaMemcpy(input_device, input_host, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    void* bindings[] = {input_device, output_device};
    context->executeV2(bindings);

    cudaMemcpy(output_host, output_device, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Inferência finalizada. Primeiro valor de saída: " << output_host[0] << std::endl;

    // Limpeza
    delete[] input_host;
    delete[] output_host;
    cudaFree(input_device);
    cudaFree(output_device);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

