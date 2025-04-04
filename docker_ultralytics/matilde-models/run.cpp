#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace std;

// Caminho para o modelo TensorRT
const string ENGINE_PATH = "lenenet.trt";
const int INPUT_H = 640;
const int INPUT_W = 640;
const int OUTPUT_SIZE = 25200 * 85;  // Tamanho da saída do YOLOv5n
const float CONF_THRESHOLD = 0.3;    // Limiar de confiança (diminuído para detectar mais objetos)
const float NMS_THRESHOLD = 0.4;     // Non-Maximum Suppression

// Classe Logger para TensorRT
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) cout << msg << endl;
    }
};

// Função para carregar o modelo TensorRT
ICudaEngine* loadEngine(const string& enginePath, Logger& logger) {
    ifstream file(enginePath, ios::binary);
    if (!file.good()) {
        cerr << "Erro ao abrir o arquivo: " << enginePath << endl;
        return nullptr;
    }

    file.seekg(0, ios::end);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(logger);
    return runtime->deserializeCudaEngine(buffer.data(), size);
}

// Função para converter imagem para formato CHW (TensorRT exige isso)
void preprocessImage(cv::Mat& img, float* inputData) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);  // Normaliza para [0, 1]

    // Convertendo BGR para CHW
    vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    int index = 0;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < INPUT_H; i++) {
            for (int j = 0; j < INPUT_W; j++) {
                inputData[index++] = channels[c].at<float>(i, j);
            }
        }
    }
}

// Função para desenhar caixas de detecção
void drawDetections(cv::Mat& img, const vector<cv::Rect>& boxes, const vector<float>& confidences, const vector<int>& class_ids) {
    for (size_t i = 0; i < boxes.size(); i++) {
        cv::rectangle(img, boxes[i], cv::Scalar(0, 255, 0), 2);
        string label = "Class " + to_string(class_ids[i]) + " - " + to_string(confidences[i]);
        cv::putText(img, label, boxes[i].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }
}

// Função de inferência e pós-processamento
void doInference(ICudaEngine* engine, cv::Mat& img) {
    IExecutionContext* context = engine->createExecutionContext();

    void* buffers[2];
    float* input = new float[3 * INPUT_H * INPUT_W]; // Entrada em CHW
    float* output = new float[OUTPUT_SIZE];

    cudaMalloc(&buffers[0], 3 * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&buffers[1], OUTPUT_SIZE * sizeof(float));

    preprocessImage(img, input);
    cudaMemcpy(buffers[0], input, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice);

    context->executeV2(buffers);

    cudaMemcpy(output, buffers[1], OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Processando as saídas
    vector<cv::Rect> boxes;
    vector<float> confidences;
    vector<int> class_ids;

    for (int i = 0; i < OUTPUT_SIZE; i += 85) {
        float confidence = output[i + 4];
        if (confidence > CONF_THRESHOLD) {
            int class_id = max_element(output + i + 5, output + i + 85) - (output + i + 5);

            float x = output[i];
            float y = output[i + 1];
            float w = output[i + 2];
            float h = output[i + 3];

            int left = static_cast<int>((x - w / 2) * img.cols);
            int top = static_cast<int>((y - h / 2) * img.rows);
            int width = static_cast<int>(w * img.cols);
            int height = static_cast<int>(h * img.rows);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(confidence);
            class_ids.push_back(class_id);
        }
    }

    // Non-Maximum Suppression (NMS) para remover detecções duplicadas
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    vector<cv::Rect> final_boxes;
    vector<float> final_confidences;
    vector<int> final_class_ids;

    for (int idx : indices) {
        final_boxes.push_back(boxes[idx]);
        final_confidences.push_back(confidences[idx]);
        final_class_ids.push_back(class_ids[idx]);
    }

    drawDetections(img, final_boxes, final_confidences, final_class_ids);

    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete[] input;
    delete[] output;
    delete context;
}

int main() {
    Logger logger;
    ICudaEngine* engine = loadEngine(ENGINE_PATH, logger);
    if (!engine) {
        cerr << "Falha ao carregar o modelo TensorRT!" << endl;
        return -1;
    }

    // Configuração da câmera para Jetson Nano
    cv::VideoCapture cap("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        cerr << "Erro ao abrir a câmera!" << endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        doInference(engine, frame);

        cv::imshow("YOLOv5 TensorRT", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    delete engine;

    return 0;
}

