#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <csignal>
#include <atomic>
#include <thread>

std::atomic<bool> running_video(true);

// Sinal para parar com Ctrl+C
void signal_handler(int) {
    running_video = false;
}

int main() {
    signal(SIGINT, signal_handler);

    // Pipeline da câmera
    std::string pipeline =
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=512, height=512, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink";

    // Abre câmera
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmera!" << std::endl;
        return -1;
    }

    // Reduz buffer da câmera
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    // Gera nome de arquivo com timestamp
    auto timestamp = std::time(nullptr);
    std::stringstream filename;
    filename << "trajeto_" << timestamp << ".mp4";

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = 30;

    // Abre vídeo para gravação
    cv::VideoWriter writer(filename.str(), cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Erro ao criar arquivo de vídeo!" << std::endl;
        return -1;
    }

    std::cout << "Gravando vídeo em: " << filename.str() << std::endl;

    // Loop principal de gravação
    cv::Mat frame;
    while (running_video) {
        cap >> frame;
        if (frame.empty()) continue;

        frame = frame.clone(); // garante cópia leve
        writer.write(frame);

        // Delay mínimo para não travar sistema
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    cap.release();
    writer.release();

    std::cout << "Gravação encerrada." << std::endl;
    return 0;
}
