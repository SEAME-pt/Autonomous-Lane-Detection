#include "jetracer.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <csignal>
#include <ctime>
#include <sstream>

JetCar* car_ptr = nullptr;
std::atomic<bool> running_video{true};

// Sinal para Ctrl+C
void signal_handler(int) {
    if (car_ptr) car_ptr->stop();
    running_video = false;
}

// Função que grava e exibe vídeo
void video_thread_func() {
    std::string pipeline =
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=512, height=512, format=NV12, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir a câmera!" << std::endl;
        return;
    }

    auto timestamp = std::time(nullptr);
    std::stringstream filename;
    filename << "trajeto_" << timestamp << ".mp4";

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = 30;

    cv::VideoWriter writer(filename.str(), cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Erro ao criar arquivo de vídeo!" << std::endl;
        return;
    }

    std::cout << "Gravando vídeo em: " << filename.str() << std::endl;
    cv::namedWindow("Trajeto", cv::WINDOW_NORMAL);
    cv::resizeWindow("Trajeto", 512, 512);

    cv::Mat frame;
    while (running_video) {
        cap >> frame;
        if (frame.empty()) break;

        writer.write(frame);
        cv::imshow("Trajeto", frame);

        if (cv::waitKey(1) == 'q') {
            running_video = false;
            if (car_ptr) car_ptr->stop();
            break;
        }
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    std::cout << "Gravação encerrada." << std::endl;
}

// Main
int main() {
    try {
        JetCar car(0x40, 0x60);
        car_ptr = &car;
        signal(SIGINT, signal_handler);

        // Inicia carro (joystick em thread separada)
        car.start();

        // Inicia thread de vídeo
        std::thread video_thread(video_thread_func);
        video_thread.join();  // Espera vídeo terminar

    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

