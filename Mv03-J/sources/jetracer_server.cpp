// jetracer_server.cpp (modo autônomo exclusivo)
#include "jetracer.hpp"
#include <iostream>
#include <thread>
#include <mutex>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <atomic>
#include <csignal>

#define PORT 65432

JetCar jetracer(0x40, 0x60);
std::atomic<bool> running(true);

void handle_autonomous_input() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen failed");
        exit(EXIT_FAILURE);
    }

    std::cout << "Servidor autônomo aguardando comandos na porta " << PORT << std::endl;

    while (running) {
        if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
            if (!running) break;
            perror("accept failed");
            continue;
        }

        memset(buffer, 0, sizeof(buffer));

        ssize_t valread = read(new_socket, buffer, sizeof(buffer));
        if (valread <= 0) {
            std::cerr << "Erro ao ler dados do socket." << std::endl;
            close(new_socket);
            continue;
        }

        float angle, speed;
        if (sscanf(buffer, "%f %f", &angle, &speed) == 2) {
            std::cout << "Comando autônomo recebido: Angulo=" << angle << " Speed=" << speed << std::endl;
            jetracer.set_steering(static_cast<int>(angle));
            jetracer.set_speed(speed);
        } else {
            std::cerr << "Falha ao interpretar comando: " << buffer << std::endl;
        }

        close(new_socket);
    }

    close(server_fd);
    std::cout << "[INFO] Servidor encerrado." << std::endl;
}

int main() {
    std::thread autonomous_thread(handle_autonomous_input);

    jetracer.start();

    autonomous_thread.join();

    jetracer.stop();

    return 0;
}

