// main.cpp
#include "jetracer.hpp"
#include <thread>
#include <chrono>

JetCar* car_ptr = nullptr;

void signal_handler(int) {
    if (car_ptr) {
        car_ptr->stop();
    }
}

int main() {
    try {
        JetCar car(0x40, 0x60); // Passe os endereços servo_addr e motor_addr
        car_ptr = &car;
        signal(SIGINT, signal_handler);
        car.start();
        while (car.is_running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Gamepad error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

/*
#include "jetracer.hpp"

JetCar* car_ptr = nullptr;

void signal_handler(int) { // Remover o nome do parâmetro para evitar o aviso de parâmetro não utilizado
    if (car_ptr) {
        car_ptr->stop();
    }
}

int main() {
    try {
        JetCar car;
        car_ptr = &car;
        signal(SIGINT, signal_handler); // Remover o namespace std
        car.start();
        while (car.is_running_()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Gamepad error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
*/
