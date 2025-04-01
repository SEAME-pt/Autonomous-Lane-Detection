# JetCar Project

## Overview

The **JetCar** project is a C++ library designed to control a robotic car using servo motors, DC motors, and a joystick. The project integrates various hardware components via the I2C protocol and utilizes SDL2 for joystick support.

Key features include:
- Servo motor control for steering.
- DC motor control for speed.
- Joystick integration for real-time control.
- Threaded execution for concurrent processing.
- Graceful handling of signals for safe shutdown.

---

## Files in the Project

### 1. `jetracer.hpp`
This header file declares the `JetCar` class, encapsulating the logic to interface with servos, motors, and joysticks.

### 2. `jetracer.cpp`
Implements the functionality defined in `jetracer.hpp`.

### 3. `main.cpp`
Contains the main application to run the JetCar control system. Handles signal interruptions to stop the car gracefully.

---

## Requirements

### Hardware:
- **Servo Motor** for steering.
- **DC Motors** for driving.
- **Joystick** for user input.
- **I2C-compatible Controller** for motor and servo control (e.g., PCA9685).

### Software:
- C++17 or later.
- SDL2 library.
- Linux-based system with I2C support.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd JetCar
   ```
2. Install dependencies:
   ```bash
   sudo apt-get install libsdl2-dev
   ```
3. Compile the project using `g++`:
   ```bash
   g++ -std=c++17 -o jetracer main.cpp jetracer.cpp -lSDL2 -pthread
   ```

---

## Usage

1. Ensure the I2C bus is enabled on your system.
2. Run the compiled program:
   ```bash
   ./jetracer
   ```
3. Use a joystick to control the car's steering and speed.

---

## Class Details

### JetCar
#### Constructor:
```cpp
JetCar(int servo_addr_ess = 0x40, int motor_addr_ess = 0x60);
```
- Initializes the servo and motor controllers via I2C.
- Default I2C addresses are 0x40 (servo) and 0x60 (motor).

#### Public Methods:
1. **`void init_servo();`**
   - Configures the servo controller.

2. **`void init_motors();`**
   - Configures the motor controller.

3. **`void set_steering(int angle);`**
   - Adjusts the steering angle (range: -180 to 180 degrees).

4. **`void set_speed(float speed);`**
   - Sets the motor speed (-100 to 100%).

5. **`void process_joystick();`**
   - Reads joystick input for speed and steering.

6. **`void start();`**
   - Starts the car control system in a separate thread.

7. **`void stop();`**
   - Stops the car safely.

8. **`bool is_running_() const;`**
   - Checks if the system is running.

#### Private Methods:
- Helper functions for I2C communication:
  - `write_byte` / `read_byte`
  - `write_byte_data` / `read_byte_data`

---

## Example Code

```cpp
#include "jetracer.hpp"

JetCar* car_ptr = nullptr;

void signal_handler(int) {
    if (car_ptr) {
        car_ptr->stop();
    }
}

int main() {
    try {
        JetCar car;
        car_ptr = &car;
        signal(SIGINT, signal_handler);
        car.start();
        while (car.is_running_()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

---

## Notes

- Ensure correct wiring of the servo and motor controllers to the I2C bus.
- Modify the I2C addresses in the constructor if using different hardware.
- Use a reliable joystick for smooth control.


---

# Projeto JetCar

## Visão Geral

O projeto **JetCar** é uma biblioteca em C++ concebida para controlar um carro robótico utilizando servomotores, motores DC e um joystick. O projeto integra diversos componentes de hardware via o protocolo I2C e utiliza a biblioteca SDL2 para suporte ao joystick.

Principais funcionalidades incluem:
- Controlo de servomotor para direcção.
- Controlo de motores DC para velocidade.
- Integração de joystick para controlo em tempo real.
- Execução com threads para processamento concorrente.
- Gestão de sinais para encerramento seguro.

---

## Ficheiros no Projeto

### 1. `jetracer.hpp`
Este ficheiro de cabeçalho declara a classe `JetCar`, que encapsula a lógica para interagir com os servos, motores e joystick.

### 2. `jetracer.cpp`
Implementa a funcionalidade definida em `jetracer.hpp`.

### 3. `main.cpp`
Contém a aplicação principal que executa o sistema de controlo JetCar. Garante o encerramento seguro do carro através do tratamento de sinais.

---

## Requisitos

### Hardware:
- **Servomotor** para controlo da direcção.
- **Motores DC** para movimentação.
- **Joystick** para entrada de utilizador.
- **Controlador compatível com I2C** para controlo de motores e servos (e.g., PCA9685).

### Software:
- C++17 ou superior.
- Biblioteca SDL2.
- Sistema operativo baseado em Linux com suporte a I2C.

---

## Instalação

1. Clone o repositório:
   ```bash
   git clone <repository-url>
   cd JetCar
   ```
2. Instale as dependências:
   ```bash
   sudo apt-get install libsdl2-dev
   ```
3. Compile o projeto utilizando `g++`:
   ```bash
   g++ -std=c++17 -o jetracer main.cpp jetracer.cpp -lSDL2 -pthread
   ```

---

## Utilização

1. Certifique-se de que o barramento I2C está ativado no seu sistema.
2. Execute o programa compilado:
   ```bash
   ./jetracer
   ```
3. Utilize o joystick para controlar a direcção e a velocidade do carro.

---

## Detalhes da Classe

### JetCar
#### Construtor:
```cpp
JetCar(int servo_addr_ess = 0x40, int motor_addr_ess = 0x60);
```
- Inicializa os controladores de servo e motor via I2C.
- Endereços I2C padrão: 0x40 (servo) e 0x60 (motor).

#### Métodos Públicos:
1. **`void init_servo();`**
   - Configura o controlador do servomotor.

2. **`void init_motors();`**
   - Configura o controlador dos motores DC.

3. **`void set_steering(int angle);`**
   - Ajusta o ângulo de direcção (intervalo: -180 a 180 graus).

4. **`void set_speed(float speed);`**
   - Define a velocidade do motor (-100 a 100%).

5. **`void process_joystick();`**
   - Lê a entrada do joystick para controlo de velocidade e direcção.

6. **`void start();`**
   - Inicia o sistema de controlo do carro numa thread separada.

7. **`void stop();`**
   - Pára o carro de forma segura.

8. **`bool is_running_() const;`**
   - Verifica se o sistema está em execução.

#### Métodos Privados:
- Funções auxiliares para comunicação I2C:
  - `write_byte` / `read_byte`
  - `write_byte_data` / `read_byte_data`

---

## Exemplo de Código

```cpp
#include "jetracer.hpp"

JetCar* car_ptr = nullptr;

void signal_handler(int) {
    if (car_ptr) {
        car_ptr->stop();
    }
}

int main() {
    try {
        JetCar car;
        car_ptr = &car;
        signal(SIGINT, signal_handler);
        car.start();
        while (car.is_running_()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

---

## Notas

- Certifique-se de que a cablagem dos controladores de servo e motor está correta no barramento I2C.
- Modifique os endereços I2C no construtor se estiver a utilizar hardware diferente.
- Utilize um joystick fiável para um controlo suave.

---
