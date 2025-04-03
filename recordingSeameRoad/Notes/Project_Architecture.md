# JetCar Project Architecture

## Overview
The JetCar project architecture consists of several components that work together to control a robotic car using servos, DC motors, and a joystick. The primary focus of the architecture is to manage hardware interfaces via the I2C protocol, handle joystick inputs using SDL2, and provide a responsive control mechanism.

---

## Architectural Components

### 1. Core Logic
**`JetCar` Class**
- The core class of the project encapsulates the logic for interacting with servos, motors, and joysticks.
- Responsibilities include:
  - Configuring servo and motor controllers.
  - Setting steering and speed.
  - Processing joystick input for real-time control.
  - Managing multi-threaded execution.

### 2. Hardware Interface
**I2C Communication**
- The project uses the I2C protocol to communicate with the servo and motor controllers.
- Key functions:
  - Reading and writing bytes to/from I2C devices.
  - Setting PWM values for precise motor and servo control.

**Servo and Motor Controllers**
- Servo: Controls the car's steering mechanism.
- Motor: Drives the car's wheels and sets its speed and direction.

### 3. Input System
**Joystick Integration**
- SDL2 library is used to handle joystick input.
- Real-time joystick updates are processed in a dedicated thread to adjust speed and steering.

### 4. Concurrency Management
**Threaded Execution**
- The joystick input is processed in a separate thread to ensure the main application loop remains responsive.
- The `std::thread` library is used for threading.

### 5. Signal Handling
**Graceful Shutdown**
- Signal handling is implemented to stop the car safely when the program is interrupted (e.g., `Ctrl+C`).
- Ensures hardware is reset to a safe state (e.g., motors stopped, steering centered).

---

## Layered Architecture
The project follows a layered architecture:

### Layer 1: Hardware Abstraction Layer (HAL)
- **Purpose:** Abstracts low-level hardware interactions, such as I2C communication and PWM control.
- **Components:**
  - `write_byte`
  - `read_byte`
  - `write_byte_data`
  - `read_byte_data`

### Layer 2: Control Logic
- **Purpose:** Implements the car's control functionality, including speed and steering management.
- **Components:**
  - `init_servo`
  - `init_motors`
  - `set_speed`
  - `set_steering`

### Layer 3: Input Handling
- **Purpose:** Processes user input from the joystick for real-time control.
- **Components:**
  - `process_joystick`

### Layer 4: Application Layer
- **Purpose:** Provides the entry point and manages the overall system lifecycle.
- **Components:**
  - `main.cpp`
  - Signal handling for safe shutdown.

---

## System Workflow
1. **Initialization**
   - I2C communication is set up for servo and motor controllers.
   - Servo and motor controllers are configured.

2. **Joystick Input Processing**
   - Joystick input is continuously read in a dedicated thread.
   - Speed and steering are adjusted based on joystick input.

3. **Execution**
   - The car runs in a loop, responding to joystick input and maintaining state.

4. **Shutdown**
   - On signal interruption (e.g., `SIGINT`), the system stops the car safely by resetting speed and steering.

---

## Interaction Diagram

```
+-------------------+         +----------------------+         +--------------------+
| Application Layer |         | Control Logic Layer |         | Hardware Interface |
|   (main.cpp)      |         |    (JetCar Class)   |         |       (I2C)        |
+-------------------+         +----------------------+         +--------------------+
        |                              |                              |
        |                              |                              |
        +------------------------------>------------------------------>
                             System Flow
```

---

## Future Improvements
- Add support for additional input devices (e.g., keyboard, game controllers).
- Implement diagnostic logging for debugging hardware issues.
- Optimize the use of multi-threading for better performance.

---

## Conclusion
The JetCar project architecture is modular, scalable, and designed to provide precise control over a robotic car. By separating hardware interaction, control logic, and user input, the architecture ensures a clean and maintainable codebase.


---


# Arquitetura do Projeto JetCar

## Visão Geral

A arquitetura do projeto JetCar consiste em diversos componentes que trabalham juntos para controlar um carro robótico usando servos, motores DC e um joystick. O principal foco da arquitetura é gerenciar interfaces de hardware via o protocolo I2C, lidar com entradas do joystick usando SDL2 e fornecer um mecanismo de controle responsivo.

---

## Componentes da Arquitetura

### 1. Lógica Central

**Classe `JetCar`**

- A classe principal do projeto encapsula a lógica para interagir com servos, motores e joysticks.
- Responsabilidades incluem:
  - Configurar os controladores de servo e motor.
  - Definir direção e velocidade.
  - Processar entradas do joystick em tempo real.
  - Gerenciar execução em múltiplas threads.

### 2. Interface de Hardware

**Comunicação I2C**

- O projeto utiliza o protocolo I2C para comunicação com os controladores de servo e motor.
- Funções principais:
  - Leitura e escrita de bytes em dispositivos I2C.
  - Configuração de valores PWM para controle preciso de motores e servos.

**Controladores de Servo e Motor**

- Servo: Controla o mecanismo de direção do carro.
- Motor: Movimenta as rodas do carro, definindo velocidade e direção.

### 3. Sistema de Entrada

**Integração com Joystick**

- A biblioteca SDL2 é utilizada para lidar com a entrada do joystick.
- Atualizações do joystick em tempo real são processadas em uma thread dedicada para ajustar velocidade e direção.

### 4. Gerenciamento de Concorrência

**Execução em Threads**

- A entrada do joystick é processada em uma thread separada para garantir que o loop principal da aplicação permaneça responsivo.
- A biblioteca `std::thread` é utilizada para gerenciamento de threads.

### 5. Tratamento de Sinais

**Encerramento Seguro**

- O tratamento de sinais é implementado para parar o carro com segurança quando o programa é interrompido (e.g., `Ctrl+C`).
- Garante que o hardware seja reiniciado para um estado seguro (e.g., motores parados, direção centralizada).

---

## Arquitetura em Camadas

O projeto segue uma arquitetura em camadas:

### Camada 1: Camada de Abstração de Hardware (HAL)

- **Objetivo:** Abstrair as interações de hardware de baixo nível, como comunicação I2C e controle PWM.
- **Componentes:**
  - `write_byte`
  - `read_byte`
  - `write_byte_data`
  - `read_byte_data`

### Camada 2: Lógica de Controle

- **Objetivo:** Implementar a funcionalidade de controle do carro, incluindo gerenciamento de velocidade e direção.
- **Componentes:**
  - `init_servo`
  - `init_motors`
  - `set_speed`
  - `set_steering`

### Camada 3: Manipulação de Entrada

- **Objetivo:** Processar a entrada do usuário a partir do joystick para controle em tempo real.
- **Componentes:**
  - `process_joystick`

### Camada 4: Camada de Aplicação

- **Objetivo:** Fornecer o ponto de entrada e gerenciar o ciclo de vida geral do sistema.
- **Componentes:**
  - `main.cpp`
  - Tratamento de sinais para encerramento seguro.

---

## Fluxo do Sistema

1. **Inicialização**

   - A comunicação I2C é configurada para os controladores de servo e motor.
   - Os controladores de servo e motor são configurados.

2. **Processamento da Entrada do Joystick**

   - A entrada do joystick é continuamente lida em uma thread dedicada.
   - Velocidade e direção são ajustadas com base na entrada do joystick.

3. **Execução**

   - O carro opera em um loop, respondendo à entrada do joystick e mantendo o estado.

4. **Encerramento**

   - Em caso de interrupção do sinal (e.g., `SIGINT`), o sistema para o carro com segurança, redefinindo velocidade e direção.

---

## Diagrama de Interação

```
+-------------------+         +----------------------+         +--------------------+
| Camada de Aplicação |         | Camada de Lógica de Controle |         | Interface de Hardware |
|   (main.cpp)      |         |       (Classe JetCar)       |         |       (I2C)        |
+-------------------+         +----------------------+         +--------------------+
        |                              |                              |
        |                              |                              |
        +------------------------------>----------------------------->
                             Fluxo do Sistema
```

---

## Melhorias Futuras

- Adicionar suporte para dispositivos de entrada adicionais (e.g., teclado, controles de jogos).
- Implementar registro de diagnósticos para depuração de problemas de hardware.
- Otimizar o uso de múltiplas threads para melhor desempenho.

---

## Conclusão

A arquitetura do projeto JetCar é modular, escalável e projetada para fornecer controle preciso sobre um carro robótico. Ao separar a interação com o hardware, lógica de controle e entrada do usuário, a arquitetura garante uma base de código limpa e de fácil manutenção.

