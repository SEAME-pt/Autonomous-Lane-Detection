// jetracer.cpp
#include "jetracer.hpp"
#include <stdexcept>
#include <thread>
#include <algorithm>
#include <cstring>

// I2CDevice Implementation
I2CDevice::I2CDevice(const std::string& device, int address) {
    fd_ = open(device.c_str(), O_RDWR);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open I2C device: " + device);
    }
    if (ioctl(fd_, I2C_SLAVE, address) < 0) {
        close(fd_);
        throw std::runtime_error("Failed to set I2C address");
    }
}

I2CDevice::~I2CDevice() {
    if (fd_ >= 0) {
        close(fd_);
    }
}

void I2CDevice::write_byte(uint8_t reg, uint8_t value) {
    uint8_t buffer[2] = {reg, value};
    if (write(fd_, buffer, 2) != 2) {
        close(fd_);
        throw std::runtime_error("Failed to write to I2C device");
    }
}

uint8_t I2CDevice::read_byte(uint8_t reg) {
    if (write(fd_, &reg, 1) != 1) {
        close(fd_);
        throw std::runtime_error("Failed to write register to I2C device");
    }
    uint8_t value;
    if (read(fd_, &value, 1) != 1) {
        close(fd_);
        throw std::runtime_error("Failed to read from I2C device");
    }
    return value;
}

// JetCar Implementation
JetCar::JetCar(int servo_addr, int motor_addr)
    : servo_addr_(servo_addr),
      motor_addr_(motor_addr),
      running_(false),
      servo_device_("/dev/i2c-1", servo_addr),
      motor_device_("/dev/i2c-1", motor_addr) {
    init_servo();
    init_motors();
}

JetCar::~JetCar() {
    stop();
}

void JetCar::init_servo() {
    try {
        servo_device_.write_byte(0x00, 0x06);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        servo_device_.write_byte(0x00, 0x10);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        servo_device_.write_byte(0xFE, 0x79);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        servo_device_.write_byte(0x01, 0x04);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        servo_device_.write_byte(0x00, 0x20);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } catch (const std::exception& e) {
        std::cerr << "Servo initialization failed: " << e.what() << std::endl;
        stop();
    }
}

void JetCar::init_motors() {
    try {
        motor_device_.write_byte(0x00, 0x20);

        int prescale = static_cast<int>(std::floor(25000000.0 / 4096.0 / 100 - 1));
        int oldmode = motor_device_.read_byte(0x00);
        int newmode = (oldmode & 0x7F) | 0x10;

        motor_device_.write_byte(0x00, newmode);
        motor_device_.write_byte(0xFE, prescale);
        motor_device_.write_byte(0x00, oldmode);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        motor_device_.write_byte(0x00, oldmode | 0xA1);
    } catch (const std::exception& e) {
        std::cerr << "Motor initialization failed: " << e.what() << std::endl;
        stop();
    }
}

void JetCar::set_steering(int angle) {
    angle = std::clamp(angle, -MAX_ANGLE_, MAX_ANGLE_);

	std::cout << "[DEBUG] set_steering ANGLE: " << angle << std::endl;

    int pwm = 0;
    if (angle < 0) {
        pwm = SERVO_CENTER_PWM_ + (angle / static_cast<float>(MAX_ANGLE_)) * (SERVO_CENTER_PWM_ - SERVO_LEFT_PWM_);
	std::cout << "[DEBUG] set_steering ANGLE LEFT | PWM: " << angle << "|" << pwm << std::endl;
    } else if (angle > 0) {
        pwm = SERVO_CENTER_PWM_ + (angle / static_cast<float>(MAX_ANGLE_)) * (SERVO_RIGHT_PWM_ - SERVO_CENTER_PWM_);
	std::cout << "[DEBUG] set_steering ANGLE RIGHT | PWM: " << angle << "|" << pwm << std::endl;
    } else {
        pwm = SERVO_CENTER_PWM_;
	std::cout << "[DEBUG] set_steering ANGLE CENTER | PWM: " << angle << "|" << pwm << std::endl;
    }

    set_servo_pwm(0, 0, pwm);
    current_angle_ = angle;
}

void JetCar::set_servo_pwm(int channel, int on_value, int off_value) {
    int base_reg = 0x06 + (channel * 4);
    servo_device_.write_byte(base_reg, on_value & 0xFF);
    servo_device_.write_byte(base_reg + 1, on_value >> 8);
    servo_device_.write_byte(base_reg + 2, off_value & 0xFF);
    servo_device_.write_byte(base_reg + 3, off_value >> 8);
}

void JetCar::set_motor_pwm(int channel, int value) {
    value = std::clamp(value, 0, 4095);
    int base_reg = 0x06 + (channel * 4);
    motor_device_.write_byte(base_reg, 0);
    motor_device_.write_byte(base_reg + 1, 0);
    motor_device_.write_byte(base_reg + 2, value & 0xFF);
    motor_device_.write_byte(base_reg + 3, value >> 8);
}

void JetCar::set_speed(float speed) {

    if(speed > 35)
        speed = 35;

    speed = std::clamp(speed, -100.0f, 100.0f);
    int pwm_value = static_cast<int>(std::abs(speed) / 100.0f * 4095);

    

    if (speed > 0) {
        set_motor_pwm(0, pwm_value);
        set_motor_pwm(1, 0);
        set_motor_pwm(2, pwm_value);
        set_motor_pwm(5, pwm_value);
        set_motor_pwm(6, 0);
        set_motor_pwm(7, pwm_value);
    } else if (speed < 0) {
        set_motor_pwm(0, pwm_value);
        set_motor_pwm(1, pwm_value);
        set_motor_pwm(2, 0);
        set_motor_pwm(6, pwm_value);
        set_motor_pwm(7, pwm_value);
        set_motor_pwm(8, 0);
    } else {
        for (int channel = 0; channel < 9; ++channel) {
            set_motor_pwm(channel, 0);
        }
    }

    current_speed_ = speed;
}

void JetCar::smooth_steering(int target_angle, int increment) {
    target_angle = std::clamp(target_angle, -MAX_ANGLE_, MAX_ANGLE_);
    int step = (target_angle > current_angle_) ? increment : -increment;

    while ((step > 0 && current_angle_ < target_angle) || (step < 0 && current_angle_ > target_angle)) {
        current_angle_ += step;
        if ((step > 0 && current_angle_ > target_angle) || (step < 0 && current_angle_ < target_angle)) {
            current_angle_ = target_angle;
        }
        set_steering(current_angle_);
        std::this_thread::sleep_for(std::chrono::milliseconds(0));
    }
}

void JetCar::process_joystick() {
    if (SDL_Init(SDL_INIT_JOYSTICK) < 0) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return;
    }

    SDL_Joystick* joystick = SDL_JoystickOpen(0);
    if (!joystick) {
        std::cerr << "Failed to open joystick: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return;
    }

    /*for (int i = 0; i < SDL_JoystickNumButtons(joystick); ++i) {
        if (SDL_JoystickGetButton(joystick, i)) {
            std::cout << "Button: " << i << "pressed!" << std::endl;
        }
    }*/

    while (running_) {
        SDL_JoystickUpdate();

        int left_joystick_y = SDL_JoystickGetAxis(joystick, 1);
        int right_joystick_x = SDL_JoystickGetAxis(joystick, 2);
        set_speed(-left_joystick_y / 32767.0f * 100);
        smooth_steering(right_joystick_x / 32767.0f * MAX_ANGLE_, 10);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    SDL_JoystickClose(joystick);
    SDL_Quit();
}

void JetCar::start() {
    running_ = true;
    std::thread joystick_thread(&JetCar::process_joystick, this);
    joystick_thread.detach();
}

void JetCar::stop() {
    running_ = false;
    set_speed(0);
    set_steering(0);
}

bool JetCar::is_running() const {
    return running_.load();
}


