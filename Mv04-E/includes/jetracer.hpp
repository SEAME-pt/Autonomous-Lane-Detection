#ifndef JETRACER_HPP
#define JETRACER_HPP

#include <cstdint>
#include <string>
#include <atomic>
#include <iostream>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <SDL2/SDL.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"


class I2CDevice {
public:
    I2CDevice(const std::string& device, int address);
    ~I2CDevice();

    void write_byte(uint8_t reg, uint8_t value);
    uint8_t read_byte(uint8_t reg);

private:
    int fd_;
};

class JetCar {
public:
    JetCar(int servo_addr, int motor_addr);
    ~JetCar();

    void start();
    void stop();
    bool is_running() const;

    void set_speed(float speed);
    void set_steering(int angle);
    void smooth_steering(int target_angle, int increment);

    bool circle_mode_ = false;

private:
    void init_servo();
    void init_motors();
    void set_servo_pwm(int channel, int on_value, int off_value);
    void set_motor_pwm(int channel, int value);
    void process_joystick();

    static constexpr int MAX_ANGLE_ = 180;
    static constexpr int SERVO_LEFT_PWM_ = 130;
    static constexpr int SERVO_CENTER_PWM_ = 310;
    static constexpr int SERVO_RIGHT_PWM_ =490;

    int servo_addr_;
    int motor_addr_;
    std::atomic<bool> running_;
    I2CDevice servo_device_;
    I2CDevice motor_device_;
    int current_angle_ = 0;
    float current_speed_ = 0.0f;
};

#endif // JETRACER_HPP
