import carla
import time
import torch
import numpy as np
import cv2
import signal
import sys
sys.path.append('/home/seame/Autonomous-Lane-Detection/pytorch') 
from model import LaneNet
import threading
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from collections import deque

device = torch.device("cuda")
model = LaneNet().to(device)
model.load_state_dict(torch.load('/home/seame/Autonomous-Lane-Detection/pytorch/models/lanes_20.pth', map_location=device))
model.eval()

vehicle = None
camera = None
latest_image = None

def show_images():
    global latest_image
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection", 800, 600)
    while True:
        if latest_image is not None:
            cv2.imshow("Lane Detection", latest_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if camera is not None:
                    camera.stop()
                    camera.destroy()
                if vehicle is not None:
                    vehicle.destroy()
                time.sleep(1)  
                sys.exit(0)
            time.sleep(0.05)  # Add a small delay to prevent excessive CPU usage
        else:
            time.sleep(0.1)  # Wait if there's no image yet
    cv2.destroyAllWindows() 

def update_camera_position(world):
    spectator = world.get_spectator()  # spectator camera
    if vehicle is not None and vehicle.is_alive:
        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),  # Birdseye view
        carla.Rotation(pitch=-80)))

steering_history = deque(maxlen=10)
previous_steering = 0
integral_error = 0
dt = 0.1 

def calculate_steering(lane_mask, image_width):
    global previous_steering
    height, width = lane_mask.shape
    roi_height = height * 3 // 4 
    roi = lane_mask[-roi_height:, :]  

    y_coords, x_coords = np.where(roi > 0)
    if len(x_coords) == 0:
        print("No lanes detected, continue straight")
        return 0, None

    image_center_x = width // 2
    left_points = [(x, y) for x, y in zip(x_coords, y_coords) if x < image_center_x]
    right_points = [(x, y) for x, y in zip(x_coords, y_coords) if x >= image_center_x]
    
    left_lane = None
    right_lane = None
    
    # Find the closest
    if left_points:
        left_lane = max(left_points, key=lambda p: (p[1], -p[0]))
    
    if right_points:
        right_lane = max(right_points, key=lambda p: (p[1], p[0]))
    
    if left_lane is None and right_lane is None:
        print("No valid lanes detected, keep straight.")
        return 0, None

    if left_lane is not None and right_lane is not None:
        target_center = (left_lane[0] + right_lane[0]) / 2
    elif left_lane is not None:
        target_center = left_lane[0] + (width * 0.25)  # Assume road width
    elif right_lane is not None:
        target_center = right_lane[0] - (width * 0.25)

    car_center = width / 2
    offset = target_center - car_center
    max_offset = width / 2
    max_angle = 200 
    steering_angle = offset / max_offset
    # Normalize steering angle to [-1, 1] for CARLA control
    normalized_steering_angle = np.clip(steering_angle, -1, 1)
    # car_center = width / 2
    # error = (target_center - car_center) / (width / 2)  # Normalize error to [-1, 1]
    
    # Kp, Ki, Kd = 0.5, 0.1, 0.2  # PID coefficients (tune these)
    
    # global integral_error
    # integral_error += error * dt
    # integral_error = np.clip(integral_error, -1, 1)  # Anti-windup
    
    # derivative = (error - previous_steering) / dt
    
    # steering_angle = Kp * error + Ki * integral_error + Kd * derivative
    
    # # Smoothing
    # steering_history.append(steering_angle)
    # smoothed_steering = sum(steering_history) / len(steering_history)
    
    # # Adaptive response (less aggressive steering at high angles)
    # adaptive_steering = np.sign(smoothed_steering) * (abs(smoothed_steering) ** 0.7)
    
    # # Update previous steering for next iteration
    # previous_steering = error
    
    # # Normalize steering angle to [-1, 1] for CARLA control
    # normalized_steering_angle = np.clip(adaptive_steering, -1, 1)
    return normalized_steering_angle, target_center


def process_image(image):
    global latest_image
    frame = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]
    frame_resized = cv2.resize(frame, (512, 512))
    frame_tensor = transforms.ToTensor()(frame_resized).unsqueeze(0).to(device)
    frame_tensor = transforms.functional.normalize(frame_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    with torch.no_grad():
        raw_output = model(frame_tensor)
    # print(raw_output)

    lane_mask = torch.sigmoid(raw_output).squeeze().cpu() 
    lane_mask = (lane_mask > 0.5).numpy().astype(np.uint8)
    # lane_mask = cv2.GaussianBlur(lane_mask, (5, 5), 0)
    steering, target_center = calculate_steering(lane_mask, 512)

    overlay = frame_resized.copy()
    height, width = frame_resized.shape[:2]

    lane_mask_color = cv2.cvtColor(lane_mask * 255, cv2.COLOR_GRAY2BGR)
    lane_mask_color[..., 2] = lane_mask_color[..., 2] * 255 
    overlay = cv2.addWeighted(overlay, 0.6, lane_mask_color, 0.4, 0)
        
    if target_center is not None:
        cv2.line(
            overlay,
            (int(target_center), overlay.shape[0]),
            (int(target_center), overlay.shape[0] // 2),
            (255, 255, 0), 
            thickness=2,
        )

    latest_image = overlay
    if vehicle is not None and vehicle.is_alive:
        control = vehicle.get_control()
        control.throttle = max(0.8, 0.8 - abs(steering) * 0.8) 
        control.brake = 0.0
        control.steer = steering
        vehicle.apply_control(control)


def start_simulation():
    global vehicle
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    try:
        print(f"Connected to CARLA {client.get_server_version()}")
    except RuntimeError:
        print("Connection failed, start CARLA first")
        sys.exit(1)

    world = client.load_world("Town03")
    blueprint_library = world.get_blueprint_library()
    settings = world.get_settings()
    settings.synchronous_mode = True #ticks
    settings.fixed_delta_seconds = 0.05  
    settings.max_substep_delta_time = 0.01
    settings.substepping = True
    settings.max_substeps = 10
    world.apply_settings(settings)

    vehicle_bp = blueprint_library.find('vehicle.volkswagen.t2_2021') 
    spawn_points = world.get_map().get_spawn_points()

    vehicle = None
    for spawn_point in spawn_points:
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            break 
        except RuntimeError:
            continue
    vehicle.set_autopilot(False)
    vehicle.set_simulate_physics(True) 
    camera_transform = carla.Transform(carla.Location(x=4.0, y=0.0, z=1.4))  #vehicle-attached camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('fov', '120') 
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    camera.listen(lambda image: process_image(image))
    try:
        while True:
            world.tick()
            update_camera_position(world) 
            time.sleep(0.05)  
    except KeyboardInterrupt:
        print(f"Frame processing failed")

threading.Thread(target=show_images, daemon=True).start()

start_simulation()