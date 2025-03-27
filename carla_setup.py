import carla
import time
import torch
import numpy as np
import cv2
import signal
import sys
sys.path.append('/home/seame/Autonomous-Lane-Detection/pytorch/scripts') 
from model import LaneNet
import threading
import torch
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN

device = torch.device("cuda")
model = LaneNet().to(device)
checkpoint = torch.load('/home/seame/Autonomous-Lane-Detection/pytorch/models/retrain.pth', map_location=device) 
model.load_state_dict(checkpoint['model_state_dict']) 
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

def calculate_steering(lane_mask, image_width):
    height, width = lane_mask.shape
    roi_height = height * 3 // 4
    roi = lane_mask[-roi_height:, :]
    y_coords, x_coords = np.where(roi > 0)
    if len(x_coords) < 15:  #minimum point threshold
        return -10, None
    left_x = x_coords[x_coords < width // 2]
    right_x = x_coords[x_coords >= width // 2]
    
    if len(left_x) < 15 or len(right_x) < 15: #single-lane behavior
        if len(x_coords) > 15:
            print("One lane")
            weighted_center = np.average(x_coords, weights=(y_coords - y_coords.min() + 1)**2)
            offset = weighted_center - (width / 2)
            steering = np.clip(offset / (width * 0.4), -1, 1) #Less agressive steering
            return steering * 0.6, None #even less steering gain single lane
        return -10, None  
    # Weighted average for lane centers
    left_center = np.average(left_x, weights=(y_coords[x_coords < width//2] - y_coords.min() + 1))
    right_center = np.average(right_x, weights=(y_coords[x_coords >= width//2] - y_coords.min() + 1))
    lane_width = abs(right_center - left_center)
    
    target_center = (left_center + right_center) / 2
    offset = target_center - (width / 2)
    if abs(offset) < width * 0.03:
        return 0.0, target_center #no steering for small offsets
    # Progressive steering response - less aggressive
    steering = np.arctan(offset / (lane_width * 0.6)) * (2/np.pi) #less aggressive steering
    # Smoothing with memory 
    if not hasattr(calculate_steering, 'history'):
        calculate_steering.history = [0, 0, 0]
    calculate_steering.history = calculate_steering.history[1:] + [steering]
    smoothed = sum([0.2, 0.3, 0.5] * np.array(calculate_steering.history))
    return np.clip(smoothed, -1, 1), target_center

def process_image(image):
    global latest_image
    frame = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]
    frame_resized = cv2.resize(frame, (512, 512))
    frame_tensor = transforms.ToTensor()(frame_resized).unsqueeze(0).to(device)
    frame_tensor = transforms.functional.normalize(frame_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    with torch.no_grad():
        raw_output = model(frame_tensor)
    lane_mask = torch.sigmoid(raw_output).squeeze().cpu() 
    lane_mask = (lane_mask > 0.5).numpy().astype(np.uint8)
    lane_mask = cv2.GaussianBlur(lane_mask, (5, 5), 0)
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
    if steering == -10:
        control = vehicle.get_control()
        control.throttle = 0.0
        print("Braking")
        control.brake = 0.7
        vehicle.apply_control(control)
    elif vehicle is not None and vehicle.is_alive:
        control = vehicle.get_control()
        control.throttle = max(0.7, 0.75 - abs(steering) * 0.9) 
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

    world = client.load_world("Town05")
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