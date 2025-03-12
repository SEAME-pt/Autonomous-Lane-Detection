import carla
import time
import torch
import numpy as np
import cv2
import signal
import sys
sys.path.append('/home/seame/Autonomous-Lane-Detection/pytorch') 
from model import LaneNet

device = torch.device("cuda")
model = LaneNet().to(device)
model.load_state_dict(torch.load('/home/seame/Autonomous-Lane-Detection/pytorch/lanenet_model5.pth', map_location=device))
model.eval()

def update_camera_position(world):
    vehicle_location = vehicle.get_location()
    # player_camera = world.get_actors().filter('sensor.camera.rgb')[0]
    print(f"Vehicle Location: {vehicle_location}")
    spectator = world.get_spectator()  # Track camera separately
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40),  # Birdseye view
        carla.Rotation(pitch=-80)))

def process_image(image):
    frame = np.array(image.raw_data)
    frame = frame.reshape((image.height, image.width, 4))  # RGBA format
    frame = frame[:, :, :3]  # Convert to RGB (ignore alpha channel)

    # Convert frame to tensor for model input
    frame_tensor = torch.tensor(frame).float().to(device)
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize and add batch dimension

    # Run the frame through the model to get the lane mask
    with torch.no_grad():
        lane_mask = model(frame_tensor).squeeze().cpu().numpy()  # Get the predicted lane mask
    lane_mask = (lane_mask > 0.5).astype(np.uint8)  

    # Resize the lane_mask to match the frame size
    lane_mask_resized = cv2.resize(lane_mask, (image.width, image.height))

    # Convert grayscale lane_mask to 3 channels (RGB)
    if len(lane_mask_resized.shape) == 2:  # If the mask is single channel
        lane_mask_resized = cv2.cvtColor(lane_mask_resized, cv2.COLOR_GRAY2BGR)

    lane_mask_resized[..., 0] = lane_mask_resized[..., 0] * 255  # Use Blue color for lane
    lane_mask_resized[..., 1] = 0  # Green channel is 0
    lane_mask_resized[..., 2] = lane_mask_resized[..., 2] * 255  # Use Red color for lane

    # Overlay the lane mask on the frame (adjust the weights if necessary)
    overlay = np.uint8(frame * 0.6 + lane_mask_resized * 0.4)
    cv2.imshow("Lane Detection Overlay", overlay)
    cv2.waitKey(1)

    lane_center = np.argmax(np.sum(lane_mask_resized, axis=0))  # Find the lane center (max sum column)
    image_center = lane_mask_resized.shape[1] // 2  # Center of the image (width)

    # Calculate the error in the lane center from the image center
    steering_error = (lane_center - image_center) / float(image_center)  # Normalize error
    control = vehicle.get_control() 
    control.steer = np.clip(steering_error, 0.0, 0.0)   # Adjust the steering angle (you can add scaling to make it smoother)
    control.throttle = 0.8 
    control.brake = 0.0 
    vehicle.apply_control(control) 


def start_simulation():
    global vehicle, camera
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    try:
        print(f"Connected to CARLA {client.get_server_version()}")
    except RuntimeError:
        print("Connection failed! Start CARLA first with:")
        sys.exit(1)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

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
    
    camera_transform = carla.Transform(carla.Location(x=4.0, y=0.0, z=2.4)) 
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    camera.listen(lambda image: process_image(image))

    def clean_up(signal, frame):
            camera.stop()
            vehicle.destroy()
            print("Cleaning up...")
            cv2.destroyAllWindows()
            sys.exit(0)

    signal.signal(signal.SIGINT, clean_up)

    try:
        while True:
            world.tick()
            update_camera_position(world) 
            time.sleep(0.05)  # Simulate some time to keep the world running
    except KeyboardInterrupt:
        print(f"Frame processing failed: {str(e)}")
        clean_up(None, None)

start_simulation()