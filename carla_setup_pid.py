# File: carla_setup_pid.py
import carla
import time
import torch
import numpy as np
import cv2
import signal
import sys
import threading
import torchvision.transforms as transforms
from model import LaneNet

# =============== CONFIG ===================
DEVICE = torch.device("cuda")
MODEL_PATH = "/home/seame/Autonomous-Lane-Detection/pytorch/models/retrain.pth"
IMG_SIZE = 512
OFFSET_CM = 12.0

# =============== PID Controller =============
class PIDController:
    def __init__(self, kp=1.5, ki=0.1, kd=0.2, max_angle=40.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_angle = max_angle
        self.erro_integral = 0.0
        self.erro_anterior = 0.0

    def calcular(self, erro, dt):
        self.erro_integral += erro * dt
        derivada = (erro - self.erro_anterior) / dt
        self.erro_anterior = erro
        saida = self.kp * erro + self.ki * self.erro_integral + self.kd * derivada
        return np.clip(saida, -self.max_angle, self.max_angle)

pid = PIDController()

# ============== Lane Detection =============
def calcular_erro_lateral(mask_binaria, offset_cm=12.0):
    altura, largura = mask_binaria.shape
    linha_y = int(altura * 4 / 5)
    colunas_ativas = np.where(mask_binaria[linha_y] > 0)[0]

    if len(colunas_ativas) == 0:
        return 0.0, None

    centro_imagem = largura // 2
    escala = 40.0 / (largura / 2)
    erro_cm = 0.0

    if len(colunas_ativas) >= 20:
        esquerda = colunas_ativas[0]
        direita = colunas_ativas[-1]
        centro_faixa = (esquerda + direita) // 2
        erro_cm = (centro_faixa - centro_imagem) * escala
        centro_pista = centro_faixa
    else:
        faixa_a_direita = colunas_ativas[-1] > centro_imagem
        borda = colunas_ativas[-1] if faixa_a_direita else colunas_ativas[0]
        erro_px = borda - centro_imagem
        erro_lateral = erro_px * escala
        erro_cm = erro_lateral - offset_cm if faixa_a_direita else erro_lateral + offset_cm
        centro_pista = borda

    return erro_cm, centro_pista

# ============== Modelo =============
model = LaneNet().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
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
            time.sleep(0.05)
        else:
            time.sleep(0.1)


def process_image(image):
    global latest_image
    frame = np.array(image.raw_data).reshape((image.height, image.width, 4))[:, :, :3]
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    frame_tensor = transforms.ToTensor()(frame_resized).unsqueeze(0).to(DEVICE)
    frame_tensor = transforms.functional.normalize(frame_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        raw_output = model(frame_tensor)
    lane_mask = torch.sigmoid(raw_output).squeeze().cpu()
    lane_mask = (lane_mask > 0.9).numpy().astype(np.uint8)

    erro_cm, centro_pista = calcular_erro_lateral(lane_mask, OFFSET_CM)
    angulo_pid = -pid.calcular(erro_cm, dt=0.1)

    overlay = frame_resized.copy()
    lane_mask_color = cv2.cvtColor(lane_mask * 255, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(overlay, 0.6, lane_mask_color, 0.4, 0)

    if centro_pista is not None:
        h = overlay.shape[0]
        cv2.line(overlay, (int(centro_pista), h), (int(centro_pista), h // 2), (255, 255, 0), 2)

    if vehicle is not None and vehicle.is_alive:
        control = vehicle.get_control()
        control.throttle = max(0.6, 0.8 - abs(angulo_pid) * 0.02)
        control.steer = np.clip(angulo_pid / 40.0, -1.0, 1.0)
        control.brake = 0.0
        vehicle.apply_control(control)

    latest_image = overlay


def update_camera_position(world):
    spectator = world.get_spectator()
    if vehicle is not None and vehicle.is_alive:
        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40), carla.Rotation(pitch=-80)))


def start_simulation():
    global vehicle, camera
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
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.max_substep_delta_time = 0.01
    settings.substepping = True
    settings.max_substeps = 10
    world.apply_settings(settings)

    vehicle_bp = blueprint_library.find('vehicle.volkswagen.t2_2021')
    spawn_points = world.get_map().get_spawn_points()

    for spawn_point in spawn_points:
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            break
        except RuntimeError:
            continue

    vehicle.set_autopilot(False)
    vehicle.set_simulate_physics(True)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('fov', '120')
    camera_transform = carla.Transform(carla.Location(x=4.0, y=0.0, z=1.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera.listen(lambda image: process_image(image))

    try:
        while True:
            world.tick()
            update_camera_position(world)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Encerrando...")

# === Main ===
threading.Thread(target=show_images, daemon=True).start()
start_simulation()
