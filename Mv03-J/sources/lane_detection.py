import cv2
import numpy as np
import socket

# ️ Configuração do Servidor C++ (JetRacer)
HOST = "127.0.0.1"  # Mude para o IP do JetRacer se necessário
PORT = 65432

# Definir pipeline GStreamer para captura da câmera do JetRacer
CAMERA_PIPELINE = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, "
    "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, "
    "format=(string)BGRx ! videoconvert ! video/x-raw, "
    "format=(string)BGR ! appsink drop=true max-buffers=1"
)

# Configuração para salvar vídeos
video_fps = 30.0
frame_size = (1280, 720)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Formato AVI

out_original = cv2.VideoWriter("captura_original.mp4", fourcc, video_fps, frame_size)
out_processed = cv2.VideoWriter("deteccao_faixa.mp4", fourcc, video_fps, frame_size)

def enviar_comando(angle, speed):
    """Envia o ângulo de direção e a velocidade para o servidor C++ via socket."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            comando = f"{angle} {speed}"
            s.sendall(comando.encode("utf-8"))
            print(f"[INFO] Comando enviado: {comando}")
    except Exception as e:
        print(f"[ERRO] Falha ao enviar comando: {e}")

def melhorar_contraste(image):
    """Melhora o contraste da imagem usando equalização de histograma."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def segmentar_faixas(image):
    """Segmenta apenas faixas brancas na imagem."""
    image = melhorar_contraste(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Ajuste para detectar somente faixas brancas
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    return cv2.bitwise_and(image, image, mask=mask_white)

def calcular_angulo(image):
    """Processa a imagem para detectar as faixas e calcular o ângulo de direção."""
    image_segmentada = segmentar_faixas(image)
    gray = cv2.cvtColor(image_segmentada, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=30)
    if lines is None:
        print("[INFO] Nenhuma linha detectada, mantendo direção reta.")
        return 0, "Centro", image

    image_width = image.shape[1]
    left_x = []
    right_x = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if slope < -0.3:
            left_x.append(x1)
        elif slope > 0.3:
            right_x.append(x2)

    if not left_x or not right_x:
        return 0, "Centro", image

    lane_center = (np.mean(left_x) + np.mean(right_x)) / 2
    car_center = image_width / 2

    offset = lane_center - car_center
#    offset = car_center - lane_center
    max_offset = image_width / 2
    max_angle = 30

    steering_angle = (offset / max_offset) * max_angle * 10
    steering_angle = np.clip(steering_angle, -30, 30)

    direction = "Centro"
    if steering_angle < -5:
        direction = "Esquerda"
    elif steering_angle > 5:
        direction = "Direita"

    return steering_angle, direction, image_segmentada

def processar_camera():
    """Captura da câmera do JetRacer, processa os frames e envia comandos"""
    CAMERA_PIPELINE = (
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, "
        "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, "
        "format=(string)BGRx ! videoconvert ! video/x-raw, "
        "format=(string)BGR ! appsink drop=true max-buffers=1"
    )

    cap = cv2.VideoCapture(CAMERA_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("[ERRO] Não foi possível acessar a câmera!")
        return

    cv2.namedWindow("JetRacer Camera", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Falha ao capturar frame!")
            break

        angulo, direcao, processed_image = calcular_angulo(frame)
        enviar_comando(angulo, 25.0)

        cv2.imshow("JetRacer Camera", processed_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[SUCESSO] Processamento finalizado.")


if __name__ == "__main__":
    processar_camera()

