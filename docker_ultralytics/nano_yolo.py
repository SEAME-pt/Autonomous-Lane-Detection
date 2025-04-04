# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO

# Carrega o modelo (detecção, no caso)
model = YOLO("yolo11n.pt")

# Define o pipeline GStreamer para a câmera Jetson Nano
pipeline = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, "
    "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, "
    "format=(string)BGRx ! videoconvert ! video/x-raw, "
    "format=(string)BGR ! appsink drop=true max-buffers=1"
)

# Abre a câmera com o pipeline
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Erro ao abrir a câmera com o pipeline GStreamer")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Executa a inferência no frame
    results = model(frame)
    # Plota os resultados no frame
    annotated_frame = results[0].plot()

    # Mostra o frame com as detecções
    cv2.imshow("YOLO Inference", annotated_frame)
    # Encerra se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

