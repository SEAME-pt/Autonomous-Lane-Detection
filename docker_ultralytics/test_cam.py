import cv2

pipeline = (
    "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, "
    "format=(string)NV12, framerate=30/1 ! nvvidconv ! video/x-raw, "
    "format=(string)BGRx ! videoconvert ! video/x-raw, "
    "format=(string)BGR ! appsink drop=true max-buffers=1"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if cap.isOpened():
    print("Câmera aberta com sucesso!")
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("frame_teste.jpg", frame)
        print("Frame capturado e salvo!")
    else:
        print("Não foi possível capturar um frame.")
    cap.release()
else:
    print("Falha ao abrir a câmera!")

