import torch
import torch.nn as nn

# Carregar YOLOv5 treinado
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Remover AutoShape para evitar problemas na conversão
model.model[-1].export = False  

# Criar entrada fictícia
dummy_input = torch.randn(1, 3, 640, 640)

# Garantir que camadas problemáticas sejam fixas
for m in model.modules():
    if isinstance(m, nn.Upsample):
        m.recompute_scale_factor = None

# Exportar para ONNX compatível com TensorRT
torch.onnx.export(
    model,
    dummy_input,
    "yolov5s_fixed.onnx",
    opset_version=11,  # Mantém compatibilidade com TensorRT 8.2.1
    input_names=["images"],
    output_names=["output"],
    dynamic_axes=None  # Remove dimensões dinâmicas
)

print("Modelo YOLOv5 exportado com sucesso para ONNX!")
