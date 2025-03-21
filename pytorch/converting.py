import torch
import torch.nn as nn
from model import LaneNet

device = torch.device("cuda")
model = LaneNet().to(device)
model.load_state_dict(torch.load('lanenet_model6.pth', map_location=device))
model.eval()


# Criar entrada fictícia
dummy_input = torch.randn(1, 3, 640, 640, device=device)

# Exportar para ONNX compatível com TensorRT
torch.onnx.export(
    model,
    dummy_input,
    "lanenet.onnx",
    opset_version=11,  # Mantém compatibilidade com TensorRT 8.2.1
    input_names=["images"],
    output_names=["output"],
    dynamic_axes=None  # Remove dimensões dinâmicas
)

