import torch
import torch.nn as nn
from model import LaneNet

# Define o dispositivo (nesse caso, CPU)
device = torch.device("cpu")

# Cria uma instância do modelo e o move para o dispositivo definido
model = LaneNet().to(device)

# Carrega o checkpoint e extrai apenas o state_dict do modelo
checkpoint = torch.load('retrain.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Coloca o modelo em modo de avaliação
model.eval()

# Cria uma entrada fictícia para definir as dimensões esperadas pelo modelo
dummy_input = torch.randn(1, 3, 512, 512, device=device)

# Exporta o modelo para o formato ONNX
torch.onnx.export(
    model,
    dummy_input,
    "lanenet_retrain.onnx",
    opset_version=11,           # Compatível com TensorRT 8.2.1
    input_names=["images"],     # Nome da entrada
    output_names=["output"],    # Nome da saída
    dynamic_axes=None           # Todas as dimensões são estáticas
)

