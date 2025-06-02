import torch
import torch.nn as nn
from model import LaneNet

device = torch.device("cuda")
model = LaneNet().to(device)

retrain_path = "../models/best_models/model_45.pth"
checkpoint = torch.load(retrain_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Criar entrada fictícia
dummy_input = torch.randn(1, 3, 256, 144, device=device)

torch.onnx.export(
   model,
   dummy_input,
   "lane.onnx",
   opset_version=11,  # Mantém compatibilidade com TensorRT 8.2.1
   input_names=["images"],
   output_names=["output"],
   dynamic_axes=None  # Remove dimensões dinâmicas

)