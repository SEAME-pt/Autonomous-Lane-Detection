from torch2trt import torch2trt
import torch

model = torch.jit.load("lanenet_model.pth").to("cuda")  # Load TorchScript model
model.eval()

input = torch.randn(1, 3, 590, 1640).cuda()  # Match your model's input size
model_trt = torch2trt(model, [input])  # Convert to TensorRT

torch.save(model_trt.state_dict(), "lanenet_model_trt.pth")  # Save optimized model