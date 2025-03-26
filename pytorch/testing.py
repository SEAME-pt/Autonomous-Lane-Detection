import torch
import torch.nn as nn
import torch.optim as optim
from model import LaneNet
from torch.utils.data import DataLoader
from dataset import LaneDataset, test_transforms
import os
from train import denormalize
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex

device = torch.device("cuda")
model = LaneNet().to(device)
model.load_state_dict(torch.load('./models/model_4.pth', map_location=device))
model.eval()

image_paths = []
mask_paths = []

image_dir = os.path.join('.', 'testing' ,'town4', 'val') 
for root, dirs, files in os.walk(image_dir):
    for file in files:
        image_path = os.path.join(root, file)
        file_name, file_ext = os.path.splitext(file)
        image_paths.append(image_path)

image_dir = os.path.join('.', 'testing', 'german_carla') 
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            image_paths.append(image_path)

test_dataset = LaneDataset(image_paths, transforms=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

def matplot_masks(images, predicted_mask, iter, path):
    img = images.squeeze().cpu().numpy()
    img = np.transpose(img, (1, 2, 0)) 
    denorm_image = denormalize(img)  
    pred_mask = predicted_mask.squeeze().cpu().numpy()
    plt.style.use('grayscale')
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.imshow(denorm_image)
    ax1.set_title('Image')
    ax1.axis('off')
    ax3.imshow(pred_mask, cmap='gray')
    ax3.set_title('Predicted Mask')
    ax3.axis('off')
    plt.tight_layout()
    print(f'image path: {path}, iter: {iter}')
    plt.savefig(f'./debug/test_img/{iter}.png')
    plt.close()

iter = 0
with torch.no_grad():  # No gradients are calculated during testing
    for images, paths in test_loader:
        images = images.to(device)
        outputs = model(images) # Forward pass
        predictions = torch.sigmoid(outputs)
        predicted_mask = (predictions > 0.5).float()  # Convert probabilities to binary predictions
        iter += 1
        if iter % 10 == 0:
            matplot_masks(images, predicted_mask, iter, paths)
