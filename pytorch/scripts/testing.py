import torch
import torch.nn as nn
import torch.optim as optim
from model import LaneNet
from torch.utils.data import DataLoader
from dataset import LaneDataset, test_transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex
import cv2
from pathlib import Path

device = torch.device("cuda")
model = LaneNet().to(device)
checkpoint = torch.load('../models/retrained/model_10.pth')
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

image_paths = []
mask_paths = []
i = 0

image_dir = os.path.join('..', 'testing', 'images') 
for root, dirs, files in os.walk(image_dir):
    for file in files:
        # if (file.endswith(".jpg")):
            image_path = os.path.join(root, file)
            file_name, file_ext = os.path.splitext(file)
            image_paths.append(image_path)

# image_dir = os.path.join('..', 'testing' ,'town4', 'val') 
# for root, dirs, files in os.walk(image_dir):
#     if i > 50:
#         break;
#     for file in files:
#         i += 1
#         image_path = os.path.join(root, file)
#         file_name, file_ext = os.path.splitext(file)
#         image_paths.append(image_path)
#         if i > 50:
#             break;

# image_dir = os.path.join('..', 'testing', 'german_carla') 
# i = 0
# for root, dirs, files in os.walk(image_dir):
#     if i > 50:
#         break;
#     for file in files:
#         if file.endswith('.jpg'):
#             i += 1;
#             image_path = os.path.join(root, file)
#             image_paths.append(image_path)
#             if i > 50:
#                 break ;

test_dataset = LaneDataset(image_paths, transforms=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = image.clone().cpu()  # Ensure tensor on CPU
    mean = torch.tensor(mean).reshape(3, 1, 1)
    std = torch.tensor(std).reshape(3, 1, 1)
    image = image * std + mean  # Denormalize
    image = image.clamp(0, 1)  # Clip to [0, 1]
    return image.permute(1, 2, 0).numpy()  # Convert to H, W, C

# Plotting function (fixed for correct visualization)
def matplot_masks(images, predicted_mask, path):
    # Extract image and mask
    img = images.squeeze(0)  # Remove batch dimension
    denorm_image = denormalize(img)  # Denormalize and convert to H, W, C
    pred_mask = predicted_mask.squeeze().cpu().numpy()  # Remove batch/channel dimensions

    # Plot
    plt.style.use('default')  # Remove grayscale style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.imshow(denorm_image)
    ax1.set_title('Image')
    ax1.axis('off')
    ax2.imshow(pred_mask, cmap='gray')
    ax2.set_title('Predicted Mask')
    ax2.axis('off')
    plt.tight_layout()
    plt.savefig(f'../debug/test_img/{Path(path[0]).stem}.png')
    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
    cv2.imwrite(f"../seame/{Path(path[0]).stem}.png", pred_mask_uint8)
    plt.close()

iter = 0
with torch.no_grad():  # No gradients are calculated during testing
    for images, paths in test_loader:
        images = images.to(device)
        outputs = model(images) # Forward pass
        predictions = torch.sigmoid(outputs)
        predicted_mask = (predictions > 0.7).float()  # Convert probabilities to binary predictions
        iter += 1
        matplot_masks(images, predicted_mask, paths)
