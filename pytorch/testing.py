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

device = torch.device("cuda")
model = LaneNet().to(device)
model.load_state_dict(torch.load('./models/lanenet_model6.pth', map_location=device))
model.eval()

image_paths = []
mask_paths = []

image_dir = os.path.join('.', 'new_data', 'val') 
for root, dirs, files in os.walk(image_dir):
    for file in files:
        image_path = os.path.join(root, file)
        file_name, file_ext = os.path.splitext(file)
        image_paths.append(image_path)

image_dir = os.path.join('.','images') 
for root, dirs, files in os.walk(image_dir):
    for file in files:
        image_path = os.path.join(root, file)
        image_paths.append(image_path)

test_dataset = LaneDataset(image_paths, transforms=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

import torch
import numpy as np

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if torch.is_tensor(image):
        denorm_image = image.clone().cpu().numpy()
    else:
        denorm_image = image.copy()

    # Ensure the image is in (C, H, W) format
    if denorm_image.shape[0] == 3:
        denorm_image = denorm_image.transpose(1, 2, 0)  # Change to (H, W, C)

    # Denormalize
    denorm_image = denorm_image * np.array(std) + np.array(mean)

    # Clip values to [0, 1] range
    return np.clip(denorm_image, 0, 1)


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
    plt.savefig(f'./test_img/{iter}.png')
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


