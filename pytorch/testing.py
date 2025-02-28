import torch
import torch.nn as nn
import torch.optim as optim
from model import LaneNet
from torch.utils.data import DataLoader
from dataset import LaneDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex

device = torch.device("cuda")
model = LaneNet().to(device)
model.load_state_dict(torch.load('lanenet_model5.pth', map_location=device))
model.eval()

mask_dir = os.path.join('.', 'TUSimple', 'train_set', 'seg_label')
image_dir = os.path.join('.', 'TUSimple', 'test_set', 'clips') 
image_paths = []
mask_paths = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file == '20.jpg':
            image_path = os.path.join(root, file)
            mask_path = os.path.join(mask_dir, os.path.relpath(image_path, image_dir)).replace('.jpg', '.png')
            mask_paths.append(mask_path)
            image_paths.append(image_path)

mask_dir = os.path.join('.', 'TUSimple', 'train_set', 'seg_label', '06040308_1062.MP4')
image_dir = os.path.join('.', 'TUSimple', 'test_set', 'clips', '06040308_1062.MP4') 
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file == '.jpg':
            image_path = os.path.join(root, file)
            mask_path = os.path.join(mask_dir, os.path.relpath(image_path, image_dir)).replace('.jpg', '.png')
            mask_paths.append(mask_path)
            image_paths.append(image_path)

test_dataset = LaneDataset(image_paths, mask_paths)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def matplot_masks(images, predicted_mask, iter):
    img = images.squeeze().cpu().numpy()
    img = np.transpose(img, (1, 2, 0)) 
    pred_mask = predicted_mask.squeeze().cpu().numpy()
    plt.style.use('dark_background')
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.imshow(img)
    ax1.set_title('Image')
    ax1.axis('off')
    ax3.imshow(pred_mask, cmap='gray')
    ax3.set_title('Predicted Mask')
    ax3.axis('off')
    plt.tight_layout()
    plt.savefig(f'./test_img/{iter}.png')
    plt.close()

iter = 0
total_iou = 0
num_images = 0
with torch.no_grad():  # No gradients are calculated during testing
    for images, masks, paths, path in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images) # Forward pass
        predictions = torch.sigmoid(outputs)
        predicted_mask = (predictions > 0.5).float()  # Convert probabilities to binary predictions
        num_images += 1
        iter += 1
        if iter % 100:
            matplot_masks(images, predicted_mask, iter)


