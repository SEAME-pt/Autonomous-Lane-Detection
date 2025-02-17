import torch
import torch.nn as nn
import torch.optim as optim
from model import LaneNet
from torch.utils.data import DataLoader
from dataset import LaneDataset
import os

device = torch.device("cuda")
model = LaneNet().to(device)
model.load_state_dict(torch.load('lanenet_model.pth', map_location=device), strict=False)
model.eval()

mask_dir = os.path.join('.', 'laneseg_label_w16', 'driver_182_30frame')
image_dir = os.path.join('.', 'driver_182_30frame') 
image_paths = []
mask_paths = []

for root, _, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            mask_path = os.path.join(mask_dir, os.path.relpath(image_path, image_dir)).replace('.jpg', '.png')
            if not os.path.exists(mask_path):  
                print(f"WARNING: Mask not found for {image_path}")
                continue  # Skip unmatched pairs
            mask_paths.append(mask_path)
            image_paths.append(image_path)


test_dataset = LaneDataset(image_paths, mask_paths)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# overlapping area between the predicted lane and the actual lane is crucial because it measures how well the model is identifying the correct regions
def compute_iou(pred_mask, true_mask, epsilon=1e-6):
    intersection = torch.sum(pred_mask * true_mask)
    union = torch.sum(pred_mask) + torch.sum(true_mask) - intersection
    iou = intersection / (union + epsilon)
    return iou

total_iou = 0
num_images = 0

with torch.no_grad():  # No gradients are calculated during testing
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images) # Forward pass
        # predictions = torch.sigmoid(outputs) # Apply sigmoid to outputs to get probabilities
        predicted_mask = (outputs > 0.5).float()  # Convert probabilities to binary predictions
        iou = compute_iou(predicted_mask, masks)
        total_iou += iou.item()
        num_images += 1
        print(f"IoU: {iou.item():.4f}")
        total_iou += iou.item()
        num_images += 1


if num_images > 0:
    avg_iou = total_iou / num_images
    print(f"\nAverage IoU over {num_images} images: {avg_iou:.4f}") #4 decimal 
else:
    print("No valid test images found.")