import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
from model import LaneNet
from loss import CombinedLoss 
from dataset import LaneDataset, train_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_paths = []
mask_paths = []

image_dir = os.path.join('..','training' ,'german_dataset') 
for root, dirs, files in os.walk(image_dir):
    if 'dataset10' in root:
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                mask_path = image_path.replace('.jpg', '.png')
                if not os.path.exists(mask_path):  
                    continue 
                mask_paths.append(mask_path)
                image_paths.append(image_path)


dataset = LaneDataset(image_paths, mask_paths, transforms=train_transforms)
train_loader = DataLoader(dataset, batch_size=6, shuffle=True,  num_workers=0, pin_memory=True)
model = LaneNet().to(device)

loss_function = CombinedLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
iou_metric = JaccardIndex(task="binary", num_classes=1).to(device)

checkpoint_path = "../models/model_26.pth"
best_iou = -float('inf')  
best_loss = float('inf')
retrain_path = ",./models/retrain.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if torch.is_tensor(image):
        denorm_image = image.clone().cpu().numpy()
    else:
        denorm_image = image.copy()
    if denorm_image.shape[0] == 3: # Ensure the image is in (C, H, W) format
        denorm_image = denorm_image.transpose(1, 2, 0)  # Change to (H, W, C)
    denorm_image = denorm_image * np.array(std) + np.array(mean) # Denormalize
    # Clip values to [0, 1] range
    return np.clip(denorm_image, 0, 1)

def matplot_masks(images, masks, predicted_mask, path):
    img = images.squeeze().cpu().numpy()
    img = np.transpose(img, (1, 2, 0)) 
    denorm_img = denormalize(img) 
    true_mask = masks.squeeze().cpu().numpy()
    pred_mask = predicted_mask.squeeze().cpu().numpy()
    plt.style.use('grayscale')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    ax1.imshow(denorm_img)
    ax1.set_title('Image')
    ax1.axis('off')
    ax2.imshow(true_mask, cmap='gray')
    ax2.set_title('Truth Mask')
    ax2.axis('off')
    ax3.imshow(pred_mask, cmap='gray')
    ax3.set_title('Predicted Mask')
    ax3.axis('off')
    plt.tight_layout()
    plt.savefig(f'../debug/training_img/{epoch + 1}_{i}.png')
    plt.close()
    gc.collect() 
    print(f"Iou: {iou.item():.4f}, Loss: {loss.item():.4f}, Iter: {i}, path: {path}")

scaler = torch.amp.GradScaler()
save_dir = "../models/"
i = 0
model.train()
for epoch in range(0, 50):
    running_loss = 0.0
    running_iou = 0.0
    for images, masks, paths in train_loader:
        i += 1
        images, masks = images.to(device), masks.to(device)
        masks = masks.unsqueeze(1)

        optimizer.zero_grad()  
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            torch.cuda.empty_cache()
            outputs = model(images)  
            loss = loss_function(outputs, masks)  
            predicted_probs = torch.sigmoid(outputs)  
        predicted_mask = (predicted_probs > 0.5).float()  
        iou = iou_metric(predicted_mask, masks)

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize() 

        running_loss += loss.item()  
        running_iou += iou.item()
        if i % 100 == 0:
            matplot_masks(images[0:1], masks[0:1], predicted_mask[0:1], paths[0])  
        del outputs, predicted_probs, predicted_mask
    avg_loss = running_loss / len(train_loader)
    avg_iou = running_iou / len(train_loader)
    if epoch > 5 and (avg_loss < best_loss or avg_iou > best_iou):
        best_loss = avg_loss
        best_iou = avg_iou
        save_path = os.path.join(save_dir, f"best_model.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.cpu().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_iou": best_iou,
            "best_loss": best_loss
        }, retrain_path)
        model.to(device)
        print(f"Best model saved (Epoch {epoch+1}, IOU: {avg_iou:.4f}, Loss: {avg_loss:.4f})")

    if (epoch + 1) % 5 == 0:
        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.cpu().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_iou": best_iou,
            "best_loss": best_loss
        }, retrain_path)
        model.to(device)
        print(f" Checkpoint saved at epoch {epoch+1}")
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, IOU: {avg_iou:.4f}")
    print(torch.cuda.memory_summary(abbreviated=False))
    print(f"RAM Usage: {psutil.Process().memory_info().rss/1e9:.2f} GB")
    scheduler.step()
