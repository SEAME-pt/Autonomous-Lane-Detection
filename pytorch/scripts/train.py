import torch
import torch.nn as nn
import torch.optim as optim
from model import LaneNet
from loss import CombinedLoss
from dataset import LaneDataset, train_loader
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import gc

device = torch.device("cuda")
model = LaneNet().to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) #adam optimizer for image segmentation  all learnable parameters, learning rate
loss_function = CombinedLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
iou_metric = JaccardIndex(task="binary", num_classes=1).to(device)

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

best_iou = -float('inf')  # Start with a very low IoU
best_loss = float('inf') 
i = 0
scaler = torch.amp.GradScaler()
for epoch in range(0, 50):
    running_loss = 0.0
    running_iou = 0.0
    for images, masks, paths in train_loader:
        i += 1
        images, masks = images.to(device), masks.to(device)
        masks = masks.unsqueeze(1)
        optimizer.zero_grad() # Zero the gradients
        with torch.amp.autocast(device_type='cuda'):
            torch.cuda.empty_cache()
            outputs = model(images) #calls forward()
            loss = loss_function(outputs, masks) #loss is a tensor
            predicted_probs = torch.sigmoid(outputs) # convert to 0 - 1, activation function, decides whats important, probability
        predicted_mask = (predicted_probs > 0.5).float() 
        iou = iou_metric(predicted_mask, masks)
        #gradients indicate how much each parameter should be adjusted to minimize the loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize() 
        running_loss += loss.item() 
        running_iou += iou.item()
        if i % 200 == 0:
            matplot_masks(images[0:1], masks[0:1], predicted_mask[0:1], paths[0]) 
        del outputs, predicted_probs, predicted_mask
    if epoch > 10 and (running_loss / len(train_loader) < best_loss or running_iou / len(train_loader) > best_iou):
        best_loss = running_loss / len(train_loader)
        best_iou = running_iou / len(train_loader)
        name = f"best_{epoch + 1}.pth"
        save_dir = "../models/best_models/"
        save_path = os.path.join(save_dir, name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
            'best_loss': best_loss,
        }, save_path)
        model.to(device)
        print("best model saved")
    # elif (epoch + 1) % 2 == 0:
    name = f"model_{epoch + 1}.pth"
    save_dir = "../models/"
    save_path = os.path.join(save_dir, name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.cpu().state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_iou,
        'best_loss': best_loss,
    }, save_path)
    model.to(device)
    print("model saved")
    print(f"\n Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, IOU: {running_iou / len(train_loader)}") # Print the average loss and iou for this epoch
    print(torch.cuda.memory_summary(abbreviated=False))
    print(f"Current RAM: {psutil.Process().memory_info().rss/1e9:.2f} GB")
    scheduler.step()