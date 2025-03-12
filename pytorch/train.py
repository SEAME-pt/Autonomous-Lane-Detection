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

device = torch.device("cuda")
model = LaneNet().to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) #adam optimizer for image segmentation  all learnable parameters, learning rate
loss_function = CombinedLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
iou_metric = JaccardIndex(task="binary", num_classes=1).to(device)

def matplot_masks(images, masks, predicted_mask, path):
    img = images.squeeze().cpu().numpy()
    img = np.transpose(img, (1, 2, 0)) 
    true_mask = masks.squeeze().cpu().numpy()
    pred_mask = predicted_mask.squeeze().cpu().numpy()
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title('Image')
    ax1.axis('off')
    ax2.imshow(true_mask, cmap='gray')
    ax2.set_title('Truth Mask')
    ax2.axis('off')
    ax3.imshow(pred_mask, cmap='gray')
    ax3.set_title('Predicted Mask')
    ax3.axis('off')
    plt.tight_layout()
    plt.savefig(f'./debug_img/{epoch}_{i}.png')
    plt.close()
    print(f"Iou: {iou.item():.4f}, Loss: {loss.item():.4f}, Iter: {i}, path: {path}")

i = 0
epochs = 35 #One epoch is completed when the model has seen every sample in the dataset once
scaler = torch.amp.GradScaler()
for epoch in range(epochs):
    torch.cuda.empty_cache()
    running_loss = 0.0
    running_iou = 0.0
    for images, masks, path, paths in train_loader:
        i += 1
        images, masks = images.to(device), masks.to(device)
        masks = masks.unsqueeze(1)
        optimizer.zero_grad() # Zero the gradients
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(images) #calls forward()
            loss = loss_function(outputs, masks) #loss is a tensor
            predicted_probs = torch.sigmoid(outputs) # convert to 0 - 1, activation function, decides whats important, probability
        predicted_mask = (predicted_probs > 0.5).float() 
        iou = iou_metric(predicted_mask, masks)
        # loss.backward() #gradients indicate how much each parameter should be adjusted to minimize the loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() 
        running_iou += iou.item()
        if i % 60 == 0:
            matplot_masks(images, masks, predicted_mask, path)
        # optimizer.step()
    print(f"\n Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, IOU: {running_iou / len(train_loader)}") # Print the average loss and iou for this epoch
    print(torch.cuda.memory_summary(abbreviated=False))
    print(f"Current RAM: {psutil.Process().memory_info().rss/1e9:.2f} GB")
    scheduler.step()

torch.save(model.cpu().state_dict(), "lanenet_model6.pth")