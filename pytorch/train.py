import torch
import torch.nn as nn
import torch.optim as optim
from model import LaneNet
from loss import CombinedLoss
from dataset import LaneDataset, train_loader
import os
from torchmetrics import JaccardIndex

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda")
torch.cuda.empty_cache()
model = LaneNet().to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=0.001) #adam optimizer for image segmentation  all learnable parameters, learning rate
# loss_function = nn.BCELoss() #binary cross-entropy loss, difference between predicted loss_function distribution and true labels
loss_function = CombinedLoss()

iou_metric = JaccardIndex(task="binary", num_classes=1).to(device)

i = 0
epochs = 10 #One epoch is completed when the model has seen every sample in the dataset once
for epoch in range(epochs):
    running_loss = 0.0
    running_iou = 0.0
    for images, masks in train_loader:
        print(f"Epoch: {epoch}")
        i += 1
        print(f"Iteration: {i}")
        images, masks = images.to(device), masks.to(device)
        masks = masks.unsqueeze(1)
        print(f"Image shape: {images.shape}, Mask shape: {masks.shape}")
        optimizer.zero_grad() # Zero the gradients
        outputs = model(images) #calls forward()
        print(f"Outputs: {outputs.max()}, {outputs.min()}")
        loss = loss_function(outputs, masks) #loss is a tensor
        predicted_mask = (outputs > 0.5).float() 
        iou = iou_metric(predicted_mask, masks)
        print(f"Iou: {iou}")
        print(f"Loss: {loss}")
        loss.backward() #gradients indicate how much each parameter should be adjusted to minimize the loss
        running_loss += loss.item() 
        running_iou += iou.item()
        optimizer.step()
    print(f"\n Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, IOU: {running_iou / len(train_loader)}") # Print the average loss and iou for this epoch

print(model)

torch.save(model.state_dict(), 'lanenet_model.pth')