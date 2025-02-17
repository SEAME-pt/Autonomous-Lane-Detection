import torch
import torch.nn as nn
import torch.optim as optim
from model import LaneNet
from dataset import LaneDataset, train_loader
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda")
torch.cuda.empty_cache()
model = LaneNet().to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=0.001) # all learnable parameters, learning rate
loss_function = nn.BCELoss() #binary cross-entropy loss, difference between predicted loss_function distribution and true labels

int i = 0
epochs = 5 #One epoch is completed when the model has seen every sample in the dataset once
for epoch in range(epochs):
    running_loss = 0.0
    for images, masks, image_paths, mask_paths in train_loader:
        print(f"Epoch: {epoch}")
        print(f"Iteration: {++i}")
        images, masks = images.to(device), masks.to(device)
        masks = masks.unsqueeze(1)
        print(f"Image shape: {images.shape}, Mask shape: {masks.shape}")
        optimizer.zero_grad() # Zero the gradients
        outputs = model(images) #calls forward()
        print(f"Outputs: {outputs.max()}, {outputs.min()}")
        loss = loss_function(outputs, masks) #loss is a tensor
        print(f"Loss: {loss}")
        loss.backward() #gradients indicate how much each parameter should be adjusted to minimize the loss
        running_loss += loss.item() 
        optimizer.step()
    print(f"\n Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}") # Print the average loss for this epoch

print(model)

torch.save(model.state_dict(), 'lanenet_model.pth')