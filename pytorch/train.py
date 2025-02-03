import torch
import torch.nn as nn
import torch.optim as optim
from model import LaneNet, model
from dataset import LaneDataset, train_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LaneNet().to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=0.001) # all learnable parameters, learning rate
probability = nn.BCEWithLogitsLoss #sigmoid activation and binary cross-entropy loss, difference between predicted probability distribution and true labels

epochs = 10 #One epoch is completed when the model has seen every sample in the dataset once
for epoch in range(epochs):
    running_loss = 0.0
    for images, masks in train_loader:
        # Move to GPU if available
        images, masks = images.to(device), masks.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        outputs = model(images) #calls forward()
        # Compute the loss
        loss = probability(outputs, masks) #loss is a Tensor
        running_loss += loss.item() 
        # Backward pass and optimization
        loss.backward() #gradients indicate how much each parameter should be adjusted to minimize the loss
        optimizer.step()
    # Print the average loss for this epoch
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

print(model)