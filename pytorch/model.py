import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


#more filters mean the model can detect more patterns in the image
#binary classification
class LaneNet(nn.Module): #neural network
    def __init__(self):
        super(LaneNet, self).__init__() #initialize and inherit the necessary functionality from parent class 
        self.encoder = nn.Sequential( # feature extraction
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # Extracts features from the image by applying filters (kernels)
            nn.Conv2d(64, 128, kernel_size=3, padding=1), #Learn spatial patterns
            nn.MaxPool2d(2) #reduce size & computation by half, downsample
        )
        self.decoder = nn.Sequential( #Reconstructs the output
            nn.Conv2d(128, 64, kernel_size=3, padding=1), #refine details
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1), #binary mask
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x) # non-linearity so the network can learn complex patterns
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  #Resizes to a larger size, upsampling
        x = self.decoder(x)
        return x