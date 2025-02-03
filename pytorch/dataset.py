import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class LaneDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = Image.open(self.mask_paths[idx])  #expected output
        label = np.array(label) 
        label = np.array(label) 
        label[label > 0] = 1  # Set all non-zero pixels to 1 (lane pixels)
        label[label == 0] = 0
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
    

#data augmentation
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5), #probability of 50%
    transforms.RandomRotation(degrees=(-10,10)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])

image_paths = os.path.join('.', 'laneseg_label_w16/driver_161_90frame/')
mask_paths = os.path.join('.', 'driver_161_90frame/') 
dataset = LaneDataset(image_paths, mask_paths, transforms)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)