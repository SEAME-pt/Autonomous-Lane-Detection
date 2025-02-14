import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class LaneDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]) #rgb
        print(f"Dataset image: {self.image_paths[idx]}")
        print(f"Array image: {np.array(image)}")
        mask = Image.open(self.mask_paths[idx])  #expected output, grayscale
        mask = np.array(mask) > 0  
        mask = mask.astype(np.float32)
        print(f"Array mask: {mask}")
        mask = torch.tensor(mask, dtype=torch.float)
        image = transforms.ToTensor()(image)
        print(f"Image tensor: {image}")
        print(f"Mask tensor: {mask}")
        return image, mask

mask_dir = os.path.join('.', 'laneseg_label_w16', 'driver_161_90frame')
image_dir = os.path.join('.', 'driver_161_90frame') 
image_paths = []
mask_paths = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            mask_path = os.path.join(mask_dir, os.path.relpath(image_path, image_dir)).replace('.jpg', '.png')
            if not os.path.exists(mask_path):  
                print(f"WARNING: Mask not found for {image_path}")
                continue  # Skip unmatched pairs
            mask_paths.append(mask_path)
            image_paths.append(image_path)

dataset = LaneDataset(image_paths, mask_paths)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)