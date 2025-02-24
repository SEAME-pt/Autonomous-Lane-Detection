import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os


class LaneDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]) #rgb
        mask = Image.open(self.mask_paths[idx])  #expected output, grayscale
        mask = np.array(mask) > 0
        mask = mask.astype(np.float32)
        if self.transforms:
            image = self.transforms(image)
        mask = torch.tensor(mask, dtype=torch.float)
        image = transforms.ToTensor()(image)
        return image, mask, self.mask_paths[idx], self.image_paths[idx]

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
                continue  # Skip unmatched pairs
            mask_paths.append(mask_path)
            image_paths.append(image_path)


def get_transforms():
    return transforms.Compose([
        # transforms.RandomHorizontalFlip(),  # Random horizontal flip
        # transforms.RandomRotation(15),      # Random rotation
        transforms.RandomResizedCrop(1640, scale=(0.8, 1.0)),  # Random resizing
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
        # transforms.ToTensor(),  # Convert to tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for RGB images
    ])

# Pass the transforms to your dataset
transform = get_transforms()

dataset = LaneDataset(image_paths, mask_paths, transform)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)