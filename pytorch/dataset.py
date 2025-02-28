import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
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
        mask = Image.open(self.mask_paths[idx]).convert("L")  #expected output, grayscale
        if self.transforms:
            image, mask = self.transforms(image, mask)
        mask = np.array(mask) > 0
        mask = mask.astype(np.float32)
        mask = torch.tensor(mask, dtype=torch.float)
        # print(mask.unique())
        image = transforms.ToTensor()(image)
        mask = mask.unsqueeze(0).unsqueeze(0)  
        mask = F.interpolate(mask, size=(590, 590), mode='nearest')
        mask = mask.squeeze(0).squeeze(0)  # Remove batch dimension -> (C, H, W)
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=(590, 590), mode='nearest')
        image = image.squeeze(0)
        return image, mask, self.mask_paths[idx], self.image_paths[idx]

mask_dir = os.path.join('.', 'TUSimple', 'train_set', 'seg_label')
image_dir = os.path.join('.', 'TUSimple', 'train_set', 'clips') 
image_paths = []
mask_paths = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file == '20.jpg':
            image_path = os.path.join(root, file)
            mask_path = os.path.join(mask_dir, os.path.relpath(image_path, image_dir)).replace('.jpg', '.png')
            if not os.path.exists(mask_path):  
                continue  # Skip unmatched pairs
            mask_paths.append(mask_path)
            image_paths.append(image_path)


mask_dir = os.path.join('.', 'TUSimple', 'train_set', 'seg_label', '06040311_1063.MP4')
image_dir = os.path.join('.', 'TUSimple', 'train_set', 'clips', '06040311_1063.MP4') 
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            print(f'adding: {image_path}')
            mask_path = os.path.join(mask_dir, os.path.relpath(image_path, image_dir)).replace('.jpg', '.png')
            mask_paths.append(mask_path)
            image_paths.append(image_path)



dataset = LaneDataset(image_paths, mask_paths)
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)