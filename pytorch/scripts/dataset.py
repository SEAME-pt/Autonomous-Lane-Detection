import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.Resize(height=144, width=256),  # Ensure consistent input size 
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.3),
    A.GaussianBlur(p=0.1),
    A.MotionBlur(p=0.1),
    A.HueSaturationValue(p=0.1),  # Added color jitter
    A.ElasticTransform(p=0.1),  # Added elastic transformation
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

test_transforms = A.Compose([
    A.Resize(height=144, width=256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class LaneDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        if self.mask_paths:
            mask = np.array(Image.open(self.mask_paths[idx]).convert("L")) 
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"].float()
            mask = augmented["mask"].float()
            mask = np.array(mask) > 0
            mask = mask.astype(np.float32)
            return image, mask, self.image_paths[idx] 
        else:
            augmented = self.transforms(image=image)  # Get full augmentation dict
            image = augmented["image"].float()
        return image, self.image_paths[idx]


image_paths = []
mask_paths = []
# mask_dir = os.path.join('.', 'training' ,'town4_images', 'train_label')
# image_dir = os.path.join('.','training' ,'town4_images', 'train') 
# for root, dirs, files in os.walk(image_dir):
#     for file in files:
#         image_path = os.path.join(root, file)
#         file_name, file_ext = os.path.splitext(file)
#         mask_file_name = f"{file_name}_label{file_ext}"
#         mask_path = os.path.join(mask_dir, os.path.relpath(os.path.join(root, mask_file_name), image_dir))
#         mask_paths.append(mask_path)
#         image_paths.append(image_path)

image_dir = os.path.join('..', 'training' ,'german_dataset') 
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            mask_path = image_path.replace('.jpg', '.png')
            if not os.path.exists(mask_path):  
                continue 
            mask_paths.append(mask_path)
            image_paths.append(image_path)

dataset = LaneDataset(image_paths, mask_paths, transforms=train_transforms)
train_loader = DataLoader(dataset, batch_size=6, shuffle=True,  num_workers=0, pin_memory=True)