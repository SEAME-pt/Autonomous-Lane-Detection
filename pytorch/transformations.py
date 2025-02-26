import torchvision.transforms as transforms
import torch
import random

class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, mask):
        if random.random() < self.prob:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        if random.random() < self.prob:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        return image, mask

class RandomRotation:
    def __init__(self, degrees=30):
        self.degrees = degrees
    
    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
        return image, mask

class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            if isinstance(t, transforms.ColorJitter):
                image = t(image)
            else:
                image, mask = t(image, mask)
        return image, mask

# Define augmentation pipeline
def get_transforms():
    return CustomCompose([
        RandomFlip(prob=0.5),
        RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0)
    ])
