import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict, deque

from albumentations import (
    Compose, OneOf,
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, 
    RandomBrightnessContrast, RandomGamma,
    GaussNoise, Blur, MotionBlur, MedianBlur,
    MultiplicativeNoise,
    ToFloat
    )
from albumentations.pytorch import ToTensorV2

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def get_split_image_files(directory, train_split=0.9, test_split=0.05, val_split=0.05):
    # Get image files
    image_files = [os.path.join(directory, file) for file in os.listdir(directory)]

    # Get number of images
    n_images = len(image_files)
    n_train = int(n_images * train_split)
    n_test = int(n_images * test_split)

    # Shuffle the image files
    np.random.shuffle(image_files)

    # Split the image files
    train_image_files = image_files[:n_train]
    test_image_files = image_files[n_train:n_train+n_test]
    val_image_files = image_files[n_train+n_test:]

    return train_image_files, test_image_files, val_image_files

class SatelliteImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files

        self.transform = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.2, rotate_limit=90),
            RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
            RandomGamma(p=0.5, gamma_limit=(80, 120)),
            OneOf([
                GaussNoise(p=1.0),
                Blur(p=1.0),
                MotionBlur(p=1.0),
                MedianBlur(p=1.0),
            ], p=0.5),
            ToFloat(max_value=255),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image
        image = np.array(Image.open(self.image_files[idx]))

        # Apply transform
        patch = self.transform(image=image)['image']

        return patch
