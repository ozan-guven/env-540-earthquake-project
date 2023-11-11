"""
    This file contains the dataset class for the satellite image dataset.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple

from albumentations import (
    Compose, OneOf,
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, 
    RandomBrightnessContrast, RandomGamma,
    GaussNoise, Blur, MotionBlur, MedianBlur,
    ToFloat
    )
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset

TRAIN_SPLIT = 0.9
TEST_SPLIT = 0.05

def get_split_image_files(
        directory: str, 
        train_split: float = TRAIN_SPLIT, 
        test_split: float = TEST_SPLIT
    ) -> Tuple[List[str], List[str], List[str]]:
    """Get split image files.

    Args:
        directory (str): Directory containing image files.
        train_split (float, optional): Train split. Defaults to TRAIN_SPLIT.
        test_split (float, optional): Test split. Defaults to TEST_SPLIT.

    Returns:
        train_image_files (List[str]): List of train image file paths.
        test_image_files (List[str]): List of test image file paths.
        val_image_files (List[str]): List of validation image file paths.
    """
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
    def __init__(self, image_files: List[str]):
        """Initialize the satellite image dataset.

        Args:
            image_files (List[str]): List of image file paths.
        """
        self.image_files = image_files

        self.transform = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.2, rotate_limit=90, border_mode=0),
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

    def __len__(self) -> int:
        """Get length of dataset.

        Returns:
            len (int): Length of dataset.
        """
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item from dataset.

        Args:
            idx (int): Index of item.

        Returns:
            patch (torch.Tensor): Patch tensor.
        """
        # Get image and apply transform
        image = np.array(Image.open(self.image_files[idx]))
        patch = self.transform(image=image)['image']

        return patch
