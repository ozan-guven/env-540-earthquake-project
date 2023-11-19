from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from albumentations import (Blur, Compose, GaussNoise, HorizontalFlip,
                            MedianBlur, MotionBlur, OneOf,
                            RandomBrightnessContrast, RandomGamma,
                            ShiftScaleRotate, ToFloat, VerticalFlip)
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(
            self, 
            data: list[Tuple[str, str, int]], 
            image_size: int = 1024,
            augment_data: bool = False
            ):
        """Initialize the paired satellite image dataset.

        Args:
            images (list[Tuple[str, str]]): List of tuples containing pre and post image file paths.
        """
        self.data = data
        self.image_size = image_size
        if augment_data:
            self.transform = Compose([
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.1, rotate_limit=20, border_mode=0),
                RandomBrightnessContrast(p=0.5),
                RandomGamma(p=0.5),
                OneOf([
                    GaussNoise(p=1.0),
                    Blur(p=1.0),
                    MotionBlur(p=1.0),
                    MedianBlur(p=1.0),
                ], p=0.5),
                ToFloat(max_value=255),
                ToTensorV2(),
            ],
            additional_targets={'image2': 'image'})
        else:
            self.transform = Compose([
                ToFloat(max_value=255),
                ToTensorV2(),
            ],
            additional_targets={'image2': 'image'})

    def __len__(self) -> int:
        """Get length of dataset.

        Returns:
            len (int): Length of dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset.

        Args:
            idx (int): Index of item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of pre and post patch tensors.
        """
        # Get pre and post file paths
        pre_path = self.data[idx][0]
        post_path = self.data[idx][1]
        label = self.data[idx][2]

        # Load and transform images
        pre = np.array(Image.open(pre_path))
        post = np.array(Image.open(post_path))

        data = self.transform(image=pre, image2=post)
        pre = data['image']
        post = data['image2']

        # Resizing
        pre = nn.functional.interpolate(pre.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        post = nn.functional.interpolate(post.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)

        # Return both pairs and their labels
        label = torch.tensor(label)

        return pre, post, label.float()