import numpy as np
from PIL import Image
from typing import Tuple

from albumentations import (Compose)
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset
import torch.nn as nn

class SiameseDataset(Dataset):
    def __init__(self, data: list[Tuple[str, str, int]], image_size: int = 1024):
        """Initialize the paired satellite image dataset.

        Args:
            images (list[Tuple[str, str]]): List of tuples containing pre and post image file paths.
        """
        self.data = data
        self.image_size = image_size
        self.transform = Compose([
            ToTensorV2(),
        ])

    def __len__(self) -> int:
        """Get length of dataset.

        Returns:
            len (int): Length of dataset.
        """
        return len(self.data) * (len(self.data) - 1) // 2 # Number of combinations of pairs (n choose 2)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset.

        Args:
            idx (int): Index of item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of pre and post patch tensors.
        """
        # Calculate indices for pair combinations
        n = len(self.data)
        row = idx // (n - 1)
        col = idx % (n - 1)

        if col >= row:
            col += 1

        (pre_img_path_1, post_img_path_1, label_1), (pre_img_path_2, post_img_path_2, label_2) = self.data[row], self.data[col]

        # Load and transform images
        pre_img_1 = np.array(Image.open(pre_img_path_1))
        post_img_1 = np.array(Image.open(post_img_path_1))
        pre_img_2 = np.array(Image.open(pre_img_path_2))
        post_img_2 = np.array(Image.open(post_img_path_2))

        pre_img_1 = self.transform(image=pre_img_1)['image']
        post_img_1 = self.transform(image=post_img_1)['image']
        pre_img_2 = self.transform(image=pre_img_2)['image']
        post_img_2 = self.transform(image=post_img_2)['image']

        # Resizing
        pre_img_1 = nn.functional.interpolate(pre_img_1.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        post_img_1 = nn.functional.interpolate(post_img_1.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        pre_img_2 = nn.functional.interpolate(pre_img_2.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        post_img_2 = nn.functional.interpolate(post_img_2.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)

        # Return both pairs and their labels
        # Label is 1 if both pairs are of the same class (both intact or both damaged)
        label = int(label_1 == label_2)
        label = torch.tensor(label)
        return pre_img_1.float(), post_img_1.float(), pre_img_2.float(), post_img_2.float(), label.float()