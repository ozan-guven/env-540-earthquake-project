from typing import Tuple

import random
from torch.utils.data import Dataset

import torch

from src.data.segmentation_dataset import SegmentationDataset


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning.
    """
    def __init__(
        self,
        data: list[Tuple[str, str, int]],
        image_size: int = 1024,
        augment_data: bool = False,
        random_sample: bool = True,
    ):
        """
        Initialize the paired satellite image dataset.

        Args:
            data (list[Tuple[str, str, int]]): List of tuples containing pre and post file paths and label
            image_size (int, optional): The size of image, defaults to 1024
            augment_data (bool, optional): Whether to augment data, defaults to False
            random_sample (bool, optional): Whether to randomly sample data, otherwise iterate through all pairs, defaults to True.
        """
        self.random_sample = random_sample
        self.dataset1 = SegmentationDataset(data, image_size, augment_data)
        self.dataset2 = SegmentationDataset(data, image_size, augment_data)

    def __len__(self) -> int:
        """
        Get length of dataset.

        Returns:
            len (int): Length of dataset.
        """
        return len(self.dataset1) * len(self.dataset2)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.

        Args:
            idx (int): The index of item.

        Returns:
            torch.Tensor: First pair of images
            torch.Tensor: Second pair of images
            None: Not used
            torch.Tensor: Label, 1 if different, 0 if same
        """
        # Get random indices
        if self.random_sample:
            idx1 = random.randint(0, len(self.dataset1) - 1)
            idx2 = random.randint(0, len(self.dataset2) - 1)
        else:
            idx1 = idx // len(self.dataset2)
            idx2 = idx % len(self.dataset2)

        # Get pairs
        pre1, post1, _, label1 = self.dataset1[idx1]
        pre2, post2, _, label2 = self.dataset2[idx2]

        pair1 = torch.cat([pre1, post1], dim=0)
        pair2 = torch.cat([pre2, post2], dim=0)

        # Get label
        label = int(label1 != label2) 
        label = torch.tensor(label)

        return pair1, pair2, torch.tensor([0]), label