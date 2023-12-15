# This file implements the dataset of images
# with pre and post earthquake images
# and the corresponding mask for the post earthquake image

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple, Union

import Albumentations as A

class ImageDataset(Dataset):
    def __init__(self, 
        pre_path: str, 
        post_path: str, 
        mask_path: str, 
        transform: Union[transforms.Compose, A.Compose] = None
    ):
        """
        Dataset of images with pre and post earthquake images
        and the corresponding mask for the post earthquake image

        Args:
            pre_path (str): Path to the pre earthquake images
            post_path (str): Path to the post earthquake images
            mask_path (str): Path to the masks
            transform (transforms.Compose | A.Compose): Transformations to be applied to the images
        """
        self.pre_path = pre_path
        self.post_path = post_path
        self.mask_path = mask_path
        self.transform = transform

        self.pre_images = os.listdir(self.pre_path)
        self.post_images = os.listdir(self.post_path)
        self.masks = os.listdir(self.mask_path)

        self.pre_images.sort()
        self.post_images.sort()
        self.masks.sort()

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_image = Image.open(os.path.join(self.pre_path, self.pre_images[idx]))
        post_image = Image.open(os.path.join(self.post_path, self.post_images[idx]))
        mask = Image.open(os.path.join(self.mask_path, self.masks[idx]))

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
            mask = self.transform(mask)

        return pre_image, post_image, mask