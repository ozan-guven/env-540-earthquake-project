# This file contains the implementation of the dataset to train the models.

import random
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Union, List, Dict, Optional

import torch
from torch.utils.data import Dataset
import cv2


class SegmentationDataset(Dataset):
    """
    Dataset of images with pre and post earthquake images and the corresponding mask for the post earthquake image.
    """

    def __init__(
        self,
        data: List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]],
        image_size: int = 1024,
        augment_data: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            data (List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]]): The list of dictionaries containing the paths to the images and masks
            image_size (int, optional): The size of the images, defaults to 1024
            augment_data (bool, optional): Whether to augment the data, defaults to False
        """
        if augment_data:
            self.transform = A.Compose(
                [
                    # Resize
                    A.Resize(image_size, image_size, interpolation=Image.BILINEAR),
                    # Color
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=1.0),
                            A.RandomGamma(p=1.0),
                        ],
                        p=0.5,
                    ),
                    # Blur
                    A.OneOf(
                        [
                            A.GaussianBlur(p=1.0),
                            A.CLAHE(p=1.0),
                        ],
                        p=0.5,
                    ),
                    # Noise
                    A.OneOf(
                        [
                            A.GaussNoise(p=1.0),
                            A.ISONoise(p=1.0),
                        ],
                        p=0.5,
                    ),
                    # Rotate, flip, scale, shift
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.05,
                        rotate_limit=90,
                        value=(0, 0, 0),
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.5,
                    ),
                    # Normalize
                    A.ToFloat(max_value=255.0),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=1,
                    ),
                    ToTensorV2(),
                ],
                additional_targets={"image2": "image"},
            )

        else:
            self.transform = A.Compose(
                [
                    # Resize
                    A.Resize(image_size, image_size, interpolation=Image.BILINEAR),
                    # Normalize
                    A.ToFloat(max_value=255.0),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                        max_pixel_value=1,
                    ),
                    ToTensorV2(),
                ],
                additional_targets={"image2": "image"},
            )

        # Merge data
        self.data = data

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset
        """
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item

        Returns:
            pre_image (torch.Tensor): The pre-earthquake image
            post_image (torch.Tensor): The post-earthquake image
            mask (torch.Tensor): The mask of the post-earthquake image
            label (float): The label of the post-earthquake image (1 if there are damages, i.e. the mask contains non-zero values, 0 otherwise)
        """
        # Get data tuple
        data_tuple = self.data[idx]

        # Get pre image, if there are multiple pre images, choose one at random
        pre_image_path = random.choice(data_tuple["pre"])

        # Get post image, if there are multiple post images, choose one at random
        post_image_path, mask_path = random.choice(data_tuple["post_mask"])

        # Get pre image
        pre_image = np.array(Image.open(pre_image_path).convert("RGB"))
        try:
            post_image = np.array(Image.open(post_image_path).convert("RGB"))
        except:
            print(f"❌ Error opening {post_image_path}, retrying...")
            return self.__getitem__(idx)
        # The mask might be None if it does not exist, in that case, we create a black image
        label = None
        if mask_path is not None:
            mask = np.array(Image.open(mask_path).convert("L")).astype(bool)
            label = 1.0
        else:
            mask = np.zeros(pre_image.shape[:2])
            label = 0.0
        mask = mask.astype(np.float32)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=pre_image, image2=post_image, mask=mask)
            pre_image = transformed["image"]
            post_image = transformed["image2"]
            mask = transformed["mask"]

        return pre_image, post_image, mask, label
