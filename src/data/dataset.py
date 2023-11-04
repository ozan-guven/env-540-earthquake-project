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

def get_split_image_files(directory, train_split=0.7, test_split=0.15, val_split=0.15):
    image_files = []
    for query_key_dir in os.listdir(directory):
        query_key_dir = os.path.join(directory, query_key_dir)
        for file in os.listdir(query_key_dir):
            image_files.append(os.path.join(query_key_dir, file))

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
    def __init__(
            self,
            image_files,
            image_size=17_408,
            patch_size=1024,
            max_cache_size=32,
            max_used_indices=100_000,
            max_black_ratio=0.8,
            ):
        self.image_files = image_files
        self.image_size = image_size
        if image_size % patch_size != 0:
            raise ValueError('❌ Image size must be divisible by patch size')
        self.patch_size = patch_size
        self.n_patches_per_image = (image_size // patch_size) ** 2
        self.max_black_ratio = max_black_ratio

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

        # Define cache
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size

        # Define used indices to avoid duplicates
        self.used_indices = set()
        self.used_indices_order = deque(maxlen=max_used_indices)
        self.max_used_indices = max_used_indices

    def __len__(self):
        return len(self.image_files) * self.n_patches_per_image
    
    def _get_image(self, idx):
        image_idx = idx // self.n_patches_per_image

        # Get image
        if image_idx not in self.cache:
            # Load image from disk if not in cache and add to cache
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)
            self.cache[image_idx] = Image.open(self.image_files[image_idx])
        else:
            # Move image to end if found in cache
            self.cache.move_to_end(image_idx)
        image = self.cache[image_idx]

        # Check image dimensions
        if image.size[0] != image.size[1]:
            raise ValueError('❌ Image must be square')
        if image.size[0] != self.image_size:
            raise ValueError('❌ Image must be of size {}'.format(self.image_size))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

    
    def _get_patch(self, idx):
        # Get image
        image = self._get_image(idx)

        # Get patch
        patch_idx = idx % self.n_patches_per_image
        patches_per_row = self.image_size // self.patch_size

        patch_x = (patch_idx % patches_per_row) * self.patch_size
        patch_y = (patch_idx // patches_per_row) * self.patch_size
        patch = image.crop((patch_x, patch_y, patch_x + self.patch_size, patch_y + self.patch_size))

        return patch
    
    def _update_used_indices(self, idx):
        # Add current index to used indices
        self.used_indices.add(idx)
        self.used_indices_order.append(idx)

        # Remove oldest index if necessary
        if len(self.used_indices) > self.max_used_indices:
            oldest_idx = self.used_indices_order.popleft()
            self.used_indices.remove(oldest_idx)

    def _reset_used_indices(self):
        self.used_indices = set()
        self.used_indices_order = deque(maxlen=self.max_used_indices)

    def __getitem__(self, idx):
        while True:
            # Skip used indices
            if idx in self.used_indices:
                idx = (idx + 1) % self.__len__()
                continue

            # Get patch
            patch = self._get_patch(idx)

            # Skip patch if too many black pixels
            n_total_pixels = self.patch_size * self.patch_size
            n_black_pixels = np.sum(np.sum(np.array(patch), axis=2) == 0)
            black_ratio = n_black_pixels / n_total_pixels
            if black_ratio > self.max_black_ratio:
                # Update used indices and continue
                self._update_used_indices(idx)
                idx = (idx + 1) % len(self)
                # Reset used indices if we have gone through all of the dataset
                if idx == 0:
                    self._reset_used_indices()
                continue

            # Apply transform
            patch = self.transform(image=np.array(patch))['image']

            self._update_used_indices(idx)
            return patch
