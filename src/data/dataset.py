import os
from PIL import Image
from collections import OrderedDict

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

DATA_PATH = '../../data/'

class SatelliteImageDataset(Dataset):
    def __init__(self, directory, image_size=17408, patch_size=1024, max_cache_size=256):
        self.image_size = image_size
        if image_size % patch_size != 0:
            raise ValueError('❌ Image size must be divisible by patch size')
        
        self.patch_size = patch_size
        self.n_patches_per_image = (image_size // patch_size) ** 2
        # List files in directory (pre or post)
        self.image_files = []
        for query_key_dir in os.listdir(directory):
            query_key_dir = os.path.join(directory, query_key_dir)
            for file in os.listdir(query_key_dir):
                self.image_files.append(os.path.join(query_key_dir, file))

        print(f"📝 Found {len(self.image_files)} images")

        # Define cache
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size

    def __len__(self):
        return len(self.image_files) * self.n_patches_per_image

    def __getitem__(self, idx):
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

        # Get patch
        image_idx = idx % self.n_patches_per_image
        patches_per_row = self.image_size // self.patch_size
        patch_x = (image_idx % patches_per_row) * self.patch_size
        patch_y = (image_idx // patches_per_row) * self.patch_size
        patch = image.crop((patch_x, patch_y, patch_x + self.patch_size, patch_y + self.patch_size))

        image = image.resize((1024, 1024))

        return image, patch
