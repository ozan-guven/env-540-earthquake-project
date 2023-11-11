import sys
sys.path.append('../../')

import os
import numpy as np
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.trainer import Trainer
from src.siamese.siamese_dataset import SiameseDataset
from src.siamese.siamese_network import SiameseNetwork

DATA_PATH = '../../data/'
MAXAR_REVIEWED_PATCHES_PATH = DATA_PATH + 'maxar_reviewed_patches/'
MAXAR_INTACT_PATCHES_PATH = MAXAR_REVIEWED_PATCHES_PATH + '/intact/'
MAXAR_DAMAGED_PATCHES_PATH = MAXAR_REVIEWED_PATCHES_PATH + '/damaged/'
MAXAR_PRE_FOLDER = 'pre/'
MAXAR_POST_FOLDER = 'post/'

TRAIN_SPLIT = 0.9
TEST_SPLIT = 0.05

IMAGE_SIZE = 1024

EPOCHS = 1
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
ACCUMULATION_STEPS = 16
NUM_WORKERS = 4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _get_short_patch_file_name(file: str) -> str:
    """Get short patch file name.

    Args:
        file (str): File path.

    Returns:
        str: Short patch file name.
    """
    short_file = file.split('/')[-1]
    short_file = short_file.split('_')[0] + '_' + short_file.split('_')[3]
    short_file = short_file.split('.')[0]
    return short_file

def _get_split_paired_satellite_image_files(
        train_split: float = TRAIN_SPLIT, 
        test_split: float = TEST_SPLIT
    ) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    # Get pre and post file names
    intact_pre_files = [f"{MAXAR_INTACT_PATCHES_PATH}{MAXAR_PRE_FOLDER}{file}" for file in os.listdir(MAXAR_INTACT_PATCHES_PATH + MAXAR_PRE_FOLDER)]
    intact_post_files = [f"{MAXAR_INTACT_PATCHES_PATH}{MAXAR_POST_FOLDER}{file}" for file in os.listdir(MAXAR_INTACT_PATCHES_PATH + MAXAR_POST_FOLDER)]
    damaged_pre_files = [f"{MAXAR_DAMAGED_PATCHES_PATH}{MAXAR_PRE_FOLDER}{file}" for file in os.listdir(MAXAR_DAMAGED_PATCHES_PATH + MAXAR_PRE_FOLDER)]
    damaged_post_files = [f"{MAXAR_DAMAGED_PATCHES_PATH}{MAXAR_POST_FOLDER}{file}" for file in os.listdir(MAXAR_DAMAGED_PATCHES_PATH + MAXAR_POST_FOLDER)]

    # Get pre and post short file names
    short_intact_pre_files = [_get_short_patch_file_name(file) for file in intact_pre_files]
    short_intact_post_files = [_get_short_patch_file_name(file) for file in intact_post_files]
    short_damaged_pre_files = [_get_short_patch_file_name(file) for file in damaged_pre_files]
    short_damaged_post_files = [_get_short_patch_file_name(file) for file in damaged_post_files]

    # Match pre and post files by short file name
    intact_dict = {}
    for pre_idx in range(len(intact_pre_files)):
        short_file = short_intact_pre_files[pre_idx]
        post_idx = short_intact_post_files.index(short_file)

        # Duplicate short file names are appended with _1, _2, etc.
        i = 1
        while short_file in intact_dict:
            short_file = short_file + f'_{i}'
            i += 1

        # Add to dictionary
        intact_dict[short_file] = (intact_pre_files[pre_idx], intact_post_files[post_idx], 0)

    damaged_dict = {}
    for pre_idx in range(len(damaged_pre_files)):
        short_file = short_damaged_pre_files[pre_idx]
        post_idx = short_damaged_post_files.index(short_file)

        # Duplicate short file names are appended with _1, _2, etc.
        i = 1
        while short_file in damaged_dict:
            short_file = short_file + f'_{i}'
            i += 1
        
        # Add to dictionary
        damaged_dict[short_file] = (damaged_pre_files[pre_idx], damaged_post_files[post_idx], 1)

    # Make dataset balanced and get random samples
    n_intact = len(intact_dict)
    n_damaged = len(damaged_dict)
    n_samples = min(n_intact, n_damaged)

    intact_keys = list(intact_dict.keys())
    np.random.shuffle(intact_keys)
    intact_keys = intact_keys[:n_samples]
    intact_dict = {key: intact_dict[key] for key in intact_keys}

    damaged_keys = list(damaged_dict.keys())
    np.random.shuffle(damaged_keys)
    damaged_keys = damaged_keys[:n_samples]
    damaged_dict = {key: damaged_dict[key] for key in damaged_keys}

    # Combine dictionaries and shuffle
    data_dict = {**intact_dict, **damaged_dict}
    data_keys = list(data_dict.keys())
    np.random.shuffle(data_keys)
    data_dict = {key: data_dict[key] for key in data_keys}

    # Split the data
    n_data = len(data_dict)
    n_train = int(n_data * train_split)
    n_test = int(n_data * test_split)

    train_data = list(data_dict.values())[:n_train]
    test_data = list(data_dict.values())[n_train:n_train+n_test]
    val_data = list(data_dict.values())[n_train+n_test:]

    return train_data, test_data, val_data

def _get_dataloaders():
    train_data, test_data, val_data = _get_split_paired_satellite_image_files()

    train_dataset = SiameseDataset(train_data, image_size=IMAGE_SIZE)
    test_dataset = SiameseDataset(test_data, image_size=IMAGE_SIZE)
    val_dataset = SiameseDataset(val_data, image_size=IMAGE_SIZE)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f'✅ Train dataloader length: {len(train_dataloader)}')
    print(f'✅ Test dataloader length: {len(test_dataloader)}')
    print(f'✅ Val dataloader length: {len(val_dataloader)}')

    return train_dataloader, test_dataloader, val_dataloader

def get_model():
    return SiameseNetwork(
        image_size = IMAGE_SIZE,
        input_channels=3,
        conv_channels=[16, 32],
        fc_layers=[512, 128],
        activation = nn.ReLU(inplace=True),
        embedding_size = 50
    ).to(DEVICE)

def get_criterion():
    return torch.nn.BCELoss()

def get_optimizer(siamese, learning_rate):
    return torch.optim.Adam(siamese.parameters(), lr=learning_rate)

def get_trainer(model, criterion):
    return Trainer(
        model=model, 
        device=DEVICE,
        criterion=criterion,
        accumulation_steps=ACCUMULATION_STEPS,
        print_statistics=True
    )

if __name__ == '__main__':
    train_dataloader, _, val_dataloader = _get_dataloaders()
    siamese = get_model()
    criterion = get_criterion()
    optimizer = get_optimizer(siamese, LEARNING_RATE)
    trainer = get_trainer(siamese, criterion)
    statistics = trainer.train_siamese(
        train_loader=train_dataloader, 
        val_loader=val_dataloader,
        optimizer=optimizer,
        num_epochs=EPOCHS
    )
    
