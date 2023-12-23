import sys
sys.path.append('../../')

import os
import numpy as np
from typing import List, Tuple


import torch
from torch.utils.data import DataLoader

from src.contrastive.contrastive_dataset import ContrastiveDataset
from src.contrastive.contrastive_loss import ContrastiveLoss
from src.models.siamese import Siamese
from src.utils.trainer import Trainer

DATA_PATH = '../../data/'
MAXAR_REVIEWED_PATCHES_PATH = 'maxar_reviewed_patches/'
MAXAR_INTACT_PATCHES_PATH = 'intact/'
MAXAR_DAMAGED_PATCHES_PATH = 'damaged/'
MAXAR_PRE_FOLDER = 'pre/'
MAXAR_POST_FOLDER = 'post/'

TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.15

IMAGE_SIZE = 1024

EPOCHS = 50
LEARNING_RATE = 1e-3 #1e-5
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
EVALUATION_STEPS = 100
DROPOUT_RATE = 0.0
SIAMESE_CONV_CHANNELS = [[3, 16, 16], [16, 32, 32], [32, 64, 64, 64], [64, 128, 128, 128]]
SIAMESE_EMBEDDING_SIZE = 16
CONTRASTIVE_LOSS_MARGIN = 2.0
NUM_WORKERS = 8

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

def get_split_data(
        data_path: str = DATA_PATH,
        train_split: float = TRAIN_SPLIT, 
        test_split: float = TEST_SPLIT
    ) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    # Get pre and post file names
    maxar_intact_patches_path = f"{data_path}{MAXAR_REVIEWED_PATCHES_PATH}{MAXAR_INTACT_PATCHES_PATH}"
    maxar_damaged_patches_path = data_path + MAXAR_REVIEWED_PATCHES_PATH + MAXAR_DAMAGED_PATCHES_PATH
    intact_pre_files = [f"{maxar_intact_patches_path}{MAXAR_PRE_FOLDER}{file}" for file in os.listdir(maxar_intact_patches_path + MAXAR_PRE_FOLDER)]
    intact_post_files = [f"{maxar_intact_patches_path}{MAXAR_POST_FOLDER}{file}" for file in os.listdir(maxar_intact_patches_path + MAXAR_POST_FOLDER)]
    damaged_pre_files = [f"{maxar_damaged_patches_path}{MAXAR_PRE_FOLDER}{file}" for file in os.listdir(maxar_damaged_patches_path + MAXAR_PRE_FOLDER)]
    damaged_post_files = [f"{maxar_damaged_patches_path}{MAXAR_POST_FOLDER}{file}" for file in os.listdir(maxar_damaged_patches_path + MAXAR_POST_FOLDER)]

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

def get_dataloaders(data_path: str = DATA_PATH, batch_size: int = BATCH_SIZE):
    train_data, test_data, val_data = get_split_data(data_path=data_path)

    train_dataset = ContrastiveDataset(train_data, image_size=IMAGE_SIZE, augment_data=True)
    test_dataset = ContrastiveDataset(test_data, image_size=IMAGE_SIZE, augment_data=False)
    val_dataset = ContrastiveDataset(val_data, image_size=IMAGE_SIZE, augment_data=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    print(f'✅ Train dataloader length: {len(train_dataloader)}')
    print(f'✅ Test dataloader length: {len(test_dataloader)}')
    print(f'✅ Val dataloader length: {len(val_dataloader)}')

    return train_dataloader, test_dataloader, val_dataloader

def get_model():
    return Siamese(
        conv_channels = SIAMESE_CONV_CHANNELS,
        embedding_size = SIAMESE_EMBEDDING_SIZE,
        dropout_rate = DROPOUT_RATE
    ).to(DEVICE)

def get_criterion():
    return ContrastiveLoss(margin=CONTRASTIVE_LOSS_MARGIN)

def get_optimizer(siamese, learning_rate):
    return torch.optim.Adam(siamese.parameters(), lr=learning_rate)

def get_trainer(model, criterion):
    return Trainer(
        model = model, 
        device = DEVICE,
        criterion = criterion,
        accumulation_steps = ACCUMULATION_STEPS,
        evaluation_steps = EVALUATION_STEPS,
        print_statistics = False,
        use_scaler = True,
    )

if __name__ == '__main__':
    train_dataloader, _, val_dataloader = get_dataloaders()
    siamese = get_model()
    criterion = get_criterion()
    optimizer = get_optimizer(siamese, LEARNING_RATE)
    trainer = get_trainer(siamese, criterion)
    statistics = trainer.train_siamese(
        train_loader=train_dataloader, 
        val_loader=val_dataloader,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
