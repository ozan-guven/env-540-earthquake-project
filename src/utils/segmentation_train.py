# This file contains utility functions for training models

import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import random
import zipfile
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import SegmentationDataset
from src.config import DEVICE, SEED, IMAGE_SIZE, POS_WEIGHT
from src.utils.random import set_seed

from src.losses.iou_loss import IoULoss
from src.losses.dice_loss import DiceLoss

DATA_PATH = str(GLOBAL_DIR / "data") + "/"

NUM_WORKERS = 12
TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT = 0.7, 0.15, 0.15


def _unzip_archive(path: str, remove_zip: bool = False) -> None:
    """
    Unzip a zip archive, given that the archive exists and is a zip file

    Args:
        path (str): Path to the archive to be unzipped
        remove_zip (bool, optional): Whether to remove the zip archive after unzipping it, defaults to False
    """
    if not (os.path.exists(path) and os.path.isfile(path) and path.endswith(".zip")):
        return

    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(path))
        print(f"✅ Archive {path} unzipped.")
    if remove_zip:
        print(f"✅ Archive {path} removed.")
        os.remove(path)


def get_data_tuples(
    folder_path: str, n_elements: int = -1
) -> List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]]:
    """
    Get a list of dictionaries, where each dictionary contains the paths to the pre and post images and their corresponding masks.

    Args:
        folder_path (str): Path to the folder containing the data
        n_elements (int, optional): Number of elements to get, defaults to -1

    Returns:
        List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]]: List of dictionaries, where each dictionary contains the paths to the pre and post images and their corresponding masks
    """
    # Unzip data if needed
    _unzip_archive(folder_path)
    folder_path = (
        folder_path.split(".")[0] if folder_path.endswith(".zip") else folder_path
    )

    # Check that in each intact and dameged folder, there are three folders: pre, post and mask
    for subfolder in ["pre", "post", "mask"]:
        assert os.path.exists(
            os.path.join(folder_path, subfolder)
        ), f"❌ Folder {os.path.join(folder_path, subfolder)} does not exist."

    # Get paths to images
    post_paths = os.listdir(os.path.join(folder_path, "post"))
    post_paths = [os.path.join(folder_path, "post", path) for path in post_paths]
    pre_paths = os.listdir(os.path.join(folder_path, "pre"))
    pre_paths = [os.path.join(folder_path, "pre", path) for path in pre_paths]

    # Create dictionary
    data_dict = {}
    for paths, key in [(post_paths, "post_mask"), (pre_paths, "pre")]:
        for path in paths:
            # Get id of the image, for example, from .../031131233232_2022-07-26_10300100D797E100-visual_254.png to 031131233232_visual_254
            id = path.split("/")[-1]
            id = id.split("_")[0] + "_" + id.split("-")[-1].split(".")[0]
            if id not in data_dict:
                data_dict[id] = {
                    "pre": [],  # List of paths to pre images
                    "post_mask": [],  # List of tuples of paths to post images and their corresponding masks, if it exists
                }
            if key == "pre":
                data_dict[id][key].append(path)
            else:
                post_path = path
                mask_path = path.replace("post", "mask")
                mask_path = mask_path if os.path.exists(mask_path) else None
                data_dict[id][key].append((post_path, mask_path))

    # Get the keys of the dictionary as a list
    values = list(data_dict.values())
    if 0 < n_elements and n_elements < len(values):
        values = random.sample(values, n_elements)

    return values


def get_split_data(use_intact: bool = False) -> (
    Tuple[
        List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]],
        List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]],
        List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]],
    ]
):
    """
    Get the data split into train, test and val.
    
    Args:
        use_intact (bool, optional): Whether to use intact images, defaults to False
    """
    if np.abs(TRAIN_SPLIT + TEST_SPLIT + VAL_SPLIT - 1) > 1e-6:
        raise ValueError("❌ The sum of the splits must be equal to 1.")
    """
    Get the data split into train, test and val.

    Returns:
        Tuple[
            List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]],
            List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]],
            List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]]
        ]: Train, test and val data
    """
    damaged = get_data_tuples(f"{DATA_PATH}/damaged")
    random.shuffle(damaged)
    intact = get_data_tuples(f"{DATA_PATH}/intact", n_elements=len(damaged))
    random.shuffle(intact)

    # Split data into train, test and val, keeping the same proportion of damaged and intact images
    train_data = (
        damaged[: int(TRAIN_SPLIT * len(damaged))]
        + (intact[: int(TRAIN_SPLIT * len(intact))] if use_intact else [])
    )
    test_data = (
        damaged[
            int(TRAIN_SPLIT * len(damaged)) : int(
                (TRAIN_SPLIT + TEST_SPLIT) * len(damaged)
            )
        ]
        + (intact[
            int(TRAIN_SPLIT * len(intact)) : int(
                (TRAIN_SPLIT + TEST_SPLIT) * len(intact)
            )
        ] if use_intact else [])
    )
    val_data = (
        damaged[int((TRAIN_SPLIT + TEST_SPLIT) * len(damaged)) :]
        + (intact[int((TRAIN_SPLIT + TEST_SPLIT) * len(intact)) :] if use_intact else [])
    )

    return train_data, test_data, val_data


def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get the train, test and val dataloaders.

    Args:
        batch_size (int): The batch size

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, test and val dataloaders
    """
    set_seed(SEED)
    train_data, test_data, val_data = get_split_data()

    train_dataset = SegmentationDataset(
        train_data, image_size=IMAGE_SIZE, augment_data=True
    )
    test_dataset = SegmentationDataset(
        test_data, image_size=IMAGE_SIZE, augment_data=False
    )
    val_dataset = SegmentationDataset(
        val_data, image_size=IMAGE_SIZE, augment_data=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    print(f"✅ Train dataloader length: {len(train_dataloader)}")
    print(f"✅ Test dataloader length: {len(test_dataloader)}")
    print(f"✅ Val dataloader length: {len(val_dataloader)}")

    return train_dataloader, test_dataloader, val_dataloader


def get_criterion(criterion_name: str = "bce") -> nn.Module:
    """
    Get the loss function.

    Args:
        criterion_name (str): The name of the loss function, defaults to "bce"

    Returns:
        nn.Module: The loss function
    """
    print(f"✅ Using {criterion_name} criterion.")
    match criterion_name:
        case "bce":
            return nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT).to(DEVICE)
        case "dice":
            return DiceLoss().to(DEVICE)
        case "iou":
            return IoULoss().to(DEVICE)


def get_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim:
    """
    Get the optimizer.

    Args:
        model (nn.Module): The model
        learning_rate (float): The learning rate
        weight_decay (float): The weight decay

    Returns:
        torch.optim: The optimizer
    """
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
