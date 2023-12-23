### OK
# This script is used to train the UNet model.

import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import random
import zipfile
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.unet.unet_dataset import ImageDataset
from src.utils.unet_trainer import UNetTrainer
from src.models.unet import UNet
from src.utils.random import set_seed
from src.config import SEED, DEVICE, IMAGE_SIZE

from typing import List, Dict, Union, Tuple, Optional

DATA_PATH = str(GLOBAL_DIR / "data") + "/"

EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
EVALUATION_STEPS = 100
DROPOUT_RATE = 0.0
NUM_WORKERS = 12
POS_WEIGHT = torch.tensor([34.8282]).reshape(1, 1, 1)

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


def get_split_data() -> (
    Tuple[
        List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]],
        List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]],
        List[Dict[str, List[Union[str, Tuple[str, Optional[str]]]]]],
    ]
):
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
        + intact[: int(TRAIN_SPLIT * len(intact))]
    )
    test_data = (
        damaged[
            int(TRAIN_SPLIT * len(damaged)) : int(
                (TRAIN_SPLIT + TEST_SPLIT) * len(damaged)
            )
        ]
        + intact[
            int(TRAIN_SPLIT * len(intact)) : int(
                (TRAIN_SPLIT + TEST_SPLIT) * len(intact)
            )
        ]
    )
    val_data = (
        damaged[int((TRAIN_SPLIT + TEST_SPLIT) * len(damaged)) :]
        + intact[int((TRAIN_SPLIT + TEST_SPLIT) * len(intact)) :]
    )

    return train_data, test_data, val_data


def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get the train, test and val dataloaders.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, test and val dataloaders
    """
    train_data, test_data, val_data = get_split_data()

    train_dataset = ImageDataset(train_data, image_size=IMAGE_SIZE, augment_data=True)
    test_dataset = ImageDataset(test_data, image_size=IMAGE_SIZE, augment_data=False)
    val_dataset = ImageDataset(val_data, image_size=IMAGE_SIZE, augment_data=False)

    train_dataloader = DataLoader(
        SEED,
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )

    print(f"✅ Train dataloader length: {len(train_dataloader)}")
    print(f"✅ Test dataloader length: {len(test_dataloader)}")
    print(f"✅ Val dataloader length: {len(val_dataloader)}")

    return train_dataloader, test_dataloader, val_dataloader


def get_model() -> nn.Module:
    """
    Get the model.

    Returns:
        nn.Module: The model
    """
    return UNet(dropout_rate=DROPOUT_RATE).to(DEVICE)


def get_criterion() -> nn.Module:
    """
    Get the loss function.

    Returns:
        nn.Module: The loss function
    """
    return nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT).to(DEVICE)


def get_optimizer(model: nn.Module, learning_rate: float) -> torch.optim:
    """
    Get the optimizer.

    Args:
        model (nn.Module): The model
        learning_rate (float): The learning rate

    Returns:
        torch.optim: The optimizer
    """
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)


def get_trainer(model: nn.Module, criterion: nn.Module) -> UNetTrainer:
    """
    Get the trainer.

    Args:
        model (nn.Module): The model
        criterion (nn.Module): The loss function

    Returns:
        UNetTrainer: The trainer
    """
    return UNetTrainer(
        model=model,
        criterion=criterion,
        accumulation_steps=ACCUMULATION_STEPS,
        evaluation_steps=EVALUATION_STEPS,
        print_statistics=False,
        use_scaler=True,
    )


if __name__ == "__main__":
    set_seed(SEED)

    if np.abs(TRAIN_SPLIT + TEST_SPLIT + VAL_SPLIT - 1) > 1e-6:
        raise ValueError("❌ The sum of the splits must be equal to 1.")

    train_loader, test_loader, val_loader = get_dataloaders()
    model = get_model()
    criterion = get_criterion()
    optimizer = get_optimizer(model, learning_rate=LEARNING_RATE)
    trainer = get_trainer(model, criterion)
    statistics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
    )
