### OK
# This script is used to train the UNet model.

import sys
from typing import List
from pathlib import Path

import pandas as pd

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch.nn as nn

from src.models.unet import UNet
from src.config import SEED, DEVICE, BATCH_SIZE
from src.utils.random import set_seed
from src.trainers.unet_trainer import UNetTrainer
from src.utils.segmentation_train import get_dataloaders, get_criterion
from src.utils.parser import get_config

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
MODELS_PATH = f'{DATA_PATH}models/'


def get_model(
        encoder_channels: List[List[int]],
        decoder_channels: List[List[int]],
        dropout_rate: float
        ) -> nn.Module:
    """
    Get the model.

    Args:
        encoder_channels (List[List[int]]): Each list of integers represents the number of channels for each convolutional layer in the encoder
        decoder_channels (List[List[int]]): Each list of integers represents the number of channels for each convolutional layer in the decoder
        dropout_rate (float): The dropout rate

    Returns:
        nn.Module: The model
    """
    return UNet(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        dropout_rate=dropout_rate
    ).to(DEVICE)


def get_trainer(
        model: nn.Module, 
        criterion: nn.Module, 
        accumulation_steps: int,
        evaluation_steps: int,
        use_scaler: bool,
        ) -> UNetTrainer:
    """
    Get the trainer.

    Args:
        model (nn.Module): The model
        criterion (nn.Module): The loss function
        accumulation_steps (int): The number of accumulation steps
        evaluation_steps (int): The number of evaluation steps
        use_scaler (bool): Whether to use the scaler

    Returns:
        UNetTrainer: The trainer
    """
    return UNetTrainer(
        model=model,
        criterion=criterion,
        accumulation_steps=accumulation_steps,
        evaluation_steps=evaluation_steps,
        print_statistics=False,
        use_scaler=use_scaler,
    )


if __name__ == "__main__":
    set_seed(SEED)

    # Get parameters from config file
    config = get_config(f"{GLOBAL_DIR}/config/unet_best_params.yml")
    encoder_channels = config["encoder_channels"]
    decoder_channels = config["decoder_channels"]
    dropout_rate = float(config["dropout_rate"])
    loss_name = config["loss_name"]
    accumulation_steps = int(config["accumulation_steps"])
    evaluation_steps = int(config["evaluation_steps"])
    use_scaler = bool(config["use_scaler"])

    # Get dataloaders, model, criterion and trainer
    train_loader, test_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    model = get_model(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        dropout_rate=dropout_rate,
    )
    criterion = get_criterion(criterion_name=loss_name)
    trainer = get_trainer(
        model, 
        criterion,
        accumulation_steps=accumulation_steps,
        evaluation_steps=evaluation_steps,
        use_scaler=use_scaler,
    )

    # Test the model
    model_save_dict = f'{MODELS_PATH}unet/'
    model_save_path = sorted(os.listdir(model_save_dict))[-1]
    statistics = trainer.test(
        model_path=f'{model_save_dict}{model_save_path}',
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )