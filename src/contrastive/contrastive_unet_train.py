import sys
from typing import List
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch.nn as nn

from src.utils.random import set_seed
from src.models.siamese import Siamese
from src.utils.parser import get_config
from src.unet.unet_train import get_dataloaders
from src.config import SEED, DEVICE, BATCH_SIZE
from src.utils.segmentation_train import get_optimizer
from src.losses.contrastive_loss import ContrastiveLoss
from src.data.contrastive_dataset import ContrastiveDataset
from src.trainers.contrastive_trainer import ContrastiveTrainer

TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT = 0.92, 0.04, 0.04


def get_model(
        encoder_channels: List[List[int]],
        embedding_size: int,
        dropout_rate: float
    ) -> nn.Module:
    """
    Get the model.

    Args:
        encoder_channels (List[List[int]]): Each list of integers represents the number of channels for each convolutional layer in the encoder
        embedding_size (int): The embedding size
        dropout_rate (float): The dropout rate

    Returns:
        nn.Module: The model
    """
    return Siamese(
        conv_channels = encoder_channels,
        embedding_size = embedding_size,
        dropout_rate = dropout_rate
    ).to(DEVICE)


def get_criterion(margin: float = 1.0):
    """
    Get the criterion.

    Args:
        margin (float, optional): The margin, defaults to 1.0

    Returns:
        ContrastiveLoss: The criterion
    """
    return ContrastiveLoss(margin=margin)

def get_trainer(
        model: nn.Module, 
        criterion: nn.Module, 
        accumulation_steps: int,
        evaluation_steps: int,
        use_scaler: bool,
    ):
    """
    Get the trainer.

    Args:
        model (nn.Module): The model
        criterion (nn.Module): The loss function
        accumulation_steps (int): The number of accumulation steps
        evaluation_steps (int): The number of evaluation steps
        use_scaler (bool): Whether to use the scaler

    Returns:
        ContrastiveTrainer: The trainer
    """
    return ContrastiveTrainer(
        model=model,
        criterion=criterion,
        accumulation_steps=accumulation_steps,
        evaluation_steps=evaluation_steps,
        print_statistics=False,
        use_scaler=use_scaler,
        name='unet'
    )

if __name__ == '__main__':
    set_seed(SEED)

    # Get parameters from config file
    config = get_config(f"{GLOBAL_DIR}/config/contrastive_unet_best_params.yml")
    encoder_channels = config["encoder_channels"]
    embedding_size = int(config["embedding_size"])
    dropout_rate = float(config["dropout_rate"])
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    accumulation_steps = int(config["accumulation_steps"])
    evaluation_steps = int(config["evaluation_steps"])
    use_scaler = bool(config["use_scaler"])
    epochs = int(config["num_epochs"])
    margin = float(config["contrastive_margin"])

    # Get dataloaders, model, criterion, optimizer, and trainer
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        use_intact=True,    # To get half intact and half damaged data
        with_val=False,
        dataset_class=ContrastiveDataset,
        train_split=TRAIN_SPLIT,
        test_split=TEST_SPLIT,
        val_split=VAL_SPLIT,
    )
    model = get_model(
        encoder_channels=encoder_channels,
        embedding_size=embedding_size,
        dropout_rate=dropout_rate,
    )

    criterion = get_criterion(margin=margin)
    optimizer = get_optimizer(
        model, 
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    trainer = get_trainer(
        model, 
        criterion=criterion,
        accumulation_steps=accumulation_steps,
        evaluation_steps=evaluation_steps,
        use_scaler=use_scaler,
    )
    
    # Train the model
    statistics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=epochs,
        learning_rate=learning_rate,
    )
    
