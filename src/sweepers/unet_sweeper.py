import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import wandb
from pathlib import Path

from src.models.unet import UNet
from src.config import BATCH_SIZE
from src.sweepers.sweeper import Sweeper
from src.utils.segmentation_train import get_dataloaders
from src.utils.parser import get_config


def train(config: dict = None) -> None:
    """
    Train the model.

    Args:
        config (dict, optional): The config, defaults to None
    """
    train_dataloader, _, val_dataloader = get_dataloaders(
        batch_size=BATCH_SIZE
    )
    sweeper = Sweeper(model_class=UNet, config=config)
    sweeper.train(train_loader=train_dataloader, val_loader=val_dataloader)


if __name__ == "__main__":
    config_path = f"{GLOBAL_DIR}/config/unet_sweep_config.yml"
    config = get_config(config_path)

    sweep_id = wandb.sweep(config, project="UNet-Sweep")
    wandb.agent(sweep_id, function=train)
