# This script is used to train the Siamese UNet model with difference.

import sys
from typing import List
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse

import torch
import torch.nn as nn

from src.utils.random import set_seed
from src.models.siamese import Siamese
from src.utils.parser import get_config
from src.config import SIAMESE_EMBEDDING_SIZE
from src.config import SEED, DEVICE, BATCH_SIZE
from src.models.siamese_unet_diff import SiameseUNetDiff
from src.trainers.siamese_unet_diff_trainer import SiameseUNetDiffTrainer
from src.utils.segmentation_train import get_dataloaders, get_criterion, get_optimizer

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
SIAMESE_PATH = f"{DATA_PATH}models/siamese/"
CONTRASTIVE_UNET_NAME = "contrastive_siamese_unet"


def get_model(
    encoder_channels: List[List[int]],
    decoder_channels: List[List[int]],
    dropout_rate: float,
    use_pretrained: bool,
    freeze_encoder: bool,
) -> nn.Module:
    """
    Get the model.

    Args:
        encoder_channels (List[List[int]]): Each list of integers represents the number of channels for each convolutional layer in the encoder
        decoder_channels (List[List[int]]): Each list of integers represents the number of channels for each convolutional layer in the decoder
        dropout_rate (float): The dropout rate
        use_pretrained (bool): Whether to use a pretrained model
        freeze_encoder (bool): Whether to freeze the encoder

    Returns:
        nn.Module: The model
    """
    if freeze_encoder and not use_pretrained:
        print(
            "⚠️  Warning, freezing the encoder without using a pretrained model may lead to unexpected results."
        )

    unet = SiameseUNetDiff(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        dropout_rate=dropout_rate,
    )

    if use_pretrained:
        # Load the corresponding Siamese model to get the pretrained encoder
        siamese_model = Siamese(
            conv_channels=encoder_channels,
            embedding_size=SIAMESE_EMBEDDING_SIZE,
            dropout_rate=dropout_rate,
        )

        # Load weights
        model_paths = sorted(
            [l for l in os.listdir(SIAMESE_PATH) if CONTRASTIVE_UNET_NAME in l]
        )
        if len(model_paths) == 0:
            raise ValueError("❌ No pretrained model found.")

        model_path = model_paths[-1]
        model_path = f"{SIAMESE_PATH}{model_path}"
        print(f"✅ Using model at {model_path}.")
        siamese_model.load_state_dict(torch.load(model_path))

        # Get the pretrained encoder
        encoder = siamese_model.branch.siamese_branch.encoder
        encoder.return_skipped_connections = True
        print("✅ Loaded pretrained model.")

        # Set the encoder of the UNet model to the pretrained encoder
        unet.encoder = encoder

        if freeze_encoder:
            for param in unet.encoder.parameters():
                param.requires_grad = False
            print("🥶 Encoder frozen.")

    learnable_parameters = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"🔢 Number of learnable parameters: {learnable_parameters:,}")

    unet = unet.to(DEVICE)
    return unet


def get_trainer(
    model: nn.Module,
    criterion: nn.Module,
    accumulation_steps: int,
    evaluation_steps: int,
    use_scaler: bool,
    use_pretrained: bool,
    freeze_encoder: bool,
) -> SiameseUNetDiffTrainer:
    """
    Get the trainer.

    Args:
        model (nn.Module): The model
        criterion (nn.Module): The loss function
        accumulation_steps (int): The number of accumulation steps
        evaluation_steps (int): The number of evaluation steps
        use_scaler (bool): Whether to use the scaler
        use_pretrained (bool): Whether to use a pretrained model
        freeze_encoder (bool): Whether to freeze the encoder

    Returns:
        SiameseUNetConcTrainer: The trainer
    """
    return SiameseUNetDiffTrainer(
        model=model,
        criterion=criterion,
        accumulation_steps=accumulation_steps,
        evaluation_steps=evaluation_steps,
        print_statistics=False,
        use_scaler=use_scaler,
        name=f"siamese_unet_diff{'_pretrained' if use_pretrained else ''}{'_frozen' if freeze_encoder else ''}",
    )


if __name__ == "__main__":
    set_seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="Whether to use a pretrained model",
    )
    parser.add_argument(
        "--freeze_encoder", action="store_true", help="Whether to freeze the encoder"
    )
    args = parser.parse_args()
    use_pretrained = args.use_pretrained
    freeze_encoder = args.freeze_encoder

    # Get parameters from config file
    config = get_config(f"{GLOBAL_DIR}/config/siamese_unet_diff_best_params.yml")
    print(config)
    encoder_channels = config["encoder_channels"]
    decoder_channels = config["decoder_channels"]
    dropout_rate = float(config["dropout_rate"])
    loss_name = config["loss_name"]
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config["weight_decay"])
    accumulation_steps = int(config["accumulation_steps"])
    evaluation_steps = int(config["evaluation_steps"])
    use_scaler = bool(config["use_scaler"])
    epochs = int(config["num_epochs"])

    # Get dataloaders, model, criterion, optimizer, and trainer
    train_loader, test_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    model = get_model(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        dropout_rate=dropout_rate,
        use_pretrained=use_pretrained,
        freeze_encoder=freeze_encoder,
    )
    criterion = get_criterion(criterion_name=loss_name)
    optimizer = get_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    trainer = get_trainer(
        model,
        criterion,
        accumulation_steps=accumulation_steps,
        evaluation_steps=evaluation_steps,
        use_scaler=use_scaler,
        use_pretrained=use_pretrained,
        freeze_encoder=freeze_encoder,
    )

    # Train the model
    statistics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=epochs,
        learning_rate=learning_rate,
    )
