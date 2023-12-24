# Sweeper class for sweeping the parameter space of a model

import wandb
import numpy as np

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.losses.iou_loss import IoULoss
from src.losses.dice_loss import DiceLoss
from src.config import DEVICE, POS_WEIGHT
from src.trainers.unet_trainer import UNetTrainer
from src.trainers.siamese_unet_conc_trainer import SiameseUNetConcTrainer
from src.trainers.siamese_unet_diff_trainer import SiameseUNetDiffTrainer

from src.models.unet import UNet
from src.models.siamese_unet_conc import SiameseUNetConc
from src.models.siamese_unet_diff import SiameseUNetDiff


class Sweeper:
    """
    Sweeper class for sweeping the parameter space of a model.
    """
    def __init__(
        self,
        model_class: str,
        config: dict,
    ):
        """
        Initialize the Sweeper.
        
        Args:
            model_class (str): The model class, for example "UNet"
            config (dict): The config
        
        """
        self.config = config
        self.model_class = model_class

        self.best_eval_val_loss = np.inf

    def get_loss(self, config: dict) -> nn.Module:
        """
        Get the loss function.
        
        Args:
            config (dict): The config
            
        Returns:
            nn.Module: The loss function
        """
        match config.loss_name:
            case "dice":
                return DiceLoss().to(DEVICE)
            case "iou":
                return IoULoss().to(DEVICE)
            case _:
                return nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT).to(DEVICE)
            
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): The train loader
            val_loader (DataLoader): The validation loader
        """
        with wandb.init(config=self.config):
            config = wandb.config
            self.num_epochs = config.num_epochs
            self.evaluation_steps = config.evaluation_steps
            self.accumulation_steps = config.accumulation_steps
            self.use_scaler = config.use_scaler

            channels = config.channels
            self.encoder_channels, self.decoder_channels = channels[0], channels[1]
            self.model = self.model_class(
                encoder_channels=self.encoder_channels,
                decoder_channels=self.decoder_channels,
                dropout_rate=config.dropout_rate,
            ).to(DEVICE)

            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            # Get criterion
            self.criterion = self.get_loss(config)

            # Define trainer
            if self.model_class == UNet:
                self.trainer = UNetTrainer
            elif self.model_class == SiameseUNetConc:
                self.trainer = SiameseUNetConcTrainer
            elif self.model_class == SiameseUNetDiff:
                self.trainer = SiameseUNetDiffTrainer
            else:
                raise ValueError(f"❌Model {self.model_class} not supported.")

            self.trainer = self.trainer(
                model=self.model,
                criterion=self.criterion,
                accumulation_steps=self.accumulation_steps,
                evaluation_steps=self.evaluation_steps,
                use_scaler=self.use_scaler,
            )

            self.trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=self.optimizer,
                num_epochs=self.num_epochs,
                save_model=False,
                sweeping=True,
            )
