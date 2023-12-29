# This file contains the general implementation of a trainer for the UNet model.

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.optim import Optimizer

from src.trainers.trainer import Trainer

from src.config import DEVICE


class UNetTrainer(Trainer):
    """
    Trainer class used to train a UNet model.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        accumulation_steps: int,
        evaluation_steps: int,
        print_statistics: bool = False,
        use_scaler: bool = False,
        name: str = '',
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model (nn.Module): The model to train
            criterion (nn.Module): The loss function
            accumulation_steps (int): The number of steps to accumulate gradients
            evaluation_steps (int): The number of steps to evaluate model
            print_statistics (bool, optional): Whether to print statistics, defaults to False
            use_scaler (bool, optional): Whether to use scaler, defaults to False
        """
        super().__init__(
            model=model,
            criterion=criterion,
            accumulation_steps=accumulation_steps,
            evaluation_steps=evaluation_steps,
            print_statistics=print_statistics,
            use_scaler=use_scaler,
            name=name,
        )

    def _get_name(
        self, optimizer: Optimizer, num_epochs: int, learning_rate: float
    ) -> str:
        """
        Get name of model.

        Args:
            optimizer (torch.optim): The optimizer used
            num_epochs (int): The number of epochs
            learning_rate (float): The learning rate

        Returns:
            str: The name of the model
        """
        name = self.name
        name += self.model.__class__.__name__
        name += f"_{optimizer.__class__.__name__}optim"
        name += f"_{num_epochs}epochs"
        name += f"_{str(learning_rate).replace('.', '')}lr"
        name += f"_{self.criterion.__class__.__name__}loss"
        name += f"_{self.model.activation.__class__.__name__}act"
        name += f"_{str(self.model.dropout_rate).replace('.', '')}dropout"
        name += f"_{self.accumulation_steps}accstep"
        name += f"_{self.evaluation_steps}evalstep"
        encoder_channels_str = sum(len(channels) for channels in self.model.encoder_channels)
        name += f"_{encoder_channels_str}encchan"
        decoder_channels_str = sum(len(channels) for channels in self.model.decoder_channels)
        name += f"_{decoder_channels_str}decchan"

        return name

    def _forward_pass(self, batch: tuple) -> torch.Tensor:
        """
        Forward pass of the UNet model.

        Args:
            batch (tuple): batch of data

        Returns:
            torch.Tensor: training loss value
        """
        # Unpack batch
        pre, post, mask, _ = batch
        pre = pre.float().to(DEVICE)
        post = post.float().to(DEVICE)
        mask = mask.float().to(DEVICE)

        # Forward pass
        with autocast(enabled=self.use_scaler):
            outputs = self.model(pre, post).squeeze(1)

            pred = (torch.sigmoid(outputs) > 0.5).int()
            targets = mask.int()

        # Compute loss
        return self.criterion(outputs, mask), pred, targets
