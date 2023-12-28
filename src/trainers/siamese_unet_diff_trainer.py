# This file contains the trainer class used to train a Siamese UNet with difference

from torch import nn

from src.trainers.unet_trainer import UNetTrainer

class SiameseUNetDiffTrainer(UNetTrainer):
    """
    Trainer class used to train a Siamese UNet with difference
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
            name (str, optional): The name of the trainer, defaults to ''
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