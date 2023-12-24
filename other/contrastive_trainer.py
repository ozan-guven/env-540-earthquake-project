import torch
from torch import nn
from torch.cuda.amp import autocast

from trainer.trainer import Trainer

class ContrastiveTrainer(Trainer):
    """
    Trainer for the Contrastive model.
    """

    def __init__(
        self, 
        model: nn.Module, 
        device: str, 
        criterion: nn.Module, 
        accumulation_steps: int, 
        evaluation_steps: int,
        print_statistics: bool = False,
        use_scaler: bool = False,
    ):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): model to train
            device (str): device to use for training
            criterion (nn.Module): loss function to use
            accumulation_steps (int, optional): accumulation steps for gradient accumulation.
            evaluation_steps (int, optional): evaluation steps for evaluation during training.
            print_statistics (bool, optional): whether to print statistics during training. Defaults to False.
            use_scaler (bool, optional): whether to use scaler. Defaults to False.

        Throws:
            ValueError: if accumulation_steps is not a positive integer
        """
        super().__init__(
            model=model,
            device=device,
            criterion=criterion,
            accumulation_steps=accumulation_steps,
            evaluation_steps=evaluation_steps,
            print_statistics=print_statistics,
            use_scaler=use_scaler,
        )
            
            
    def _forward_pass(self, batch: tuple) -> torch.Tensor:
        """
        Forward pass of the Contrastive model.

        Args:
            batch (tuple): batch of data

        Returns:
            torch.Tensor: training loss value
        """
        # Unpack batch
        pre, post, label = batch
        pre.to(self.device)
        post.to(self.device)
        label.to(self.device)
        
        # Forward pass
        with autocast(enabled=self.use_scaler):
            pre_embeddings, post_embeddings = self.model(pre, post)

        # Compute loss
        return self.criterion(pre_embeddings, post_embeddings, label)