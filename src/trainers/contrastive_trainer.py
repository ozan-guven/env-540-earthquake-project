import wandb
from tqdm import tqdm

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.config import DEVICE
from src.trainers.trainer import Trainer


class ContrastiveTrainer(Trainer):
    """
    Trainer for the Contrastive model.
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
    ):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): model to train
            criterion (nn.Module): loss function to use
            accumulation_steps (int, optional): accumulation steps for gradient accumulation.
            evaluation_steps (int, optional): evaluation steps for evaluation during training.
            print_statistics (bool, optional): whether to print statistics during training. Defaults to False.
            use_scaler (bool, optional): whether to use scaler. Defaults to False.
            name (str, optional): name of the model. Defaults to None.
        Throws:
            ValueError: if accumulation_steps is not a positive integer
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
            
    def _get_name(self, optimizer: Optimizer, num_epochs: int, learning_rate: int) -> str:
        """
        Get name of model.

        Args:
            optimizer (torch.optim): The optimizer used
            num_epochs (int): The number of epochs
            learning_rate (float): The learning rate

        Returns:
            str: The name of the model
        """
        name = "contrastive"
        name += f"_{self.name}" if self.name is not None else ""
        return name
        
     
    def _forward_pass(self, batch: tuple) -> torch.Tensor:
        """
        Forward pass of the Contrastive model.

        Args:
            batch (tuple): batch of data

        Returns:
            torch.Tensor: training loss value
        """
        # Unpack batch
        x, y, _, label = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        label = label.to(DEVICE)
        
        # Forward pass
        with autocast(enabled=self.use_scaler):
            pre_embeddings, post_embeddings = self.model(x, y)

        # Compute loss
        return self.criterion(pre_embeddings, post_embeddings, label)
    
    
    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        statistics: dict,
        scaler: GradScaler,
        save_path: str = None,
        bar: tqdm = None,
    ) -> None:
        """
        Train the model for one epoch.

        Args:
            train_loader (DataLoader): The training data loader
            val_loader (DataLoader): The validation data loader
            optimizer (Optimizer): The optimizer to use during training
            statistics (dict): The statistics of the training
            scaler (GradScaler): The scaler to use
            save_path (str, optional): The path to save the best model, defaults to None
            bar (tqdm, optional): The progress bar to use, efaults to None
        """
        self.model.train()

        total_train_loss = 0
        n_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()
            train_loss = self._forward_pass(batch)
            total_train_loss += train_loss.item()
            n_train_loss += 1

            scaler.scale(train_loss).backward()

            # Optimize every accumulation steps
            if ((batch_idx + 1) % self.accumulation_steps == 0) or (
                batch_idx + 1 == len(train_loader)
            ):
                scaler.step(optimizer)
                scaler.update()

            if bar is not None:
                bar.set_postfix(
                    {
                        "batch": f"{batch_idx + 1}/{len(train_loader)}",
                        "train_loss": f"{self.eval_train_loss:.4f}",
                        "val_loss": f"{self.eval_val_loss:.4f}",
                    }
                )

            if (batch_idx + 1) % self.evaluation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                # Get and update training loss
                self.eval_train_loss = total_train_loss / n_train_loss
                statistics["train_loss"].append(train_loss)
                total_train_loss = 0
                n_train_loss = 0

                # Get validation loss ans statistics
                stats = self._evaluate(val_loader)
                self.eval_val_loss = stats["loss"]
                statistics["val_loss"].append(self.eval_val_loss)
                
                if self.eval_val_loss < self.best_eval_val_loss and save_path is not None:
                    print(f"🎉 Saving model with new best loss: {self.eval_val_loss:.4f}")
                    torch.save(self.model.state_dict(), save_path)
                    self.best_eval_val_loss = self.eval_val_loss

                if bar is None and self.print_statistics:
                    print(
                        f"➡️ Training loss: {self.eval_train_loss:.4f}, Validation loss: {self.eval_val_loss:.4f}"
                    )
                else:
                    bar.set_postfix(
                        {
                            "batch": f"{batch_idx + 1}/{len(train_loader)}",
                            "train_loss": f"{self.eval_train_loss:.4f}",
                            "val_loss": f"{self.eval_val_loss:.4f}",
                        }
                    )
                wandb.log(
                    {
                        "train_loss": self.eval_train_loss,
                        "val_loss": self.eval_val_loss,
                    }
                )

    def _evaluate(self, loader: DataLoader) -> dict[str, float]:
        """
        Evaluate the model on the given loader.

        Args:
            loader (DataLoader): The loader to evaluate on

        Returns:
            dict[str, float]: The evaluation statistics. The evaluation statistics contain:
                - the validation loss
        """
        self.model.eval()

        total_val_loss = 0
        with torch.no_grad():
            for batch in loader:
                val_loss = self._forward_pass(batch)
                total_val_loss += val_loss.item()

        total_val_loss /= len(loader)

        stats = {
            "loss": total_val_loss,
        }

        return stats