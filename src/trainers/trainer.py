### OK
# This file contains the general implementation of a trainer.

import os
import time
import wandb
import numpy as np
from tqdm import tqdm
from typing import Tuple
from abc import ABC, abstractmethod


import torch
from torch import nn
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_precision,
    binary_recall,
    binary_f1_score,
    binary_jaccard_index,
    dice,
)


class EarlyStopper:
    """
    Early stopper to stop the training if the validation loss does not improve for a certain number of epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        """
        Initialize the early stopper.

        Args:
            patience (int, optional): The number of epochs to wait before stopping the training, defaults to 1
            min_delta (float, optional): The minimum difference between the current validation loss and the previous best validation loss, defaults to 0.0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        """
        Check if the training should be stopped.

        Args:
            validation_loss (float): The current validation loss

        Returns:
            bool: Whether the training should be stopped
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer(ABC):
    """
    Trainer class used to train a model.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        accumulation_steps: int,
        evaluation_steps: int,
        print_statistics: bool = False,
        use_scaler: bool = False,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model (nn.Module): The model to train
            criterion (nn.Module): The loss function to use
            accumulation_steps (int, optional): The accumulation steps for gradient accumulation
            evaluation_steps (int, optional): The evaluation steps for evaluation
            print_statistics (bool, optional): Whether to print statistics during training, defaults to False
            use_scaler (bool, optional): Whether to use scaler, defaults to False

        Throws:
            ValueError: If accumulation_steps is not a positive integer
        """
        if accumulation_steps <= 0:
            raise ValueError("❌ Accumulation steps must be a positive integer")

        self.model = model
        self.criterion = criterion
        self.accumulation_steps = accumulation_steps
        self.evaluation_steps = evaluation_steps
        self.print_statistics = print_statistics
        self.use_scaler = use_scaler

        self.best_eval_val_loss = np.inf
        self.best_eval_val_iou = 0
        self.eval_train_loss = 0
        self.eval_val_loss = 0

    @abstractmethod
    def _get_name(
        self, optimizer: Optimizer, num_epochs: int, learning_rate: int
    ) -> str:
        """
        Get the name of the model. This should be implemented by the child class.

        Args:
            optimizer (Optimizer): The optimizer used
            num_epochs (int): The number of epochs
            learning_rate (int): The learning rate

        Raises:
            NotImplementedError: if not implemented by child class

        Returns:
            str: The name of the model
        """
        raise NotImplementedError

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        num_epochs: int,
        learning_rate: int = 0,
        save_model: bool = True,
        sweeping: bool = False,
    ) -> dict:
        """
        Train the UNet model.

        Args:
            train_loader (DataLoader): The training data loader
            val_loader (DataLoader): The validation data loader
            optimizer (Optimizer): The optimizer to use during training
            num_epochs (int): The number of epochs to train
            learning_rate (int, optional): The learning rate to use, defaults to 0
            save_model (bool, optional): Whether to save the best model, defaults to True
            sweeping (bool, optional): Whether the training is part of a sweeping, defaults to False

        Throws:
            ValueError: If num_epochs is not a positive integer.

        Returns:
            dict: Statistics of the training.
        """
        if num_epochs <= 0:
            raise ValueError("❌ num_epochs must be a positive integer")

        name = self._get_name(optimizer, num_epochs, learning_rate)

        # Get a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{timestamp}_{name}"

        # Create a folder in data/models folder with the name of the model
        model_dir = f"../../data/models/{self.model.__class__.__name__.lower()}/"
        os.makedirs(model_dir, exist_ok=True)

        save_path = f"{model_dir}{name}.pth" if save_model else None

        # Setup WandB and watch
        if not sweeping:
            wandb.init(
                project=self.__class__.__name__.lower(),
                config={
                    "architecture": self.__class__.__name__,
                    "name": name,
                    "dataset": "Maxar 2023 Turkish Earthquake",
                    "epochs": num_epochs,
                    "learning_rate": learning_rate,
                },
            )
        wandb.watch(self.model, log_freq=4, log="all")

        print(f"🚀 Training {self.__class__.__name__} method for {num_epochs} epochs...")

        # Recording statistics for each epoch
        statistics = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_prec": [],
            "val_rec": [],
            "val_f1": [],
            "val_iou": [],
            "val_dice": [],
            "val_acc_std": [],
            "val_prec_std": [],
            "val_rec_std": [],
            "val_f1_std": [],
            "val_iou_std": [],
            "val_dice_std": [],
            "train_loader_length": len(train_loader),
            "val_loader_length": len(val_loader),
            "num_epochs": num_epochs,
            "evaluation_steps": self.evaluation_steps,
        }

        # Scaler
        scaler = GradScaler(enabled=self.use_scaler)

        # Training loop
        with tqdm(range(num_epochs), desc="Epochs", unit="epoch") as bar:
            for _ in bar:
                self._train_one_epoch(
                    train_loader,
                    val_loader,
                    optimizer,
                    statistics,
                    scaler,
                    save_path=save_path,
                    bar=bar,
                )

        wandb.unwatch(self.model)
        if not sweeping:
            wandb.finish()

        return statistics

    @abstractmethod
    def _forward_pass(
        self, batch: tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of a batch. This should be implemented by the child class.

        Args:
            batch (tuple): The batch of data

        Raises:
            NotImplementedError: If not implemented by child class

        Returns:
            train_loss (torch.Tensor): The loss of the batch
            pred (torch.Tensor): The prediction of the batch
            target (torch.Tensor): The target of the batch
        """
        raise NotImplementedError

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
            train_loss, _, _ = self._forward_pass(batch)
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
                
                # Get validation iou and update best model
                self.eval_val_iou = stats["iou"].mean()
                if self.eval_val_iou > self.best_eval_val_iou and save_path is not None:
                    print(f"🎉 Saving model with new best IoU: {self.eval_val_iou:.4f}")
                    torch.save(self.model.state_dict(), save_path)
                    self.best_eval_val_iou = self.eval_val_iou

                # Log statistics
                statistics["val_acc"].append(stats["accuracy"].mean())
                statistics["val_prec"].append(stats["precision"].mean())
                statistics["val_rec"].append(stats["recall"].mean())
                statistics["val_f1"].append(stats["f1"].mean())
                statistics["val_iou"].append(stats["iou"].mean())
                statistics["val_dice"].append(stats["dice"].mean())

                statistics["val_acc_std"].append(stats["accuracy"].std())
                statistics["val_prec_std"].append(stats["precision"].std())
                statistics["val_rec_std"].append(stats["recall"].std())
                statistics["val_f1_std"].append(stats["f1"].std())
                statistics["val_iou_std"].append(stats["iou"].std())
                statistics["val_dice_std"].append(stats["dice"].std())

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
                        "val_acc": stats["accuracy"].mean(),
                        "val_prec": stats["precision"].mean(),
                        "val_rec": stats["recall"].mean(),
                        "val_f1": stats["f1"].mean(),
                        "val_iou": stats["iou"].mean(),
                        "val_dice": stats["dice"].mean(),
                        "val_acc_std": stats["accuracy"].std(),
                        "val_prec_std": stats["precision"].std(),
                        "val_rec_std": stats["recall"].std(),
                        "val_f1_std": stats["f1"].std(),
                        "val_iou_std": stats["iou"].std(),
                        "val_dice_std": stats["dice"].std(),
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
                - the accuracy
                - the precision
                - the recall
                - the f1 score
                - the iou
                - the dice score
        """
        self.model.eval()

        total_val_loss = 0
        accs, precs, recs, f1s, ious, dice_scores = [], [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                val_loss, pred, target = self._forward_pass(batch)
                total_val_loss += val_loss.item()

                # Compute all metrics using torchmetrics
                accs.append(binary_accuracy(pred, target).cpu().item())
                precs.append(binary_precision(pred, target).cpu().item())
                recs.append(binary_recall(pred, target).cpu().item())
                f1s.append(binary_f1_score(pred, target).cpu().item())
                iou = binary_jaccard_index(pred, target).cpu().item()
                if not np.isnan(iou):
                    ious.append(iou)
                dice_scores.append(dice(pred, target).cpu().item())

        total_val_loss /= len(loader)

        stats = {
            "loss": total_val_loss,
            "accuracy": np.asarray(accs),
            "precision": np.asarray(precs),
            "recall": np.asarray(recs),
            "f1": np.asarray(f1s),
            "iou": np.asarray(ious),
            "dice": np.asarray(dice_scores),
        }

        return stats
