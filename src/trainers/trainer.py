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

import pandas as pd

RESULTS_FOLDER_PATH = "../../data/results/"
SAVE_STATS_PATH = f"{RESULTS_FOLDER_PATH}results.csv"


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
        name: str = "",
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
            name (str, optional): The name of the model, defaults to None

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
        self.name = name

        self.best_eval_val_loss = np.inf
        self.best_eval_val_iou = 0
        self.eval_train_loss = 0
        self.eval_val_loss = 0

    def _get_name(
        self, optimizer: Optimizer, num_epochs: int, learning_rate: int
    ) -> str:
        """
        Get the name of the model.

        Args:
            optimizer (Optimizer): The optimizer used
            num_epochs (int): The number of epochs
            learning_rate (int): The learning rate

        Returns:
            str: The name of the model
        """
        return self.name

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

    def test(
        self,
        model_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> dict:
        # Load model
        self.model.load_state_dict(torch.load(model_path))
        print("✅ Using model at ", model_path)

        # Create pandas dataframe to store results
        results = pd.DataFrame()
        results.loc[0, "name"] = self.model.__class__.__name__

        metrics = ["loss", "accuracy", "precision", "recall", "f1", "iou", "dice"]

        print("📊 Results:")
        for name, loader in [
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ]:
            stats = self._evaluate(loader, show_time=True)
            for metric in metrics:
                col_name = f"{name}_{metric}"

                # Print results
                metric_display = metric.capitalize()
                if metric == "loss":
                    print(f"\t{name} {metric_display}: {stats[metric]:.4f}")

                    # Save the results to the dataframe
                    results.loc[0, col_name] = stats[metric]
                else:
                    print(f"\t{name} {metric_display}: {stats[metric]:.4f}")

                    # Save the results to the dataframe
                    number = np.round(stats[metric] * 100, 2)
                    results.loc[0, col_name] = rf"${number}$"  # LaTeX format

        # Create the folder if it does not exist
        os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

        # Save the results to a csv file
        if not os.path.isfile(SAVE_STATS_PATH):
            results.to_csv(SAVE_STATS_PATH, index=False, sep="&")
        else:
            results.to_csv(
                SAVE_STATS_PATH, mode="a", header=False, index=False, sep="&"
            )

        return stats

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

            if (batch_idx + 1) % self.evaluation_steps == 0 or (
                batch_idx + 1 == len(train_loader)
            ):
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
                self.eval_val_iou = stats["iou"]
                if self.eval_val_iou > self.best_eval_val_iou and save_path is not None:
                    print(f"🎉 Saving model with new best IoU: {self.eval_val_iou:.4f}")
                    torch.save(self.model.state_dict(), save_path)
                    self.best_eval_val_iou = self.eval_val_iou

                # Log statistics
                statistics["val_acc"].append(stats["accuracy"])
                statistics["val_prec"].append(stats["precision"])
                statistics["val_rec"].append(stats["recall"])
                statistics["val_f1"].append(stats["f1"])
                statistics["val_iou"].append(stats["iou"])
                statistics["val_dice"].append(stats["dice"])

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
                        "val_acc": stats["accuracy"],
                        "val_prec": stats["precision"],
                        "val_rec": stats["recall"],
                        "val_f1": stats["f1"],
                        "val_iou": stats["iou"],
                        "val_dice": stats["dice"],
                    }
                )

    def _evaluate(
        self, loader: DataLoader, show_time: bool = False
    ) -> dict[str, float]:
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

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        with torch.no_grad():
            start_time = time.time()
            for batch in loader:
                val_loss, pred, target = self._forward_pass(batch)
                total_val_loss += val_loss.item()

                TP += ((pred == 1) & (target == 1)).sum().item()
                TN += ((pred == 0) & (target == 0)).sum().item()
                FP += ((pred == 1) & (target == 0)).sum().item()
                FN += ((pred == 0) & (target == 1)).sum().item()

            if show_time:
                print(f"⏲️ Time taken for evaluation: {time.time() - start_time:.4f}s")

        acc = ((TP + TN) / (TP + TN + FP + FN)) if (TP + TN + FP + FN) != 0 else 0
        prec = (TP / (TP + FP)) if (TP + FP) != 0 else 0
        rec = (TP / (TP + FN)) if (TP + FN) != 0 else 0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) != 0 else 0
        iou = (TP / (TP + FP + FN)) if (TP + FP + FN) != 0 else 0
        dice_score = (2 * TP / (2 * TP + FP + FN)) if (2 * TP + FP + FN) != 0 else 0

        total_val_loss /= len(loader)

        stats = {
            "loss": total_val_loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "iou": iou,
            "dice": dice_score,
        }

        return stats
