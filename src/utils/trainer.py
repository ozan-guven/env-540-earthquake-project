import numpy as np
import torch
import wandb
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer():
    """
    Trainer class used to train a model.
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
        """Initialize the trainer.

        Args:
            model (nn.Module): model to train
            device (str): device to use for training
            criterion (nn.Module): loss function to use
            accumulation_steps (int, optional): accumulation steps for gradient accumulation.
            print_statistics (bool, optional): whether to print statistics during training. Defaults to False.
            use_scaler (bool, optional): whether to use scaler. Defaults to False.

        Throws:
            ValueError: if accumulation_steps is not a positive integer
        """
        if accumulation_steps <= 0:
            raise ValueError("❌ accumulation_steps must be a positive integer")

        self.model = model
        self.device = device
        self.criterion = criterion
        self.accumulation_steps = accumulation_steps
        self.evaluation_steps = evaluation_steps
        self.print_statistics = print_statistics
        self.use_scaler = use_scaler

        self.best_eval_val_loss = np.inf
        self.eval_train_loss = 0
        self.eval_val_loss = 0
    
    def train_siamese(
            self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: Optimizer,
            num_epochs: int,
            learning_rate: int = 0,
            save_path: str = 'best_model.pth'
        ) -> dict:
        """Train the Siamese network model.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            optimizer (Optimizer): Optimizer to use during training.
            num_epochs (int): Number of epochs to train.
            save_path (str, optional): Path to save the best model. Defaults to 'best_model.pth'.

        Throws:
            ValueError: If num_epochs is not a positive integer.

        Returns:
            dict: Statistics of the training.
        """
        if num_epochs <= 0:
            raise ValueError("❌ num_epochs must be a positive integer")
        
        # Setup WandB and watch
        wandb.init(
            project="siamese-network", 
            config={
                "architecture": "Siamese",
                "dataset": "Maxar 2023 Turkish Earthquake",
                "epochs": num_epochs,
                "learning_rate": learning_rate
            }
        )
        wandb.watch(self.model, log_freq=4, log="all")

        print(f"🚀 Training siamese model for {num_epochs} epochs...")

        # Recording statistics for each epoch
        statistics = {
            'train_loss': [],
            'val_loss': [],
            'train_loader_length': len(train_loader),
            'val_loader_length': len(val_loader),
            'num_epochs': num_epochs,
            'evaluation_steps': self.evaluation_steps,
        }

        # Scaler
        scaler = GradScaler(enabled=self.use_scaler)
        
        with tqdm(range(num_epochs), desc='Epochs', unit='epoch') as bar:
            for _ in bar:
                self._train_one_epoch_siamese(
                    train_loader, 
                    val_loader, 
                    optimizer, 
                    statistics, 
                    scaler, 
                    save_path = save_path,
                    bar = bar
                )

        wandb.finish()
        
        return statistics

    def _train_one_epoch_siamese(
            self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: Optimizer,
            statistics: dict,
            scaler: GradScaler,
            save_path = 'best_model.pth',
            bar: tqdm = None,
        ):
        self.model.train()

        total_train_loss = 0
        n_train_loss = 0
        for batch_idx, (pre, post, label) in enumerate(train_loader):
            # Move the data to the device
            pre, post = pre.to(self.device), post.to(self.device)
            label = label.to(self.device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            with autocast(enabled=self.use_scaler):
                pre_embeddings, post_embeddings = self.model(pre, post)

                # Apply similarity function and compute loss
                train_loss = self.criterion(pre_embeddings, post_embeddings, label)
                total_train_loss += train_loss.item()
                n_train_loss += 1

            scaler.scale(train_loss).backward()

            # Optimize every accumulation steps
            if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()

            if bar is not None:
                bar.set_postfix({
                    'batch': f"{batch_idx + 1}/{len(train_loader)}", 
                    'train_loss': f"{self.eval_train_loss:.4f}", 
                    'val_loss': f"{self.eval_val_loss:.4f}"}
                )

            if (batch_idx + 1) % self.evaluation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                # Get validation loss and update best model
                self.eval_val_loss = self._evaluate_siamese(val_loader)
                if self.eval_val_loss < self.best_eval_val_loss:
                    print(f"🎉 Saving model with new best loss: {self.eval_val_loss:.4}")
                    statistics['val_loss'].append(self.eval_val_loss)
                    torch.save(self.model.state_dict(), save_path)
                    self.best_eval_val_loss = self.eval_val_loss

                # Get and update training loss
                self.eval_train_loss = total_train_loss / n_train_loss
                statistics['train_loss'].append(train_loss)
                total_train_loss = 0
                n_train_loss = 0

                if bar is None and self.print_statistics:
                    print(f"➡️ Training loss: {self.eval_train_loss:.4f}, Validation loss: {self.eval_val_loss:.4f}")
                else:
                    bar.set_postfix({
                        'batch': f"{batch_idx + 1}/{len(train_loader)}", 
                        'train_loss': f"{self.eval_train_loss:.4f}", 
                        'val_loss': f"{self.eval_val_loss:.4f}"}
                    )
                wandb.log({
                    "train_loss": self.eval_train_loss,
                    "val_loss": self.eval_val_loss
                })

    def _evaluate_siamese(self, loader: DataLoader):
        self.model.eval()

        total_val_loss = 0
        with torch.no_grad():
            for pre, post, label in loader:
                # Move the data to the device
                pre, post = pre.to(self.device), post.to(self.device)
                label = label.to(self.device)

                # Forward pass
                with autocast(enabled=self.use_scaler):
                    pre_embeddings, post_embeddings = self.model(pre, post)

                    # Apply similarity function and compute loss
                    val_loss = self.criterion(pre_embeddings, post_embeddings, label)
                    total_val_loss += val_loss.item()

        total_val_loss /= len(loader)
        
        wandb.save('models/best_model.onnx')
        
        return total_val_loss