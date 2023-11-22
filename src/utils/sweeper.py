# This class is used to tune the parameters of the model
import sys

sys.path.append('../../')

import numpy as np
import torch
import wandb
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.siamese import Siamese

SIAMESE_CONV_CHANNELS = [[3, 16, 16], [16, 32, 32], [32, 64, 64, 64], [64, 128, 128, 128]]
SIAMESE_EMBEDDING_SIZE = 32

class Sweeper():
    def __init__(self, config, criterion, evaluation_steps, use_scaler, device):
        self.config = config
        self.criterion = criterion
        self.evaluation_steps = evaluation_steps
        self.use_scaler = use_scaler
        self.device = device
        
        self.best_eval_val_loss = np.inf
        self.eval_train_loss = 0
        self.eval_val_loss = 0
    
    def _get_optimizer(self, optimizer, learning_rate, weight_decay):
        match optimizer:
            case "adam": return torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            case "adamw": return torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            case "sgd": return torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            case _: raise ValueError("❌ optimizer must be one of 'adam', 'adamW', 'sgd'")

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            save_path: str = 'sweeps/best_model.pth'
        ) -> dict:
        # Setup WandB and watch
        with wandb.init(config=self.config):
            config = wandb.config
            num_epochs = config.epochs

            self.model = Siamese(conv_channels = SIAMESE_CONV_CHANNELS,
                                 embedding_size = config.siamese_embedding_size,
                                 dropout_rate = config.dropout_rate,
            ).to(self.device)
            
            optimizer = self._get_optimizer(config.optimizer, config.learning_rate, config.weight_decay)
            self.accumulation_steps = config.accumulation_step
            
            if num_epochs <= 0:
                raise ValueError("❌ num_epochs must be a positive integer")
            
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