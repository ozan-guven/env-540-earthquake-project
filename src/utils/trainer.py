import itertools
import numpy as np
from typing import Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import torch
from torch import nn
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
        accumulation_steps: int = 50, 
        print_statistics: bool = False
        ):
        """Initialize the trainer.

        Args:
            model (nn.Module): model to train
            device (str): device to use for training
            criterion (nn.Module): loss function to use
            accumulation_steps (int, optional): accumulation steps for gradient accumulation. Defaults to 50.
            print_statistics (bool, optional): _description_. Defaults to False.

        Throws:
            ValueError: if accumulation_steps is not a positive integer
        """
        if accumulation_steps <= 0:
            raise ValueError("❌ accumulation_steps must be a positive integer")

        self.model = model
        self.device = device
        self.criterion = criterion
        self.accumulation_steps = accumulation_steps
        self.print_statistics = print_statistics
        
    def train_autoencoder(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer : Optimizer,
        num_epochs: int,
        save_path: str = 'best_model.pth',
        ) -> dict:
        """Train the model.

        Args:
            train_loader (DataLoader): training data loader
            val_loader (DataLoader): validation data loader
            optimizer (Optimizer): optimizer to use during training
            num_epochs (int): number of epochs to train
            save_path (str, optional): path to save the best model. Defaults to 'best_model.pth'.

        Throws:
            ValueError: if num_epochs is not a positive integer

        Returns:
            dict: statistics of the training
        """
        if num_epochs <= 0:
            raise ValueError("❌ num_epochs must be a positive integer")

        print(f"🚀 Training autoencoder model for {num_epochs} epochs...")

        # Recording statistics for each epoch
        statistics = {
            'train_loss': [],
            'val_loss': [],
            'best_index': 0
        }
        
        # Iterate over epochs
        best_loss = np.inf
        with tqdm(range(num_epochs), desc='Epochs') as bar:
            for epoch in bar:
                self._train_one_epoch_autoencoder(train_loader, optimizer, statistics, bar)
                loss = self._evaluate_autoencoder(val_loader, statistics, bar)
                        
                # If loss is better than the previous best, save the model
                if loss < best_loss:
                    print(f"🎉 Saving model with new best loss: {loss:.2}")
                    statistics['best_index'] = epoch
                    torch.save(self.model.state_dict(), save_path)
                    best_loss = loss
                    
        return statistics

    def train(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer : Optimizer,
        num_epochs: int,
        save_path: str = 'best_model.pth',
        ) -> dict:
        """Train the model.

        Args:
            train_loader (DataLoader): training data loader
            val_loader (DataLoader): validation data loader
            optimizer (Optimizer): optimizer to use during training
            num_epochs (int): number of epochs to train
            save_path (str, optional): path to save the best model. Defaults to 'best_model.pth'.

        Throws:
            ValueError: if num_epochs is not a positive integer

        Returns:
            dict: statistics of the training
        """
        if num_epochs <= 0:
            raise ValueError("❌ num_epochs must be a positive integer")

        print(f"🚀 Training model for {num_epochs} epochs...")

        # Recording statistics for each epoch
        statistics = {
            'train_loss': [],
            'train_acc': [],
            'train_cm': [],
            'train_rec': [],
            'train_prec': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_cm': [],
            'val_rec': [],
            'val_prec': [],
            'val_f1': [],
            'best_index': 0
        }
        
        # Iterate over epochs
        best_f1 = 0
        with tqdm(range(num_epochs), desc='Epochs') as pbar_epochs:
            for epoch in pbar_epochs:
                self._train_one_epoch(train_loader, optimizer, statistics)
                _, f1 = self._evaluate(val_loader, statistics)
                        
                # If f1 is better than the previous best, save the model
                if f1 > best_f1:
                    print(f"🎉 Saving model with new best F1 score: {f1:.2%}")
                    statistics['best_index'] = epoch
                    torch.save(self.model.state_dict(), save_path)
                    best_f1 = f1
                    
        return statistics

    def _train_one_epoch_autoencoder(
        self, 
        train_loader: DataLoader,
        optimizer: Optimizer,
        statistics: dict,
        bar = None
        ) -> Tuple[float, float]:
        """Train the model for one epoch.

        Args:
            train_loader (DataLoader): training data loader
            optimizer (Optimizer): optimizer to use during training
            statistics (dict): statistics of the training

        Returns:
            total_loss (float): total loss of the epoch
            f1 (float): F1 score of the epoch
        """
        # Train the model
        self.model.train()
        optimizer.zero_grad()
        
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            # Move the data to the device
            x, y_true = data, data.clone()
            x = x.to(self.device)
            y_true = y_true.to(self.device)

            # Forward pass
            outputs = self.model(x)
            loss = self.criterion(outputs, y_true) / self.accumulation_steps
            loss.backward()

            # Optimize every accumulation steps
            if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad() # Zero the parameter gradients

            if bar is not None:
                bar.set_postfix({'batch': f"{batch_idx + 1}/{len(train_loader)}", 'train_loss': f"{loss.item():.4f}"})
            
            total_loss += loss.item()
        total_loss /= len(train_loader)

        # Record the training statistics
        statistics['train_loss'].append(total_loss)

        if self.print_statistics and bar is None:
            print(f"➡️ Train loss: {total_loss:.4f}")
    
    def _train_one_epoch(
        self, 
        train_loader: DataLoader,
        optimizer: Optimizer,
        statistics: dict,
        ) -> Tuple[float, float]:
        """Train the model for one epoch.

        Args:
            train_loader (DataLoader): training data loader
            optimizer (Optimizer): optimizer to use during training
            statistics (dict): statistics of the training

        Returns:
            total_loss (float): total loss of the epoch
            f1 (float): F1 score of the epoch
        """
        # Train the model
        self.model.train()
        optimizer.zero_grad()
        
        total_loss = 0
        y_trues = []
        y_preds = []
        for batch_idx, data in enumerate(train_loader):
            # Move the data to the device
            x, y_true = data
            x = x.to(self.device)
            y_true = y_true.to(self.device)

            # Forward pass
            outputs = self.model(x)
            loss = self.criterion(outputs, y_true) / self.accumulation_steps
            loss.backward()
            
            # Optimize every accumulation steps
            if ((batch_idx + 1) % self.accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad() # Zero the parameter gradients
            
            y_pred = int(torch.sigmoid(outputs) > 0.5)
            y_trues.append(y_true)
            y_preds.append(y_pred)
            
            total_loss += loss.item()
        total_loss /= len(train_loader)

        # Calculate the training statistics
        acc, rec, prec, f1, cm = self._compute_statistics(y_true, y_pred)
        
        # Record the training statistics
        statistics['train_loss'].append(total_loss)
        statistics['train_acc'].append(acc)
        statistics['train_cm'].append(cm)
        statistics['train_rec'].append(rec)
        statistics['train_prec'].append(prec)
        statistics['train_f1'].append(f1)

        if self.print_statistics:
            print(f"➡️ Train loss: {total_loss:.4f}, Train accuracy: {acc:.2%}, Train F1: {f1:.2%}")
        
    def _evaluate_autoencoder(self, loader, statistics, bar=None):
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for data in loader:
                # Move the data to the device
                x, y_true = data, data.clone()
                x = x.to(self.device)
                y_true = y_true.to(self.device)

                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y_true)
                
                total_loss += loss.item()
            total_loss /= len(loader)

            # Record the evaluation statistics
            total_loss = total_loss / len(loader)
            statistics['val_loss'].append(total_loss)
            
            if self.print_statistics:
                if bar is None:
                    print(f"➡️ Validation loss: {total_loss:.4f}")
                else:
                    bar.set_postfix({'val_loss': f"{total_loss:.4f}"})
            
        return total_loss

    def _evaluate(self, loader, statistics):
        self.model.eval()

        total_loss = 0
        y_trues = []
        y_preds = []
        with torch.no_grad():
            for data in loader:
                # Move the data to the device
                x, y_true = data
                x = x.to(self.device)
                y_true = y_true.to(self.device)

                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y_true)
                
                y_pred = int(torch.sigmoid(outputs) > 0.5)
                y_trues.append(y_true)
                y_preds.append(y_pred)

                total_loss += loss.item()
            total_loss /= len(loader)

            # Calculate the evaluation statistics
            acc, rec, prec, f1, cm = self._compute_statistics(y_true, y_pred)
            
            # Record the evaluation statistics
            total_loss = total_loss / len(loader)
            statistics['val_loss'].append(total_loss)
            statistics['val_acc'].append(acc)
            statistics['val_cm'].append(cm)
            statistics['val_rec'].append(rec)
            statistics['val_prec'].append(prec)
            statistics['val_f1'].append(f1)
            
            if self.print_statistics:
                print(f"➡️ Validation loss: {total_loss:.4f}, Validation accuracy: {acc:.2%}, Validation F1: {f1:.2%}")
            
        return total_loss, f1
    
    def test(
        self,
        test_loader: DataLoader,
        threshold: float = 0.5,
        best_model_path: str = 'best_model.pth'
        ) -> Tuple[float, float, float, float, np.ndarray]:
        """Test the model.

        Args:
            test_loader (DataLoader): test data loader
            threshold (float, optional): threshold to use for the prediction. Defaults to 0.5.
            best_model_path (str, optional): path to the best model. Defaults to 'best_model.pth'.

        Returns:
            Tuple[float, float, float, float, np.ndarray]: accuracy, recall, precision, F1 score and confusion matrix
        """
        if best_model_path is not None:
            self.model.load_state_dict(torch.load(best_model_path))
        
        self.model.eval()
        
        y_trues = []
        y_preds = []
        with torch.no_grad():
            for data in test_loader:
                # Move the data to the device
                x, y_true = data

                # Forward pass
                outputs = self.model(x)
                y_pred = int(torch.sigmoid(outputs) > threshold)
                y_trues.append(y_true)
                y_preds.append(y_pred)
        
        return self._compute_statistics(y_trues, y_preds)
    
    @staticmethod
    def _compute_statistics(y_true: list, y_pred: list) -> Tuple[float, float, float, float, np.ndarray]:
        """Compute the accuracy, recall, precision, F1 score and confusion matrix.

        Args:
            y_true (list): true labels
            y_pred (list): predicted labels

        Throws:
            ValueError: if y_true is empty
            ValueError: if y_true and y_pred have different lengths

        Returns:
            Tuple[float, float, float, float, np.ndarray]: accuracy, recall, precision, F1 score and confusion matrix
        """
        if len(y_true) == 0:
            raise ValueError("❌ y_true must not be empty")
        if len(y_true) != len(y_pred):
            raise ValueError("❌ y_true and y_pred must have the same length")

        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        return acc, rec, prec, f1, cm