import numpy as np
import itertools

import torch
from torch import nn

from tqdm.notebook import tqdm

class Trainer():
    def __init__(self, model, device, criterion, accum_iter=50, print_statistics=False):
        """ 
            The Trainer need to receive the model and the device.
        """

        self.model = model
        self.device = device
        self.criterion = criterion
        self.accum_iter = accum_iter # Number of iterations to accumulate gradients (creating a batch from multiple samples)
        self.print_statistics = print_statistics
        
    def train(self, train_loader, val_loader, optimizer, num_epochs):
        """
            The train function takes as input the train and validation loaders, the optimizer, and the number of epochs.
            It returns the training and validation loss and accuracy for each epoch.
        """
        # Recording the loss and accuracy for each epoch
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
        
        best_f1 = 0
        
        with tqdm(range(num_epochs), desc='Epochs') as pbar_epochs:
            for epoch in pbar_epochs:
                if self.print_statistics: print(f"Epoch {epoch}/{num_epochs}")
                self._train_one_epoch(train_loader, optimizer, statistics)
                validation_loss, f1 = self._evaluate(val_loader, statistics)
                        
                # If f1 is better than the previous best, save the model
                if f1 > best_f1:
                    statistics['best_index'] = epoch
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    best_f1 = f1
                    
        return statistics
    
    def _train_one_epoch(self, train_loader, optimizer, statistics):
        # Train the model
        self.model.train()
        optimizer.zero_grad()
        
        epoch_loss = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for batch_idx, data in enumerate(train_loader):
            # Move the data to the device
            x = data['node_feat'].to(self.device)
            edge_attr = data['edge_attr'].to(self.device)
            adj = data['adj'].to(self.device)
            y = data['y'].to(self.device)

            # Forward pass
            outputs = self.model(x, adj, edge_attr)

            loss = self.criterion(outputs, y.float()) / self.accum_iter
            loss.backward()
            
            # Optimize every accum_iter iterations
            if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                # Zero the parameter gradients
                optimizer.zero_grad()
            
            pred = int(torch.sigmoid(outputs) > 0.5)
            
            epoch_loss += loss.item()
            tp += int(pred == 1 and y == 1)
            tn += int(pred == 0 and y == 0)
            fp += int(pred == 1 and y == 0)
            fn += int(pred == 0 and y == 1)
                
        # Calculate the training statistics
        acc, rec, prec, f1 = self._copmute_statistics(tp, tn, fp, fn)
        
        # Record the training statistics
        statistics['train_loss'].append(epoch_loss / len(train_loader))
        statistics['train_acc'].append(acc)
        statistics['train_cm'].append([tp, tn, fp, fn])
        
        statistics['train_rec'].append(rec)
        statistics['train_prec'].append(prec)
        statistics['train_f1'].append(f1)
        if self.print_statistics:
            print(f"Train loss: {epoch_loss / len(train_loader):.4f}, Train accuracy: {acc:.2%}, Train F1: {f1:.2%}")
        
    def _evaluate(self, loader, statistics):
        self.model.eval()
        total_loss = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        with torch.no_grad():
            for data in loader:
                # Move the data to the device
                x = data['node_feat'].to(self.device)
                adj = data['adj'].to(self.device)
                edge_attr = data['edge_attr'].to(self.device)
                y = data['y'].to(self.device)

                # Forward pass
                outputs = self.model(x, adj, edge_attr)
                loss = self.criterion(outputs, y.float())
                
                pred = int(torch.sigmoid(outputs) > 0.5)

                total_loss += loss.item()
                tp += int(pred == 1 and y == 1)
                tn += int(pred == 0 and y == 0)
                fp += int(pred == 1 and y == 0)
                fn += int(pred == 0 and y == 1)
                
            # Calculate the evaluation statistics
            acc, rec, prec, f1 = self._copmute_statistics(tp, tn, fp, fn)
            
            # Record the evaluation statistics
            total_loss = total_loss / len(loader)
            statistics['val_loss'].append(total_loss)
            statistics['val_acc'].append(acc)
            statistics['val_cm'].append([tp, tn, fp, fn])
            
            statistics['val_rec'].append(rec)
            statistics['val_prec'].append(prec)
            statistics['val_f1'].append(f1)
            
            if self.print_statistics:
                print(f"Validation loss: {total_loss / len(loader):.4f}, Validation accuracy: {acc:.2%}, Validation F1: {f1:.2%}")
            
        return total_loss, f1
    
    def test(self, test_loader, threshold, best_model_path='best_model.pth'):
        if best_model_path is not None:
            self.model.load_state_dict(torch.load(best_model_path))
        
        self.model.eval()
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        with torch.no_grad():
            for data in test_loader:
                # Move the data to the device
                x = data['node_feat'].to(self.device)
                adj = data['adj'].to(self.device)
                edge_attr = data['edge_attr'].to(self.device)
                y = data['y'].to(self.device)

                # Forward pass
                outputs = self.model(x, adj, edge_attr)
                pred = int(torch.sigmoid(outputs) > threshold)

                tp += int(pred == 1 and y == 1)
                tn += int(pred == 0 and y == 0)
                fp += int(pred == 1 and y == 0)
                fn += int(pred == 0 and y == 1)
        
        return self._copmute_statistics(tp, tn, fp, fn), (tp, tn, fp, fn)
    
    @staticmethod
    def tune(model_class, num_features, train_loader, val_loader, num_epochs, device):
        """Tune the hyperparameters of the model using grid search."""
        # conv_dims=[2**8, 2**6, 2**4, 2] , activation=nn.LeakyReLU(), dropout=0.2, learning rate, optimizer, weight decay
        conv_dims = [[2**2, 2], 
                     [2**3, 2**2, 2], 
                     [2**4, 2**3, 2**2, 2]]
        activation = nn.ReLU()
        dropouts = [0, 0.3]
        lrs = np.logspace(-4, -1, 5)
        weight_decays = [0, 1e-3]
        # Do the cartesian product of all the hyperparameters with their name, i.e. a dictionnary product
        hyperparameters = {
            'conv_dims': conv_dims,
            'dropout': dropouts,
            'lr': lrs,
            'weight_decay': weight_decays
        }
        # Create a list of all the possible combinations of hyperparameters
        statistics_list = []
        best_val_loss = float('inf')
        best_hyperparameter = None
        best_statistics = None
        hyperparameters = [dict(zip(hyperparameters.keys(), values)) for values in itertools.product(*hyperparameters.values())]
        for hyperparameter in tqdm(hyperparameters, desc='Fine-tuning', leave=False):
            #print(f"Training with hyperparameters: {hyperparameter}")
            # Create the model with the hyperparameters
            model = model_class(num_features, conv_dims=hyperparameter['conv_dims'], activation=activation, dropout=hyperparameter['dropout']).to(device)
            # Create the trainer
            trainer = Trainer(model, device, nn.BCEWithLogitsLoss(), print_statistics=False)
            # Create the optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameter['lr'], weight_decay=hyperparameter['weight_decay'])
            # Train the model
            statistics = trainer.train(train_loader, val_loader, optimizer, None, num_epochs=num_epochs)
            
            # find the lowest validation loss
            val_loss = min(statistics['val_loss'])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hyperparameter = hyperparameter
                best_statistics = statistics
            
            # Record the statistics with the hyperparameters
            statistics_list.append((statistics, hyperparameter))
        return statistics_list, best_statistics, best_hyperparameter

    @staticmethod
    def _copmute_statistics(tp, tn, fp, fn):
        assert tp + tn + fp + fn > 0
        acc = (tp + tn) / (tp + tn + fp + fn)
        rec = 0 if (tp + fn) == 0 else tp / (tp + fn)
        prec = 0 if (tp + fp) == 0 else tp / (tp + fp)
        f1 = 0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        
        return acc, rec, prec, f1