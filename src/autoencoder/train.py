import sys
sys.path.append('../../')

import torch
from torch.utils.data import DataLoader

from src.autoencoder.autoencoder import ConvAutoencoder
from src.data.dataset import get_split_image_files, SatelliteImageDataset
from src.utils.trainer import Trainer

DATA_PATH = '../../data/'
MAXAR_PRE_PATH = DATA_PATH + 'maxar_patches_city/pre/'

EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
ACCUMULATION_STEPS = 1
NUM_WORKERS = 1 #12

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dataloaders(directory, batch_size, num_workers):
    train_image_files, test_image_files, val_image_files = get_split_image_files(directory)

    train_dataset = SatelliteImageDataset(image_files=train_image_files)
    test_dataset = SatelliteImageDataset(image_files=test_image_files)
    val_dataset = SatelliteImageDataset(image_files=val_image_files)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'✅ Train dataloader length: {len(train_dataloader)}')
    print(f'✅ Test dataloader length: {len(test_dataloader)}')
    print(f'✅ Val dataloader length: {len(val_dataloader)}')

    return train_dataloader, test_dataloader, val_dataloader

def get_model():
    return ConvAutoencoder(
        encoder_channels=[3, 32, 64, 128, 512], 
        decoder_channels=[512, 128, 64, 32, 3],
        stride=2,
        kernel_size=3,
        padding=1,
        output_padding=1,
    ).to(DEVICE)

def get_criterion():
    return torch.nn.MSELoss()

def get_optimizer(autoencoder, learning_rate):
    return torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

def get_trainer(model, criterion):
    return Trainer(
        model=model, 
        device=DEVICE,
        criterion=criterion,
        accumulation_steps=ACCUMULATION_STEPS,
        print_statistics=True
    )

if __name__ == '__main__':
    train_dataloader, _, val_dataloader = get_dataloaders(MAXAR_PRE_PATH, BATCH_SIZE, NUM_WORKERS)
    autoencoder = get_model()
    criterion = get_criterion()
    optimizer = get_optimizer(autoencoder, LEARNING_RATE)
    trainer = get_trainer(autoencoder, criterion)
    statistics = trainer.train_autoencoder(
        train_loader=train_dataloader, 
        val_loader=val_dataloader,
        optimizer=optimizer,
        num_epochs=EPOCHS
    )