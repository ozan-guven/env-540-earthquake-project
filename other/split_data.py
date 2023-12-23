### OK, NOT USED
# This script is used to split the data into train, val and test.

import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import shutil
import random
import numpy as np

from src.utils.random import set_seed
from src.config import SEED

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
TRAIN_SPLIT, TEST_SPLIT, VAL_SPLIT = 0.7, 0.15, 0.15

if __name__ == "__main__":
    set_seed(SEED)

    if np.abs(TRAIN_SPLIT + TEST_SPLIT + VAL_SPLIT - 1) > 1e-6:
        raise ValueError("❌ The sum of the splits must be equal to 1.")
    
    # Create three folders, train, val and test
    os.makedirs(f"{DATA_PATH}train/damaged", exist_ok=True)
    os.makedirs(f"{DATA_PATH}train/intact", exist_ok=True)
    
    os.makedirs(f"{DATA_PATH}val/damaged", exist_ok=True)
    os.makedirs(f"{DATA_PATH}val/intact", exist_ok=True)
    
    os.makedirs(f"{DATA_PATH}test/damaged", exist_ok=True)
    os.makedirs(f"{DATA_PATH}test/intact", exist_ok=True)

    # We get the files in "../data/damaged" and "../data/intact"
    damaged_folders = os.listdir(f"{DATA_PATH}damaged")
    intact_folders = os.listdir(f"{DATA_PATH}intact")
    
    # Split the data into train, val and test
    for folders, key in [(damaged_folders, 'damaged'), (intact_folders, 'intact')]:
        for folder in folders:
            files = os.listdir(f"{DATA_PATH}{key}/{folder}")
            random.shuffle(files)
            
            train_files = files[:int(len(files) * TRAIN_SPLIT)]
            val_files = files[int(len(files) * TRAIN_SPLIT):int(len(files) * (TRAIN_SPLIT + VAL_SPLIT))]
            test_files = files[int(len(files) * (TRAIN_SPLIT + VAL_SPLIT)):]
            
            for file in train_files:
                shutil.copy(f"{DATA_PATH}{key}/{folder}/{file}", f"{DATA_PATH}train/{key}/{file}")
            for file in val_files:
                shutil.copy(f"{DATA_PATH}{key}/{folder}/{file}", f"{DATA_PATH}val/{key}/{file}")
            for file in test_files:
                shutil.copy(f"{DATA_PATH}{key}/{folder}/{file}", f"{DATA_PATH}test/{key}/{file}")