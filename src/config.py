# This file contains all the constants used in the project

import torch

IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGE_SIZE = 1024
BATCH_SIZE = 4
POS_WEIGHT = torch.tensor([16.2578])
SIAMESE_EMBEDDING_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
