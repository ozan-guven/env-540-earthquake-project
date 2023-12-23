import torch

IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGE_SIZE = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
