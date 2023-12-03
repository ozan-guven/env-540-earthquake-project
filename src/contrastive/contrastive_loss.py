import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()

        self.margin = margin

    def forward(self, pre, post, label):
        # Compute the contrastive loss
        embedding_distance = pre - post
        norm_squared = torch.sum(embedding_distance.pow(2), dim=1)
        contrastive_loss = (1 - label) * norm_squared + label * torch.clamp(self.margin - norm_squared, min=0.0).pow(2)
        
        # Average over batch
        contrastive_loss = torch.mean(contrastive_loss)
        
        return contrastive_loss