import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin):
        """
        Initialize the contrastive loss function.
        Args:
            margin (float): Margin for contrastive loss.
        """
        super().__init__()

        self.margin = margin

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        label: torch.Tensor,
    ):
        """
        Compute the contrastive loss.
        
        Args:
            x (torch.Tensor): Embedding of the first image.
            y (torch.Tensor): Embedding of the second image.
            label (torch.Tensor): Label for the pair.
        """
        # Normalize the embeddings
        x = nn.functional.normalize(x, dim=1)
        y = nn.functional.normalize(y, dim=1)

        # Compute the contrastive loss
        embedding_distance = x - y

        norm_squared = torch.sum(embedding_distance.pow(2), dim=1)
        contrastive_loss = (1 - label) * norm_squared + label * torch.clamp(self.margin - norm_squared, min=0.0).pow(2)
        
        # Average over batch
        contrastive_loss = torch.mean(contrastive_loss).reshape(1)
        
        return contrastive_loss