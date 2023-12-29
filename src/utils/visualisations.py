# This file is used for visualising the data.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch import Tensor


def plot_pre_post_mask(
    pre_image: Tensor,
    post_image: Tensor,
    mask: Tensor,
    label: int,
    pred: Tensor = None,
    is_normalised: bool = True,
    with_title: bool = True,
) -> None:
    """
    Plot the pre, post, mask and prediction images (if given) in a grid.

    Args:
        pre_image (torch.Tensor): The pre image
        post_image (torch.Tensor): The post image
        mask (torch.Tensor): The mask
        label (int): The label
        pred (torch.Tensor, optional): The prediction, defaults to None
        is_normalised (bool, optional): Whether the images are normalised, defaults to True
    """

    plt.figure(figsize=(20, 6))
    n_subplots = 3 if pred is None else 5
    gs = gridspec.GridSpec(1, n_subplots, width_ratios=[1] * n_subplots)

    # Pre image
    ax = plt.subplot(gs[0])
    pre_image = pre_image.permute(1, 2, 0)
    if is_normalised:
        pre_image = pre_image * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor(
            [0.485, 0.456, 0.406]
        )
        pre_image = pre_image.clip(0, 1)
    ax.imshow(pre_image)
    ax.set_title("Pre-Event Image")
    ax.axis("off")

    # post
    ax = plt.subplot(gs[1])
    post_image = post_image.permute(1, 2, 0)
    if is_normalised:
        post_image = post_image * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor(
            [0.485, 0.456, 0.406]
        )
        post_image = post_image.clip(0, 1)
    ax.imshow(post_image)
    ax.set_title("Post-Event Image")
    ax.axis("off")

    # mask
    ax = plt.subplot(gs[2])
    ax.imshow(mask)
    ax.set_title("Mask")
    ax.axis("off")

    # Prediction
    if pred is not None:
        ax = plt.subplot(gs[3])
        ax.imshow(pred)
        ax.set_title("Prediction")
        ax.axis("off")

        # Difference
        diff = torch.zeros_like(post_image)
        diff[(mask == 1) & (pred == 1)] = torch.Tensor([0, 1, 0])
        diff[(mask == 1) & (pred == 0)] = torch.Tensor([1, 0, 0])
        diff[(mask == 0) & (pred == 1)] = torch.Tensor([0, 0, 1])
        ax = plt.subplot(gs[4])
        ax.imshow(diff)
        ax.set_title("Difference")
        ax.axis("off")

    # label
    if with_title:
        plt.suptitle(f"{'Damaged' if label else 'Intact'} Sample ({label})")
    plt.tight_layout()
    plt.show()
