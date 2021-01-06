import torch
from typing import Dict


def dice_loss(
    target_masks_pred: torch.Tensor,
    target_masks_true: torch.Tensor,
    eps: float = 1e-5,
) -> float:
    #     pred = pred.contiguous()
    #     target = target.contiguous()
    """
    Compute dice loss for image segmentation problem.

    Parameters
    ----------
    target_masks_pred: torch.Tensor of shape (B * C * H * W)
        Assumes logit values/inverse sigmoid values are passed where
        B: batch_size

        C: channels

        H: height

        W: width

    target_masks_true: torch.Tensor of shape (B * C * H * W)
        Assumes one-hot encoded values are passed where
        B: batch_size

        C: channels

        H: height

        W: width
    """
    target_masks_pred = torch.sigmoid(target_masks_pred)
    intersection = 2.0 * torch.sum(
        target_masks_pred * target_masks_true, dim=(2, 3)
    )
    union = torch.sum(
        torch.square(target_masks_pred) + torch.square(target_masks_true),
        dim=(2, 3),
    )
    loss = 1.0 - (intersection + eps) / (union + eps)
    return torch.mean(loss)


def binary_cross_entropy_loss_with_logits(
    target_masks_pred: torch.Tensor,
    target_masks_true: torch.Tensor,
    eps: float = 1e-10,
):
    """
    Compute binary cross entropy loss.

    Binary cross entropy loss is used in case of multi-label classification.

    Parameters
    ----------
    target_masks_pred: torch.Tensor of shape (B * C * H * W)
        Assumes logit values/ inverse sigmoid values are passed where
        B: batch_size

        C: channels

        H: height

        W: width

    target_masks_true: torch.Tensor of shape (B * C * H * W)
        Assumes target masks are passed as one-hot encoded where
        B: batch_size

        C: channels

        H: height

        W: width
    """
    target_masks_pred = torch.sigmoid(target_masks_pred)
    loss = -(
        target_masks_true * torch.log(target_masks_pred + eps)
        + (1 - target_masks_true) * torch.log(1 - target_masks_pred + eps)
    )
    return torch.mean(loss)


def total_loss(
    target_masks_pred: torch.Tensor,
    target_masks_true: torch.Tensor,
    loss_metrics: Dict[str, float],
    bce_coeff: float = 0.5,
) -> float:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        input=target_masks_pred, target=target_masks_true
    )

    dice = dice_loss(
        target_masks_pred=target_masks_pred,
        target_masks_true=target_masks_true,
    )

    total = bce_coeff * bce + (1 - bce_coeff) * dice

    loss_metrics["bce"] += bce.data.cpu().numpy()
    loss_metrics["dice"] += dice.data.cpu().numpy()
    loss_metrics["total"] += total.data.cpu().numpy()

    return total