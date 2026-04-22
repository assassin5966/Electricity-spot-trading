import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


def quantile_loss(pred: torch.Tensor, target: torch.Tensor,
                  quantiles: list = None) -> torch.Tensor:
    from config import QUANTILES
    quantiles = quantiles or QUANTILES
    losses = []
    for i, q in enumerate(quantiles):
        error = target - pred[..., i]
        loss = torch.max(q * error, (1 - q) * error)
        losses.append(loss.mean())
    return sum(losses) / len(losses)


def spike_classification_loss(pred: torch.Tensor, target: torch.Tensor,
                              pos_weight: float = 5.0) -> torch.Tensor:
    weights = torch.where(target == 1, pos_weight, 1.0)
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return (bce * weights).mean()


def combined_price_loss(pred_dict: dict, target_dict: dict,
                        quantiles: list = None) -> torch.Tensor:
    from config import PRICE_LOSS_WEIGHTS, QUANTILES
    quantiles = quantiles or QUANTILES
    w = PRICE_LOSS_WEIGHTS

    total_loss = torch.tensor(0.0, device=next(iter(pred_dict.values())).device
                              if pred_dict else "cpu")

    if "load" in pred_dict and "load" in target_dict:
        load_loss = quantile_loss(pred_dict["load"], target_dict["load"], quantiles)
        total_loss = total_loss + w["quantile_load"] * load_loss

    if "price" in pred_dict and "price" in target_dict:
        price_loss = quantile_loss(pred_dict["price"], target_dict["price"], quantiles)
        total_loss = total_loss + w["quantile_price"] * price_loss

    if "spike" in pred_dict and "spike" in target_dict:
        spike_loss = spike_classification_loss(pred_dict["spike"], target_dict["spike"])
        total_loss = total_loss + w["spike_classification"] * spike_loss

    return total_loss


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: list = None):
        super().__init__()
        from config import QUANTILES
        self.quantiles = quantiles or QUANTILES

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return quantile_loss(pred, target, self.quantiles)


class SpikeLoss(nn.Module):
    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return spike_classification_loss(pred, target, self.pos_weight)


class CombinedPriceLoss(nn.Module):
    def __init__(self, quantiles: list = None):
        super().__init__()
        from config import QUANTILES
        self.quantiles = quantiles or QUANTILES

    def forward(self, pred_dict: dict, target_dict: dict) -> torch.Tensor:
        return combined_price_loss(pred_dict, target_dict, self.quantiles)
