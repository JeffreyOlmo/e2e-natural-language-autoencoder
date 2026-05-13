"""Loss helpers for AR (MSE on L2-normalized vectors).

The reconstruction loss is direction-only: with both vectors normalized to
unit norm, MSE = (1/d) · ||p̂ − ĝ||² = 2/d · (1 − cos(p, g)).

FVE = 1 − MSE / MSE_predict_mean, where MSE_predict_mean is the loss of
constantly predicting normalize(mean(h_train)) on the eval set.
"""
import torch
import torch.nn.functional as F


def normalize_to(v: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return F.normalize(v, dim=-1) * scale


def ar_mse_loss(pred: torch.Tensor, gold: torch.Tensor, mse_norm: float = 1.0) -> torch.Tensor:
    p = normalize_to(pred, mse_norm)
    g = normalize_to(gold, mse_norm)
    return ((p - g) ** 2).mean()


def per_sample_mse(pred: torch.Tensor, gold: torch.Tensor, mse_norm: float = 1.0) -> torch.Tensor:
    """[B] per-sample MSE, mean over dims. Use for FVE numerator."""
    p = normalize_to(pred, mse_norm)
    g = normalize_to(gold, mse_norm)
    return ((p - g) ** 2).mean(dim=-1)


def predict_mean_mse(h_all: torch.Tensor, mse_norm: float = 1.0) -> float:
    """MSE of constant pred = mean(h). The 'baseline' for FVE."""
    mu = h_all.mean(dim=0, keepdim=True).expand_as(h_all)
    return per_sample_mse(mu, h_all, mse_norm).mean().item()


def fve(pred: torch.Tensor, gold: torch.Tensor, baseline_mse: float, mse_norm: float = 1.0) -> torch.Tensor:
    """1 − MSE/baseline_mse. Returns a scalar tensor on pred's device."""
    mse = ar_mse_loss(pred, gold, mse_norm)
    return 1.0 - mse / baseline_mse
