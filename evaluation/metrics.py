import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (1 - quantile) * error))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def spread_direction_accuracy(y_true_spread: np.ndarray, y_pred_spread: np.ndarray) -> float:
    true_sign = np.sign(y_true_spread)
    pred_sign = np.sign(y_pred_spread)
    return np.mean(true_sign == pred_sign)


def spike_hit_rate(y_true_spike: np.ndarray, y_pred_prob: np.ndarray,
                   threshold: float = 0.5) -> dict:
    pred_spike = (y_pred_prob >= threshold).astype(int)
    tp = np.sum((pred_spike == 1) & (y_true_spike == 1))
    fp = np.sum((pred_spike == 1) & (y_true_spike == 0))
    fn = np.sum((pred_spike == 0) & (y_true_spike == 1))
    tn = np.sum((pred_spike == 0) & (y_true_spike == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def quantile_coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    return np.mean((y_true >= y_lower) & (y_true <= y_upper))


def winkler_score(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray,
                  alpha: float = 0.2) -> float:
    width = y_upper - y_lower
    penalty_lower = (2 / alpha) * (y_lower - y_true) * (y_true < y_lower)
    penalty_upper = (2 / alpha) * (y_true - y_upper) * (y_true > y_upper)
    return np.mean(width + penalty_lower + penalty_upper)


def compute_all_metrics(y_true: np.ndarray, predictions: dict,
                        quantiles: list = None) -> dict:
    from config import QUANTILES, QUANTILE_LABELS
    quantiles = quantiles or QUANTILES

    metrics = {}

    if "P50" in predictions:
        metrics["MAE"] = mean_absolute_error(y_true, predictions["P50"])
        metrics["RMSE"] = rmse(y_true, predictions["P50"])

    for q, label in zip(quantiles, QUANTILE_LABELS):
        if label in predictions:
            metrics[f"pinball_{label}"] = pinball_loss(y_true, predictions[label], q)

    if "P10" in predictions and "P90" in predictions:
        metrics["coverage_80"] = quantile_coverage(y_true, predictions["P10"], predictions["P90"])
        metrics["winkler_80"] = winkler_score(y_true, predictions["P10"], predictions["P90"], alpha=0.2)

    return metrics
