import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SampleWeightCalculator:
    def __init__(self, config: dict = None):
        from config import TRAINING_WEIGHTS
        self.config = config or TRAINING_WEIGHTS.copy()

    def compute(self, df: pd.DataFrame, target_col: str = None) -> np.ndarray:
        weights = np.ones(len(df))

        weights = self._apply_recent_weight(df, weights)
        weights = self._apply_holiday_weight(df, weights)

        if target_col and target_col in df.columns:
            weights = self._apply_extreme_weight(df, weights, target_col)

        return weights

    def _apply_recent_weight(self, df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        w = self.config.get("recent_30d", 2.0)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            now = pd.Timestamp.now(tz=df.index.tz)
        else:
            now = pd.Timestamp.now()

        cutoff = now - pd.Timedelta(days=30)
        recent_mask = df.index >= cutoff
        weights[recent_mask] *= w
        return weights

    def _apply_holiday_weight(self, df: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        w = self.config.get("holiday", 2.5)
        if "is_holiday" in df.columns:
            holiday_mask = df["is_holiday"] == 1
            weights[holiday_mask] *= w
        if "is_post_holiday" in df.columns:
            post_mask = df["is_post_holiday"] == 1
            weights[post_mask] *= w * 0.8
        return weights

    def _apply_extreme_weight(self, df: pd.DataFrame, weights: np.ndarray,
                               target_col: str) -> np.ndarray:
        values = df[target_col].dropna()
        if len(values) == 0:
            return weights

        q10 = values.quantile(0.1)
        q90 = values.quantile(0.9)

        if "price" in target_col.lower():
            w = self.config.get("extreme_price", 2.0)
        elif "load" in target_col.lower():
            w = self.config.get("extreme_load", 2.0)
        elif "renewable" in target_col.lower():
            w = self.config.get("abnormal_renewable", 2.0)
        else:
            w = 1.5

        extreme_mask = (df[target_col] <= q10) | (df[target_col] >= q90)
        weights[extreme_mask] *= w

        return weights

    def compute_for_training(self, df: pd.DataFrame,
                              target_cols: list = None) -> np.ndarray:
        if target_cols is None:
            target_cols = []

        weights = self.compute(df)

        for col in target_cols:
            if col in df.columns:
                col_weights = self.compute(df, target_col=col)
                weights = np.maximum(weights, col_weights)

        max_w = weights.max()
        if max_w > 0:
            weights = weights / max_w

        return weights
