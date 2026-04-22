import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from models.base import GBDTQuantileModel

logger = logging.getLogger(__name__)


class MoERouter:
    def __init__(self):
        from config import EXPERT_TYPES
        self.expert_types = EXPERT_TYPES

    def get_expert_label(self, df: pd.DataFrame) -> pd.Series:
        if "is_holiday" in df.columns and "is_post_holiday" in df.columns:
            labels = pd.Series("normal_day", index=df.index)
            labels.loc[df["is_holiday"] == 1] = "holiday"
            labels.loc[df["is_post_holiday"] == 1] = "post_holiday"
            return labels
        return pd.Series("normal_day", index=df.index)


class HolidayExpertEnsemble:
    def __init__(self, quantiles=None):
        from config import QUANTILES
        self.quantiles = quantiles or QUANTILES
        self.router = MoERouter()
        self.global_model = None
        self.residual_models = {}

    def train_global(self, X: pd.DataFrame, y: pd.Series,
                     sample_weight=None, **kwargs):
        logger.info("  [MoE] Training global model ...")
        self.global_model = GBDTQuantileModel(quantiles=self.quantiles, name="moe_global")
        self.global_model.fit(X, y, sample_weight=sample_weight, **kwargs)
        return self

    def train_residual_experts(self, X: pd.DataFrame, y: pd.Series,
                                sample_weight=None):
        logger.info("  [MoE] Training residual correction experts ...")
        if self.global_model is None:
            raise RuntimeError("Train global model first")

        global_pred = self.global_model.predict(X)
        labels = self.router.get_expert_label(X)

        for expert_type in self.router.expert_types:
            mask = labels == expert_type
            if mask.sum() < 50:
                logger.warning(f"    Expert '{expert_type}': only {mask.sum()} samples, skipping")
                continue

            X_expert = X[mask]
            y_expert = y[mask]
            global_p50 = global_pred["P50"][mask.values]

            residual = y_expert.values - global_p50

            residual_model = GBDTQuantileModel(
                quantiles=self.quantiles, name=f"residual_{expert_type}"
            )
            w_expert = sample_weight[mask] if sample_weight is not None else None
            residual_model.fit(X_expert, pd.Series(residual, index=X_expert.index), sample_weight=w_expert)
            self.residual_models[expert_type] = residual_model
            logger.info(f"    Expert '{expert_type}': {mask.sum()} samples, residual model trained")

    def predict(self, X: pd.DataFrame, blend_weight: float = 0.7) -> Dict[str, np.ndarray]:
        if self.global_model is None:
            raise RuntimeError("Global model not trained")

        global_pred = self.global_model.predict(X)
        labels = self.router.get_expert_label(X)

        results = {}
        for label in global_pred:
            results[label] = global_pred[label].copy()

        for expert_type in self.residual_models:
            mask = labels == expert_type
            if mask.sum() == 0:
                continue

            indices = np.where(mask)[0]
            X_expert = X.iloc[indices]
            residual_pred = self.residual_models[expert_type].predict(X_expert)

            for label in results:
                corrected = results[label][indices] + blend_weight * residual_pred[label]
                results[label][indices] = corrected

        self._enforce_monotonicity(results)
        return results

    def _enforce_monotonicity(self, results):
        if "P10" in results and "P50" in results and "P90" in results:
            results["P50"] = np.maximum(results["P10"], np.minimum(results["P90"], results["P50"]))
            results["P10"] = np.minimum(results["P10"], results["P50"])
            results["P90"] = np.maximum(results["P90"], results["P50"])

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        if self.global_model is not None:
            self.global_model.save(os.path.join(model_dir, "moe_global.pkl"))
        for expert_type, model in self.residual_models.items():
            model.save(os.path.join(model_dir, f"residual_{expert_type}.pkl"))

    def load(self, model_dir: str):
        global_path = os.path.join(model_dir, "moe_global.pkl")
        if os.path.exists(global_path):
            self.global_model = GBDTQuantileModel(quantiles=self.quantiles, name="moe_global")
            self.global_model.load(global_path)
        for expert_type in self.router.expert_types:
            path = os.path.join(model_dir, f"residual_{expert_type}.pkl")
            if os.path.exists(path):
                model = GBDTQuantileModel(quantiles=self.quantiles, name=f"residual_{expert_type}")
                model.load(path)
                self.residual_models[expert_type] = model
