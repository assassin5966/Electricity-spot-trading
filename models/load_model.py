import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from models.base import GBDTQuantileModel, DeepQuantileModel

logger = logging.getLogger(__name__)


class LoadModel:
    def __init__(self, quantiles=None, model_dir: str = "./saved_models/load"):
        from config import QUANTILES
        self.quantiles = quantiles or QUANTILES
        self.model_dir = model_dir
        self.gbdt_model = GBDTQuantileModel(quantiles=self.quantiles, name="load_gbdt")
        self.deep_model = None
        self.expert_models = {}
        self.feature_columns = []

    def _build_features(self, df: pd.DataFrame, renewable_pred: Optional[Dict] = None) -> pd.DataFrame:
        from config import LOAD_FEATURES
        features = []

        for col in LOAD_FEATURES["load_forecast"]:
            if col in df.columns:
                features.append(col)

        if renewable_pred is not None:
            for label in ["P10", "P50", "P90"]:
                col = f"renewable_prediction_{label}"
                if label in renewable_pred:
                    df = df.copy()
                    df[col] = renewable_pred[label][:len(df)] if len(renewable_pred[label]) >= len(df) else np.nan
                    features.append(col)

        for col in LOAD_FEATURES["historical"]:
            if col in df.columns:
                features.append(col)

        for col in LOAD_FEATURES["calendar"]:
            if col in df.columns:
                features.append(col)

        self.feature_columns = features
        return df[features].fillna(0)

    def _detect_post_holiday(self, df: pd.DataFrame) -> pd.Series:
        if "is_holiday" not in df.columns:
            return pd.Series(0, index=df.index)
        is_holiday = df["is_holiday"].astype(bool)
        shifted = is_holiday.shift(1).fillna(False)
        return (shifted & ~is_holiday).astype(int)

    def fit_gbdt(self, X: pd.DataFrame, y: pd.Series,
                 sample_weight: Optional[np.ndarray] = None,
                 eval_set: Optional[tuple] = None,
                 renewable_pred: Optional[Dict] = None):
        logger.info("  [LoadModel] Training GBDT baseline ...")
        X_feat = self._build_features(X, renewable_pred)
        self.gbdt_model.fit(X_feat, y, sample_weight=sample_weight, eval_set=eval_set)
        logger.info(f"  [LoadModel] GBDT trained with {len(self.feature_columns)} features")
        return self

    def fit_deep(self, X: np.ndarray, y: np.ndarray,
                 input_dim: int = None,
                 eval_set: Optional[tuple] = None,
                 **kwargs):
        logger.info("  [LoadModel] Training Deep model (TFT with MoE) ...")
        if input_dim is None:
            input_dim = X.shape[-1] if X.ndim == 3 else X.shape[1]

        self.deep_model = DeepQuantileModel(
            quantiles=self.quantiles,
            name="load_deep",
            input_dim=input_dim,
        )
        self.deep_model.fit(X, y, eval_set=eval_set, **kwargs)
        return self

    def fit_experts(self, X: pd.DataFrame, y: pd.Series,
                    expert_labels: pd.Series,
                    sample_weight: Optional[np.ndarray] = None,
                    finetune_epochs: int = 5):
        logger.info("  [LoadModel] Training MoE expert models ...")
        from config import EXPERT_TYPES

        for expert_type in EXPERT_TYPES:
            mask = expert_labels == expert_type
            if mask.sum() == 0:
                logger.warning(f"    No samples for expert '{expert_type}', skipping")
                continue

            X_expert = X[mask]
            y_expert = y[mask]
            w_expert = sample_weight[mask] if sample_weight is not None else None

            expert_model = GBDTQuantileModel(
                quantiles=self.quantiles,
                name=f"load_expert_{expert_type}",
            )
            expert_model.fit(X_expert, y_expert, sample_weight=w_expert)
            self.expert_models[expert_type] = expert_model
            logger.info(f"    Expert '{expert_type}': {mask.sum()} samples")

    def predict(self, X: pd.DataFrame, expert_type: Optional[str] = None,
                renewable_pred: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        X_feat = self._build_features(X, renewable_pred)

        if expert_type is not None and expert_type in self.expert_models:
            return self.expert_models[expert_type].predict(X_feat)

        return self.gbdt_model.predict(X_feat)

    def predict_deep(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if self.deep_model is None:
            raise RuntimeError("Deep model not fitted")
        return self.deep_model.predict(X)

    def predict_with_moe(self, X: pd.DataFrame, expert_weights: Dict[str, float],
                         renewable_pred: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        X_feat = self._build_features(X, renewable_pred)
        base_pred = self.gbdt_model.predict(X_feat)

        weighted_pred = {label: base_pred[label] * (1.0 - sum(expert_weights.values())) for label in base_pred}

        for expert_type, weight in expert_weights.items():
            if expert_type in self.expert_models and weight > 0:
                expert_pred = self.expert_models[expert_type].predict(X_feat)
                for label in base_pred:
                    weighted_pred[label] = weighted_pred[label] + weight * expert_pred[label]

        return weighted_pred

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        self.gbdt_model.save(os.path.join(self.model_dir, "load_gbdt.pkl"))
        if self.deep_model is not None:
            self.deep_model.save(os.path.join(self.model_dir, "load_deep.pt"))
        for expert_type, model in self.expert_models.items():
            model.save(os.path.join(self.model_dir, f"load_expert_{expert_type}.pkl"))

    def load(self):
        self.gbdt_model.load(os.path.join(self.model_dir, "load_gbdt.pkl"))
        deep_path = os.path.join(self.model_dir, "load_deep.pt")
        if os.path.exists(deep_path):
            self.deep_model = DeepQuantileModel(quantiles=self.quantiles, name="load_deep")
            self.deep_model.load(deep_path)
        from config import EXPERT_TYPES
        for expert_type in EXPERT_TYPES:
            path = os.path.join(self.model_dir, f"load_expert_{expert_type}.pkl")
            if os.path.exists(path):
                model = GBDTQuantileModel(quantiles=self.quantiles, name=f"load_expert_{expert_type}")
                model.load(path)
                self.expert_models[expert_type] = model
