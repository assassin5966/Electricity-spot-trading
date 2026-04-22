import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from models.base import GBDTQuantileModel, DeepQuantileModel

logger = logging.getLogger(__name__)


class RenewableModel:
    def __init__(self, quantiles=None, model_dir: str = "./saved_models/renewable"):
        from config import QUANTILES
        self.quantiles = quantiles or QUANTILES
        self.model_dir = model_dir
        self.gbdt_model = GBDTQuantileModel(quantiles=self.quantiles, name="renewable_gbdt")
        self.deep_model = None
        self.feature_columns = []

    def _build_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        from config import RENEWABLE_FEATURES
        features = []

        if "hour_index" in df.columns:
            features.append("hour_index")

        if "day_of_year" not in df.columns and df.index.dtype.kind == "M":
            df = df.copy()
            idx = df.index
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_convert("Asia/Shanghai")
            df["day_of_year"] = idx.day_of_year
            df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
            df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        for col in ["day_of_year", "sin_day_of_year", "cos_day_of_year"]:
            if col in df.columns:
                features.append(col)

        gfs_cols = [c for c in RENEWABLE_FEATURES["gfs_features"] if c in df.columns]
        features.extend(gfs_cols)

        gfs_derived_cols = [c for c in RENEWABLE_FEATURES["gfs_derived"] if c in df.columns]
        features.extend(gfs_derived_cols)

        hist_cols = [c for c in RENEWABLE_FEATURES["history_features"] if c in df.columns]
        features.extend(hist_cols)

        cal_cols = [c for c in RENEWABLE_FEATURES["calendar_features"] if c in df.columns]
        features.extend(cal_cols)

        self.feature_columns = features
        return df[features].fillna(0)

    def fit_gbdt(self, X: pd.DataFrame, y: pd.Series,
                 sample_weight: Optional[np.ndarray] = None,
                 eval_set: Optional[tuple] = None):
        logger.info("  [RenewableModel] Training GBDT baseline ...")
        X_feat = self._build_features(X)
        self.gbdt_model.fit(X_feat, y, sample_weight=sample_weight, eval_set=eval_set)
        logger.info(f"  [RenewableModel] GBDT trained with {len(self.feature_columns)} features")
        return self

    def fit_deep(self, X: np.ndarray, y: np.ndarray,
                 input_dim: int = None,
                 eval_set: Optional[tuple] = None,
                 **kwargs):
        logger.info("  [RenewableModel] Training Deep model (TFT) ...")
        if input_dim is None:
            input_dim = X.shape[-1] if X.ndim == 3 else X.shape[1]

        self.deep_model = DeepQuantileModel(
            quantiles=self.quantiles,
            name="renewable_deep",
            input_dim=input_dim,
        )
        self.deep_model.fit(X, y, eval_set=eval_set, **kwargs)
        return self

    def predict(self, X: pd.DataFrame, use_model: str = "gbdt") -> Dict[str, np.ndarray]:
        if use_model == "gbdt":
            X_feat = self._build_features(X, is_training=False)
            return self.gbdt_model.predict(X_feat)
        elif use_model == "deep" and self.deep_model is not None:
            return self.deep_model.predict(X)
        else:
            raise ValueError(f"Model '{use_model}' not available")

    def predict_ensemble(self, X_gbdt: pd.DataFrame, X_deep: np.ndarray,
                         gbdt_weight: float = 0.4, deep_weight: float = 0.6) -> Dict[str, np.ndarray]:
        gbdt_pred = self.predict(X_gbdt, use_model="gbdt")
        if self.deep_model is not None:
            deep_pred = self.predict(X_deep, use_model="deep")
            results = {}
            for label in gbdt_pred:
                results[label] = gbdt_weight * gbdt_pred[label] + deep_weight * deep_pred[label]
            return results
        return gbdt_pred

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        gbdt_path = os.path.join(self.model_dir, "renewable_gbdt.pkl")
        self.gbdt_model.save(gbdt_path)
        if self.deep_model is not None:
            deep_path = os.path.join(self.model_dir, "renewable_deep.pt")
            self.deep_model.save(deep_path)

    def load(self):
        gbdt_path = os.path.join(self.model_dir, "renewable_gbdt.pkl")
        if os.path.exists(gbdt_path):
            self.gbdt_model.load(gbdt_path)
        deep_path = os.path.join(self.model_dir, "renewable_deep.pt")
        if os.path.exists(deep_path):
            self.deep_model = DeepQuantileModel(quantiles=self.quantiles, name="renewable_deep")
            self.deep_model.load(deep_path)
