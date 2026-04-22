import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from models.base import GBDTQuantileModel, DeepQuantileModel

logger = logging.getLogger(__name__)


class PriceModel:
    def __init__(self, quantiles=None, model_dir: str = "./saved_models/price"):
        from config import QUANTILES, PRICE_LOSS_WEIGHTS
        self.quantiles = quantiles or QUANTILES
        self.model_dir = model_dir
        self.loss_weights = PRICE_LOSS_WEIGHTS
        self.gbdt_model = None
        self.deep_model = None
        self.spike_classifier = None
        self.feature_columns = []

    def _build_features(self, df: pd.DataFrame,
                        load_pred: Optional[Dict] = None,
                        renewable_pred: Optional[Dict] = None,
                        missing_mask: Optional[Dict] = None) -> pd.DataFrame:
        from config import PRICE_FEATURES
        features = []

        if load_pred is not None:
            for label in ["P10", "P50", "P90"]:
                col = f"load_prediction_{label}"
                if label in load_pred:
                    df = df.copy()
                    df[col] = load_pred[label][:len(df)] if len(load_pred[label]) >= len(df) else np.nan
                    features.append(col)

        if renewable_pred is not None:
            for label in ["P10", "P50", "P90"]:
                col = f"renewable_prediction_{label}"
                if label in renewable_pred:
                    df = df.copy()
                    df[col] = renewable_pred[label][:len(df)] if len(renewable_pred[label]) >= len(df) else np.nan
                    features.append(col)

        if load_pred is not None and renewable_pred is not None:
            for label in ["P10", "P50", "P90"]:
                l_col = f"load_prediction_{label}"
                r_col = f"renewable_prediction_{label}"
                if l_col in df.columns and r_col in df.columns:
                    net_col = f"net_load_prediction_{label}"
                    df[net_col] = df[l_col] - df[r_col]
                    features.append(net_col)

        for col in PRICE_FEATURES["historical_price"]:
            if col in df.columns:
                features.append(col)

        for col in PRICE_FEATURES["calendar"]:
            if col in df.columns:
                features.append(col)

        if missing_mask is not None:
            for key in ["mask_flag", "missing_length"]:
                if key in missing_mask:
                    df = df.copy()
                    df[key] = missing_mask[key][:len(df)] if len(missing_mask[key]) >= len(df) else 0
                    features.append(key)

        self.feature_columns = features
        return df[features].fillna(0)

    def fit_gbdt(self, X: pd.DataFrame, y: pd.DataFrame,
                 sample_weight: Optional[np.ndarray] = None,
                 eval_set: Optional[tuple] = None,
                 load_pred: Optional[Dict] = None,
                 renewable_pred: Optional[Dict] = None,
                 missing_mask: Optional[Dict] = None):
        logger.info("  [PriceModel] Training GBDT models for DA/RT/spread ...")
        self.gbdt_model = {}

        for target in ["price_DA", "price_RT", "spread"]:
            if target not in y.columns:
                logger.warning(f"    Target '{target}' not in y, skipping")
                continue

            X_feat = self._build_features(X, load_pred, renewable_pred, missing_mask)
            model = GBDTQuantileModel(quantiles=self.quantiles, name=f"price_gbdt_{target}")
            model.fit(X_feat, y[target], sample_weight=sample_weight,
                      eval_set=eval_set)
            self.gbdt_model[target] = model
            logger.info(f"    GBDT trained for {target}")

    def fit_spike_classifier(self, X: pd.DataFrame, spike_labels: pd.Series,
                             load_pred: Optional[Dict] = None,
                             renewable_pred: Optional[Dict] = None,
                             missing_mask: Optional[Dict] = None):
        logger.info("  [PriceModel] Training spike classifier ...")
        import lightgbm as lgb

        X_feat = self._build_features(X, load_pred, renewable_pred, missing_mask)

        self.spike_classifier = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            verbose=-1,
        )
        self.spike_classifier.fit(X_feat, spike_labels)
        logger.info(f"    Spike classifier trained, positive rate: {spike_labels.mean():.4f}")

    def fit_deep(self, X: np.ndarray, y: np.ndarray,
                 input_dim: int = None,
                 eval_set: Optional[tuple] = None,
                 **kwargs):
        logger.info("  [PriceModel] Training Deep model (TFT) ...")
        if input_dim is None:
            input_dim = X.shape[-1] if X.ndim == 3 else X.shape[1]

        self.deep_model = DeepQuantileModel(
            quantiles=self.quantiles,
            name="price_deep",
            input_dim=input_dim,
        )
        self.deep_model.fit(X, y, eval_set=eval_set, **kwargs)
        return self

    def predict(self, X: pd.DataFrame,
                load_pred: Optional[Dict] = None,
                renewable_pred: Optional[Dict] = None,
                missing_mask: Optional[Dict] = None,
                use_model: str = "gbdt") -> Dict[str, Dict[str, np.ndarray]]:
        results = {}

        if use_model == "gbdt" and self.gbdt_model is not None:
            X_feat = self._build_features(X, load_pred, renewable_pred, missing_mask)
            for target, model in self.gbdt_model.items():
                results[target] = model.predict(X_feat)

        elif use_model == "deep" and self.deep_model is not None:
            deep_pred = self.deep_model.predict(X)
            for target in ["price_DA", "price_RT", "spread"]:
                results[target] = deep_pred

        if self.spike_classifier is not None:
            X_feat = self._build_features(X, load_pred, renewable_pred, missing_mask)
            spike_prob = self.spike_classifier.predict_proba(X_feat)[:, 1]
            results["spike_prob"] = spike_prob

        return results

    def predict_ensemble(self, X_gbdt: pd.DataFrame, X_deep: np.ndarray,
                         load_pred: Optional[Dict] = None,
                         renewable_pred: Optional[Dict] = None,
                         missing_mask: Optional[Dict] = None,
                         gbdt_weight: float = 0.4, deep_weight: float = 0.6) -> Dict[str, Dict[str, np.ndarray]]:
        gbdt_results = self.predict(X_gbdt, load_pred, renewable_pred, missing_mask, use_model="gbdt")
        if self.deep_model is not None:
            deep_results = self.predict(X_deep, use_model="deep")
            results = {}
            for target in gbdt_results:
                if target in deep_results and target != "spike_prob":
                    results[target] = {}
                    for label in gbdt_results[target]:
                        results[target][label] = (
                            gbdt_weight * gbdt_results[target][label]
                            + deep_weight * deep_results[target][label]
                        )
                else:
                    results[target] = gbdt_results[target]
            return results
        return gbdt_results

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        if self.gbdt_model is not None:
            for target, model in self.gbdt_model.items():
                model.save(os.path.join(self.model_dir, f"price_gbdt_{target}.pkl"))
        if self.deep_model is not None:
            self.deep_model.save(os.path.join(self.model_dir, "price_deep.pt"))
        if self.spike_classifier is not None:
            import pickle
            with open(os.path.join(self.model_dir, "spike_classifier.pkl"), "wb") as f:
                pickle.dump(self.spike_classifier, f)

    def load(self):
        import pickle
        for target in ["price_DA", "price_RT", "spread"]:
            path = os.path.join(self.model_dir, f"price_gbdt_{target}.pkl")
            if os.path.exists(path):
                if self.gbdt_model is None:
                    self.gbdt_model = {}
                model = GBDTQuantileModel(quantiles=self.quantiles, name=f"price_gbdt_{target}")
                model.load(path)
                self.gbdt_model[target] = model

        deep_path = os.path.join(self.model_dir, "price_deep.pt")
        if os.path.exists(deep_path):
            self.deep_model = DeepQuantileModel(quantiles=self.quantiles, name="price_deep")
            self.deep_model.load(deep_path)

        spike_path = os.path.join(self.model_dir, "spike_classifier.pkl")
        if os.path.exists(spike_path):
            with open(spike_path, "rb") as f:
                self.spike_classifier = pickle.load(f)
