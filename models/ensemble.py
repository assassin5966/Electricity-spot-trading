import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from itertools import product

logger = logging.getLogger(__name__)


class ModelEnsemble:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scenario_weights = {}

    def add_model(self, name: str, model, weight: float = 0.0):
        self.models[name] = model
        self.weights[name] = weight

    def optimize_weights(self, X: pd.DataFrame, y: pd.Series,
                         metric_fn=None, n_grid: int = 20) -> Dict[str, float]:
        logger.info("  [Ensemble] Optimizing weights on validation set ...")
        if metric_fn is None:
            from evaluation.metrics import mean_absolute_error
            metric_fn = mean_absolute_error

        all_preds = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                all_preds[name] = pred
            except Exception as e:
                logger.warning(f"    {name} prediction failed: {e}")

        if len(all_preds) < 2:
            logger.warning("    Need at least 2 models for weight optimization")
            return self.weights

        model_names = list(all_preds.keys())
        best_mae = float("inf")
        best_weights = {n: 1.0 / len(model_names) for n in model_names}

        for weights_tuple in product(np.linspace(0, 1, n_grid), repeat=len(model_names) - 1):
            w_last = 1.0 - sum(weights_tuple)
            if w_last < 0:
                continue
            w_list = list(weights_tuple) + [w_last]

            blended = np.zeros(len(y))
            for i, name in enumerate(model_names):
                blended += w_list[i] * all_preds[name]["P50"]

            mae = metric_fn(y.values, blended)
            if mae < best_mae:
                best_mae = mae
                best_weights = {name: w_list[i] for i, name in enumerate(model_names)}

        self.weights = best_weights
        logger.info(f"    Optimal weights: {best_weights}, MAE={best_mae:.4f}")
        return best_weights

    def optimize_scenario_weights(self, X: pd.DataFrame, y: pd.Series,
                                   expert_labels: pd.Series) -> dict:
        logger.info("  [Ensemble] Optimizing scenario-specific weights ...")
        self.scenario_weights = {}

        for scenario in ["normal_day", "holiday", "post_holiday"]:
            mask = expert_labels == scenario
            if mask.sum() < 10:
                continue
            X_sub = X[mask]
            y_sub = y[mask]
            weights = self.optimize_weights(X_sub, y_sub, n_grid=11)
            self.scenario_weights[scenario] = weights

        return self.scenario_weights

    def predict(self, X: pd.DataFrame, expert_labels: pd.Series = None) -> Dict[str, np.ndarray]:
        all_preds = {}
        for name, model in self.models.items():
            try:
                all_preds[name] = model.predict(X)
            except Exception:
                continue

        if not all_preds:
            raise RuntimeError("No valid model predictions")

        first_pred = list(all_preds.values())[0]
        results = {label: np.zeros(len(X)) for label in first_pred}

        if expert_labels is not None and self.scenario_weights:
            for scenario in self.scenario_weights:
                mask = expert_labels == scenario
                if mask.sum() == 0:
                    continue
                indices = np.where(mask)[0]
                w = self.scenario_weights[scenario]
                for label in results:
                    blended = np.zeros(len(indices))
                    total_w = sum(w.get(n, 0) for n in all_preds)
                    if total_w > 0:
                        for name, pred in all_preds.items():
                            blended += w.get(name, 0) / total_w * pred[label][indices]
                    results[label][indices] = blended
        else:
            total_w = sum(self.weights.get(n, 0) for n in all_preds)
            if total_w > 0:
                for name, pred in all_preds.items():
                    w = self.weights.get(name, 0) / total_w
                    for label in results:
                        results[label] = results[label] + w * pred[label]

        return results


class SimpleBaseline:
    def __init__(self, quantiles=None):
        from config import QUANTILES, POINTS_PER_DAY
        self.quantiles = quantiles or QUANTILES
        self.points_per_day = POINTS_PER_DAY
        self.historical_stats = {}

    def fit(self, df: pd.DataFrame, target_columns: list):
        for col in target_columns:
            if col not in df.columns:
                continue
            values = df[col].dropna()
            if len(values) == 0:
                continue

            if hasattr(df.index, "tz") and df.index.tz is not None:
                idx = df.index.tz_convert("Asia/Shanghai")
            else:
                idx = df.index

            quarter_index = pd.Series(idx.hour * 4 + idx.minute // 15, index=df.index)
            stat_df = pd.DataFrame({"val": values, "qi": quarter_index.loc[values.index]})

            stats = {}
            for qi in range(self.points_per_day):
                qi_vals = stat_df[stat_df["qi"] == qi]["val"]
                if len(qi_vals) > 0:
                    stats[qi] = {
                        "P10": qi_vals.quantile(0.1),
                        "P50": qi_vals.quantile(0.5),
                        "P90": qi_vals.quantile(0.9),
                    }
                else:
                    stats[qi] = {"P10": 0, "P50": 0, "P90": 0}

            self.historical_stats[col] = stats

        logger.info(f"  [Baseline] Fitted for {len(self.historical_stats)} targets, 96 slots each")

    def predict(self, X: pd.DataFrame, target: str) -> Dict[str, np.ndarray]:
        if target not in self.historical_stats:
            n = len(X)
            return {"P10": np.zeros(n), "P50": np.zeros(n), "P90": np.zeros(n)}

        stats = self.historical_stats[target]
        if hasattr(X.index, "tz") and X.index.tz is not None:
            idx = X.index.tz_convert("Asia/Shanghai")
        else:
            idx = X.index
        qi = idx.hour * 4 + idx.minute // 15

        results = {}
        for label in ["P10", "P50", "P90"]:
            results[label] = np.array([stats.get(q, {label: 0}).get(label, 0) for q in qi])

        return results
