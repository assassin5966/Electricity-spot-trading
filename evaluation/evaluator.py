import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from evaluation.metrics import compute_all_metrics, spread_direction_accuracy, spike_hit_rate

logger = logging.getLogger(__name__)


class ScenarioEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate(self, df: pd.DataFrame, predictions: Dict[str, dict],
                 target_cols: list = None) -> dict:
        if target_cols is None:
            target_cols = list(predictions.keys())

        all_metrics = {}

        logger.info("  Evaluating ALL data ...")
        for target in target_cols:
            if target in predictions and target in df.columns:
                y_true = df[target].values
                pred = predictions[target]
                valid = ~np.isnan(y_true)
                if valid.sum() > 0:
                    metrics = compute_all_metrics(y_true[valid], {k: v[valid] for k, v in pred.items()})
                    all_metrics[f"all_{target}"] = metrics
                    logger.info(f"    {target}: MAE={metrics.get('MAE', 0):.4f}, "
                                f"pinball_P50={metrics.get('pinball_P50', 0):.4f}")

        for scenario in ["normal", "holiday", "post_holiday"]:
            mask = self._get_scenario_mask(df, scenario)
            if mask.sum() == 0:
                continue

            logger.info(f"  Evaluating scenario: {scenario} ({mask.sum()} samples) ...")
            for target in target_cols:
                if target in predictions and target in df.columns:
                    y_true = df.loc[mask, target].values
                    pred = predictions[target]
                    if isinstance(pred, dict):
                        pred_scenario = {k: v[mask] for k, v in pred.items()}
                    else:
                        pred_scenario = pred[mask]

                    valid = ~np.isnan(y_true)
                    if valid.sum() > 0:
                        metrics = compute_all_metrics(y_true[valid],
                                                      {k: v[valid] for k, v in pred_scenario.items()})
                        all_metrics[f"{scenario}_{target}"] = metrics
                        logger.info(f"    {target}: MAE={metrics.get('MAE', 0):.4f}")

        if "spread" in predictions and "spread" in df.columns:
            y_spread = df["spread"].values
            pred_spread = predictions["spread"].get("P50", None)
            if pred_spread is not None:
                valid = ~np.isnan(y_spread)
                if valid.sum() > 0:
                    acc = spread_direction_accuracy(y_spread[valid], pred_spread[valid])
                    all_metrics["spread_direction_accuracy"] = acc
                    logger.info(f"  Spread direction accuracy: {acc:.4f}")

        self.results = all_metrics
        return all_metrics

    def _get_scenario_mask(self, df: pd.DataFrame, scenario: str) -> pd.Series:
        if scenario == "normal":
            if "is_holiday" in df.columns and "is_post_holiday" in df.columns:
                return (df["is_holiday"] == 0) & (df["is_post_holiday"] == 0)
            return pd.Series(True, index=df.index)
        elif scenario == "holiday":
            if "is_holiday" in df.columns:
                return df["is_holiday"] == 1
            return pd.Series(False, index=df.index)
        elif scenario == "post_holiday":
            if "is_post_holiday" in df.columns:
                return df["is_post_holiday"] == 1
            return pd.Series(False, index=df.index)
        return pd.Series(False, index=df.index)

    def evaluate_missing_scenario(self, df: pd.DataFrame, predictions: Dict[str, dict],
                                  missing_length: int = 5) -> dict:
        from models.missing_simulation import MissingSimulator
        simulator = MissingSimulator()
        df_masked, mask_info = simulator.create_scenario(df, f"{missing_length}day_missing")

        metrics = {}
        for target, pred in predictions.items():
            if target in df.columns:
                y_true = df[target].values
                valid = ~np.isnan(y_true)
                if valid.sum() > 0:
                    metrics[target] = compute_all_metrics(y_true[valid],
                                                          {k: v[valid] for k, v in pred.items()})

        return metrics

    def compare_models(self, model_results: Dict[str, dict]) -> pd.DataFrame:
        rows = []
        for model_name, metrics in model_results.items():
            for metric_key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        rows.append({"model": model_name, "metric": f"{metric_key}_{sub_key}", "value": sub_value})
                else:
                    rows.append({"model": model_name, "metric": metric_key, "value": value})

        df = pd.DataFrame(rows)
        if len(df) > 0:
            pivot = df.pivot_table(index="metric", columns="model", values="value")
            logger.info("\n" + pivot.to_string())
        return df

    def generate_report(self) -> str:
        lines = ["=" * 60, "EVALUATION REPORT", "=" * 60]
        for key, value in sorted(self.results.items()):
            if isinstance(value, dict):
                lines.append(f"\n{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v:.6f}")
            else:
                lines.append(f"{key}: {value:.6f}")
        lines.append("=" * 60)
        return "\n".join(lines)
