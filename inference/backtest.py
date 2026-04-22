import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from inference.predictor import PipelinePredictor
from evaluation.evaluator import ScenarioEvaluator
from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self, model_dir: str = "./saved_models"):
        self.predictor = PipelinePredictor(model_dir=model_dir)
        self.evaluator = ScenarioEvaluator()

    def run_backtest(self, df: pd.DataFrame,
                     start_date: str, end_date: str,
                     target_cols: list = None) -> Dict[str, dict]:
        logger.info("=" * 60)
        logger.info(f"BACKTEST: {start_date} ~ {end_date}")
        logger.info("=" * 60)

        self.predictor.load_models()

        if target_cols is None:
            target_cols = ["load_actual", "price_DA", "price_RT", "spread"]

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        all_predictions = {target: {"P10": [], "P50": [], "P90": []} for target in target_cols}
        all_actuals = {target: [] for target in target_cols}
        all_dates = []

        current = start
        while current <= end:
            day_mask = df.index.date == current.date()
            if day_mask.sum() == 0:
                current += pd.Timedelta(days=1)
                continue

            day_results = self.predictor.predict_day(df, str(current.date()))

            for target in target_cols:
                if target in day_results:
                    pred = day_results[target]
                    for label in ["P10", "P50", "P90"]:
                        if label in pred:
                            all_predictions[target][label].extend(pred[label].tolist())

                    actual_col = target
                    if actual_col in df.columns:
                        actual_vals = df.loc[day_mask, actual_col].values
                        all_actuals[target].extend(actual_vals.tolist())

            all_dates.append(current.date())
            current += pd.Timedelta(days=1)

        results = {}
        for target in target_cols:
            if all_actuals[target] and all_predictions[target]["P50"]:
                y_true = np.array(all_actuals[target])
                pred = {k: np.array(v[:len(y_true)]) for k, v in all_predictions[target].items()}

                valid = ~np.isnan(y_true)
                if valid.sum() > 0:
                    metrics = compute_all_metrics(y_true[valid], {k: v[valid] for k, v in pred.items()})
                    results[target] = {"metrics": metrics, "predictions": pred, "actuals": y_true}

                    logger.info(f"  {target}: MAE={metrics.get('MAE', 0):.4f}, "
                                f"pinball_P50={metrics.get('pinball_P50', 0):.4f}")

        return results

    def run_missing_backtest(self, df: pd.DataFrame,
                              start_date: str, end_date: str,
                              missing_days: int = 5) -> Dict[str, dict]:
        logger.info("=" * 60)
        logger.info(f"MISSING BACKTEST: {start_date} ~ {end_date} (missing={missing_days} days)")
        logger.info("=" * 60)

        from models.missing_simulation import MissingSimulator
        simulator = MissingSimulator()

        df_sim, _ = simulator.create_scenario(df, f"{missing_days}day_missing")

        return self.run_backtest(df_sim, start_date, end_date)

    def export_predictions(self, results: Dict[str, dict], output_path: str):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        rows = []
        for target, data in results.items():
            if "predictions" in data and "actuals" in data:
                preds = data["predictions"]
                actuals = data["actuals"]
                n = len(actuals)
                for i in range(n):
                    row = {"target": target, "index": i, "actual": actuals[i]}
                    for label in ["P10", "P50", "P90"]:
                        if label in preds and i < len(preds[label]):
                            row[label] = preds[label][i]
                    rows.append(row)

        df_out = pd.DataFrame(rows)
        df_out.to_parquet(output_path, index=False)
        logger.info(f"  Predictions exported to {output_path}")
