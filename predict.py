import argparse
import logging
import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("./logs/predict.log", mode="a")],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Power Spot Market Prediction")
    parser.add_argument("--data", type=str, default="./data/output/feature_master_table.parquet")
    parser.add_argument("--model-dir", type=str, default="./saved_models")
    parser.add_argument("--start-date", type=str, default="2026-05-01")
    parser.add_argument("--end-date", type=str, default="2026-05-10")
    parser.add_argument("--output", type=str, default="./data/output/predictions.parquet")
    parser.add_argument("--backtest", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    logger.info("Loading data from %s", args.data)
    df = pd.read_parquet(args.data)

    from config import TARGET_COLUMNS, TIMEZONE, POINTS_PER_DAY
    from models.base import GBDTQuantileModel
    from models.moe_router import HolidayExpertEnsemble, MoERouter
    from models.ensemble import SimpleBaseline

    start_ts = pd.Timestamp(args.start_date, tz=TIMEZONE)
    end_ts = pd.Timestamp(args.end_date, tz=TIMEZONE) + pd.Timedelta(hours=23, minutes=45)
    mask = (df.index >= start_ts) & (df.index <= end_ts)
    pred_df = df[mask]

    if len(pred_df) == 0:
        logger.error(f"No data for {args.start_date} ~ {args.end_date}")
        return

    rows = []
    for target in TARGET_COLUMNS:
        gbdt_path = os.path.join(args.model_dir, target, f"gbdt_{target}.pkl")
        if not os.path.exists(gbdt_path):
            logger.warning(f"Model for {target} not found")
            continue

        model = GBDTQuantileModel(name=f"gbdt_{target}")
        model.load(gbdt_path)

        feature_cols = model.feature_cols
        if feature_cols is None:
            from training.trainer import PipelineTrainer
            trainer = PipelineTrainer(model_dir=args.model_dir)
            feature_cols = trainer._get_base_feature_cols(df, target)

        for col in feature_cols:
            if col not in pred_df.columns:
                pred_df = pred_df.copy()
                pred_df[col] = 0.0
        X = pred_df[feature_cols].fillna(0)

        pred = model.predict(X)

        for i in range(len(pred_df)):
            row = {
                "datetime": pred_df.index[i],
                "target": target,
            }
            for label in ["P10", "P50", "P90"]:
                row[label] = pred[label][i]
            if target in pred_df.columns:
                row["actual"] = pred_df[target].iloc[i]
            rows.append(row)

    if rows:
        df_pred = pd.DataFrame(rows)
        df_pred.to_parquet(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}, {len(df_pred)} rows")
    else:
        logger.warning("No predictions generated")


if __name__ == "__main__":
    main()
