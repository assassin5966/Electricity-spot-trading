import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("./logs/train.log", mode="a")],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Power Spot Market Model Training")
    parser.add_argument("--data", type=str, default="./data/output/feature_master_table.parquet")
    parser.add_argument("--model-dir", type=str, default="./saved_models")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    logger.info("Loading data from %s", args.data)
    df = pd.read_parquet(args.data)
    logger.info(f"Data shape: {df.shape}, range: {df.index.min()} ~ {df.index.max()}")

    from .trainer import PipelineTrainer
    trainer = PipelineTrainer(model_dir=args.model_dir)
    trainer.train_all(df)

    if args.evaluate:
        from evaluation.evaluator import ScenarioEvaluator
        evaluator = ScenarioEvaluator()
        from config import TARGET_COLUMNS, SPLIT_DATES, TIMEZONE

        valid_start = pd.Timestamp(SPLIT_DATES["valid"][0], tz=TIMEZONE)
        valid_end = pd.Timestamp(SPLIT_DATES["valid"][1], tz=TIMEZONE) + pd.Timedelta(hours=23, minutes=45)
        valid_mask = (df.index >= valid_start) & (df.index <= valid_end)
        valid_df = df[valid_mask]

        for target in TARGET_COLUMNS:
            if target not in valid_df.columns or target not in trainer.gbdt_models:
                continue
            y_valid = valid_df[target].dropna()
            if len(y_valid) < 10:
                continue

            pred = trainer.predict(df, target, X=valid_df.loc[y_valid.index])
            metrics = evaluator.evaluate(valid_df.loc[y_valid.index], {target: pred}, [target])
            report = evaluator.generate_report()
            logger.info(f"\n{report}")


if __name__ == "__main__":
    main()
