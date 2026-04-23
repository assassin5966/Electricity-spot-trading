import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import MODEL_SAVE_DIR, POINTS_PER_DAY, TARGET_COLUMNS, TIMEZONE
from download_data import run_download
from features import run_feature_pipeline
from models.base import GBDTQuantileModel
from strategy.strategy_engine import StrategyEngine, StrategyInput


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class DailyPrediction:
    date: str
    quantiles: Dict[str, Dict[str, np.ndarray]]


class AutoRunner:
    def __init__(self, model_dir: str = MODEL_SAVE_DIR):
        self.model_dir = model_dir
        self.models: Dict[str, GBDTQuantileModel] = {}
        self.feature_cols: Dict[str, List[str]] = {}
        self.strategy_engine = StrategyEngine()

    def load_models(self) -> None:
        for target in TARGET_COLUMNS:
            model_path = os.path.join(self.model_dir, target, f"gbdt_{target}.pkl")
            if not os.path.exists(model_path):
                logger.warning("Model missing for target=%s: %s", target, model_path)
                continue
            model = GBDTQuantileModel(name=f"gbdt_{target}")
            model.load(model_path)
            self.models[target] = model
            self.feature_cols[target] = list(getattr(model, "feature_cols", []) or [])
        if not self.models:
            raise FileNotFoundError("No trained model found under saved_models")

    def get_latest_features(self, skip_download: bool = False) -> pd.DataFrame:
        main_df = weather_df = None
        if not skip_download:
            try:
                main_df, weather_df = run_download()
            except Exception as exc:
                logger.warning("Download failed, fallback to local raw parquet: %s", exc)
        df = run_feature_pipeline(main_df=main_df, weather_df=weather_df)
        return df

    def _ensure_day_frame(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        day_mask = df.index.date == target_date.date()
        if day_mask.sum() == POINTS_PER_DAY:
            return df, df.loc[day_mask].copy()

        logger.warning("Date %s does not have %s points, creating synthetic day frame", target_date.date(), POINTS_PER_DAY)
        if df.empty:
            raise ValueError("Feature dataframe is empty")

        last_ts = df.index.max()
        start_ts = target_date.tz_localize(TIMEZONE)
        day_index = pd.date_range(start=start_ts, periods=POINTS_PER_DAY, freq="15min", tz=TIMEZONE)

        template = df.loc[[last_ts]].copy()
        day_df = pd.concat([template] * POINTS_PER_DAY, ignore_index=True)
        day_df.index = day_index

        day_df["quarter_index"] = np.arange(POINTS_PER_DAY)
        day_df["hour"] = day_df.index.hour
        day_df["weekday"] = day_df.index.weekday
        day_df["is_weekend"] = (day_df["weekday"] >= 5).astype(int)
        day_df["month"] = day_df.index.month
        day_df["day_of_month"] = day_df.index.day
        day_df["day_of_year"] = day_df.index.day_of_year
        day_df["sin_hour"] = np.sin(2 * np.pi * day_df["quarter_index"] / POINTS_PER_DAY)
        day_df["cos_hour"] = np.cos(2 * np.pi * day_df["quarter_index"] / POINTS_PER_DAY)
        day_df["sin_doy"] = np.sin(2 * np.pi * day_df["day_of_year"] / 365)
        day_df["cos_doy"] = np.cos(2 * np.pi * day_df["day_of_year"] / 365)

        df = pd.concat([df, day_df], axis=0).sort_index()
        return df, day_df

    def _fill_dayago_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle cases where D-2~D-6 are missing: use stable fallback chain."""
        df = df.copy()
        candidates = [c for c in df.columns if c.endswith("_DAYAGO")]
        for col in candidates:
            lag_7d_col = f"{col}_lag_7d"
            roll_col = f"{col}_rolling_mean_7d"

            if lag_7d_col in df.columns:
                df[col] = df[col].fillna(df[lag_7d_col])
            if roll_col in df.columns:
                df[col] = df[col].fillna(df[roll_col])

            # 最后兜底：按日内时点前向填充，再整体前向填充
            df[col] = df.groupby(df.index.time)[col].transform(lambda s: s.ffill().bfill())
            df[col] = df[col].ffill().bfill()
        return df

    def _predict_for_day(self, day_df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        results: Dict[str, Dict[str, np.ndarray]] = {}
        for target, model in self.models.items():
            feat_cols = self.feature_cols.get(target, [])
            X = day_df.copy()
            for col in feat_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[feat_cols].fillna(0) if feat_cols else X.select_dtypes(include=[np.number]).fillna(0)
            results[target] = model.predict(X)
        return results

    @staticmethod
    def _build_strategy_input(pred: Dict[str, Dict[str, np.ndarray]], day_df: pd.DataFrame) -> StrategyInput:
        load = pred.get("LOAD_REAL", {})
        pda = pred.get("PRICE_DAYAGO", {})
        prt = pred.get("PRICE_REAL", {})
        spread = pred.get("PRICE_R_D", {})

        def q(source: Dict[str, np.ndarray], name: str, fallback: float = 0.0) -> np.ndarray:
            arr = source.get(name)
            if arr is None:
                return np.full(POINTS_PER_DAY, fallback)
            return np.asarray(arr)[:POINTS_PER_DAY]

        return StrategyInput(
            load_pred_p10=q(load, "P10"),
            load_pred_p50=q(load, "P50"),
            load_pred_p90=q(load, "P90"),
            price_da_pred_p10=q(pda, "P10"),
            price_da_pred_p50=q(pda, "P50"),
            price_da_pred_p90=q(pda, "P90"),
            price_rt_pred_p10=q(prt, "P10"),
            price_rt_pred_p50=q(prt, "P50"),
            price_rt_pred_p90=q(prt, "P90"),
            spread_pred_p10=q(spread, "P10"),
            spread_pred_p50=q(spread, "P50"),
            spread_pred_p90=q(spread, "P90"),
            is_holiday=bool(day_df.get("is_holiday", pd.Series([0])).max()),
            is_post_holiday=bool(day_df.get("is_post_holiday", pd.Series([0])).max()),
            mask_flag=int(day_df.get("mask_flag", pd.Series([0])).max()),
            missing_length=int(day_df.get("missing_length", pd.Series([0])).max()),
            contract_curve=q(load, "P50"),
        )

    def run(self, declare_date: str, n_days: int, output_dir: str, skip_download: bool = False) -> Dict[str, DailyPrediction]:
        os.makedirs(output_dir, exist_ok=True)
        self.load_models()

        feature_df = self.get_latest_features(skip_download=skip_download)
        feature_df = self._fill_dayago_gaps(feature_df)

        start = pd.Timestamp(declare_date)
        all_results: Dict[str, DailyPrediction] = {}

        for step in range(n_days):
            target_date = start + pd.Timedelta(days=step)
            feature_df, day_df = self._ensure_day_frame(feature_df, target_date)
            day_df = self._fill_dayago_gaps(day_df)

            pred = self._predict_for_day(day_df)
            all_results[str(target_date.date())] = DailyPrediction(date=str(target_date.date()), quantiles=pred)

            strategy_inp = self._build_strategy_input(pred, day_df)
            strategy_out = self.strategy_engine.generate_daily_strategy(strategy_inp)

            payload = {
                "declare_date": str(target_date.date()),
                "data": [float(np.round(v, 6)) for v in strategy_out.q_final[:POINTS_PER_DAY]],
            }
            out_path = os.path.join(output_dir, f"strategy_{target_date.date()}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info("Saved strategy json: %s", out_path)

            # 迭代预测：将当天预测中位数回灌，供下一天 lag/dayago 使用
            for tgt in TARGET_COLUMNS:
                if tgt in pred and "P50" in pred[tgt]:
                    feature_df.loc[day_df.index, tgt] = pred[tgt]["P50"][: len(day_df)]

            feature_df = self._fill_dayago_gaps(feature_df)

        return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="自动化电力现货预测+策略执行")
    parser.add_argument("--declare-date", type=str, required=True, help="申报日期 YYYY-MM-DD")
    parser.add_argument("--n-days", type=int, default=1, help="迭代预测天数")
    parser.add_argument("--output-dir", type=str, default="./data/output/strategy_json")
    parser.add_argument("--skip-download", action="store_true", help="跳过数据库下载，使用本地原始数据")
    parser.add_argument("--model-dir", type=str, default=MODEL_SAVE_DIR)
    args = parser.parse_args()

    runner = AutoRunner(model_dir=args.model_dir)
    runner.run(
        declare_date=args.declare_date,
        n_days=args.n_days,
        output_dir=args.output_dir,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
