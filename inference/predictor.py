import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

from models.renewable_model import RenewableModel
from models.load_model import LoadModel
from models.price_model import PriceModel
from models.moe_router import MoERouter
from models.ensemble import ModelEnsemble, SimpleBaseline
from models.missing_simulation import MissingSimulator
from config import MODEL_SAVE_DIR

logger = logging.getLogger(__name__)


class PipelinePredictor:
    def __init__(self, model_dir: str = MODEL_SAVE_DIR):
        self.model_dir = model_dir
        self.moe_router = MoERouter()
        self.missing_simulator = MissingSimulator()

        self.renewable_model = RenewableModel(model_dir=os.path.join(model_dir, "renewable"))
        self.load_model = LoadModel(model_dir=os.path.join(model_dir, "load"))
        self.price_model = PriceModel(model_dir=os.path.join(model_dir, "price"))
        self.baseline_model = SimpleBaseline()
        self.ensemble = ModelEnsemble()

        self._loaded = False

    def load_models(self):
        logger.info("Loading all models ...")
        try:
            self.renewable_model.load()
        except Exception as e:
            logger.warning(f"  Failed to load renewable model: {e}")
        try:
            self.load_model.load()
        except Exception as e:
            logger.warning(f"  Failed to load load model: {e}")
        try:
            self.price_model.load()
        except Exception as e:
            logger.warning(f"  Failed to load price model: {e}")
        self._loaded = True
        logger.info("  Models loaded")

    def predict_day(self, df: pd.DataFrame, target_date: str,
                    expert_type: Optional[str] = None) -> Dict[str, dict]:
        if not self._loaded:
            self.load_models()

        logger.info(f"Predicting for {target_date} ...")

        target_date = pd.Timestamp(target_date)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            target_date = target_date.tz_localize(df.index.tz)

        day_mask = df.index.date == target_date.date()
        if day_mask.sum() == 0:
            logger.error(f"  No data for {target_date}")
            return {}

        df_day = df[day_mask].copy()

        df_day = self._add_post_holiday(df_day)
        df_day = self._add_missing_features(df_day)

        results = {}

        try:
            renewable_pred = self.renewable_model.predict(df_day, use_model="gbdt")
            results["renewable"] = {
                "wind_power": renewable_pred,
                "solar_power": renewable_pred,
                "renewable_total": renewable_pred,
            }
        except Exception as e:
            logger.warning(f"  Renewable prediction failed: {e}")
            renewable_pred = None

        try:
            load_pred = self.load_model.predict(df_day, expert_type=expert_type,
                                                 renewable_pred=renewable_pred)
            results["load_actual"] = load_pred
        except Exception as e:
            logger.warning(f"  Load prediction failed: {e}")
            load_pred = None

        try:
            missing_mask = {"mask_flag": df_day["mask_flag"].values,
                            "missing_length": df_day["missing_length"].values} if "mask_flag" in df_day.columns else None
            price_pred = self.price_model.predict(df_day, load_pred=load_pred,
                                                   renewable_pred=renewable_pred,
                                                   missing_mask=missing_mask,
                                                   use_model="gbdt")
            results.update(price_pred)
        except Exception as e:
            logger.warning(f"  Price prediction failed: {e}")

        return results

    def predict_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> Dict[str, dict]:
        if not self._loaded:
            self.load_models()

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        all_results = {}
        current = start
        while current <= end:
            day_results = self.predict_day(df, str(current.date()))
            all_results[str(current.date())] = day_results
            current += pd.Timedelta(days=1)

        return all_results

    def predict_competition(self, df: pd.DataFrame) -> Dict[str, dict]:
        logger.info("=" * 60)
        logger.info("COMPETITION PREDICTION: May 1-10")
        logger.info("=" * 60)

        results = self.predict_range(df, "2026-05-01", "2026-05-10")

        for date, day_pred in results.items():
            logger.info(f"  {date}: {list(day_pred.keys())}")

        return results

    def _add_post_holiday(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "is_holiday" in df.columns:
            is_holiday = df["is_holiday"].astype(bool)
            shifted = is_holiday.shift(1).fillna(False)
            df["is_post_holiday"] = (shifted & ~is_holiday).astype(int)
        else:
            df["is_post_holiday"] = 0
        return df

    def _add_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        from config import MISSING_MASK_COLUMNS
        df["mask_flag"] = 0
        df["missing_length"] = 0

        for col in MISSING_MASK_COLUMNS:
            matching = [c for c in df.columns if col in c]
            for m in matching:
                is_missing = df[m].isna()
                df.loc[is_missing, "mask_flag"] = 1

        return df
