import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

from models.base import GBDTQuantileModel
from models.moe_router import HolidayExpertEnsemble, MoERouter
from models.ensemble import ModelEnsemble, SimpleBaseline
from training.weights import SampleWeightCalculator
from config import TARGET_COLUMNS, SPLIT_DATES, TIMEZONE, MODEL_SAVE_DIR

logger = logging.getLogger(__name__)


class PipelineTrainer:
    def __init__(self, model_dir: str = MODEL_SAVE_DIR):
        self.model_dir = model_dir
        self.weight_calc = SampleWeightCalculator()
        self.moe_router = MoERouter()

        self.gbdt_models = {}
        self.moe_models = {}
        self.baseline = SimpleBaseline()
        self.ensemble = ModelEnsemble()
        self.oof_predictions = {}
        self.feature_cols_map = {}

    def _get_base_feature_cols(self, df: pd.DataFrame, target: str) -> list:
        exclude = set(TARGET_COLUMNS)
        exclude.add(target)
        return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

    def _time_split(self, df: pd.DataFrame, split_name: str):
        start, end = SPLIT_DATES[split_name]
        start_ts = pd.Timestamp(start, tz=TIMEZONE)
        end_ts = pd.Timestamp(end, tz=TIMEZONE) + pd.Timedelta(hours=23, minutes=45)
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        return df[mask]

    def _generate_oof_predictions(self, df: pd.DataFrame, target: str,
                                    n_folds: int = 5) -> pd.Series:
        logger.info(f"    Generating OOF predictions for {target} ...")
        train_df = self._time_split(df, "train")
        n = len(train_df)
        fold_size = n // n_folds
        oof_pred = pd.Series(np.nan, index=train_df.index)

        feature_cols = self._get_base_feature_cols(df, target)
        y = train_df[target]

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = min((fold + 1) * fold_size, n)

            train_idx = list(range(0, val_start)) + list(range(val_end, n))
            val_idx = list(range(val_start, val_end))

            X_train = train_df[feature_cols].iloc[train_idx].fillna(0)
            y_train = y.iloc[train_idx]
            X_val = train_df[feature_cols].iloc[val_idx].fillna(0)

            model = GBDTQuantileModel(name=f"oof_{target}_fold{fold}")
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            oof_pred.iloc[val_idx] = pred["P50"]

        valid = oof_pred.notna()
        logger.info(f"    OOF coverage: {valid.sum()}/{n}")
        return oof_pred

    def train_target_gbdt(self, df: pd.DataFrame, target: str):
        logger.info(f"  Training GBDT for target: {target}")
        train_df = self._time_split(df, "train")
        valid_df = self._time_split(df, "valid")

        feature_cols = self._get_base_feature_cols(df, target)
        oof_cols = []

        for upstream_target in TARGET_COLUMNS:
            if upstream_target != target and upstream_target in TARGET_COLUMNS:
                oof_col = f"{upstream_target}_oof_pred"
                if oof_col not in train_df.columns and upstream_target in df.columns:
                    oof = self._generate_oof_predictions(df, upstream_target)
                    train_df = train_df.copy()
                    train_df[oof_col] = oof.reindex(train_df.index)
                    oof_cols.append(oof_col)
                    feature_cols.append(oof_col)

        valid_df = valid_df.copy()
        base_feature_cols = [c for c in feature_cols if c not in oof_cols]
        for oof_col in oof_cols:
            upstream_target = oof_col.replace("_oof_pred", "")
            upstream_model = self.gbdt_models.get(upstream_target)
            if upstream_model is not None and upstream_target in valid_df.columns:
                try:
                    upstream_pred = upstream_model.predict(valid_df[base_feature_cols].fillna(0))
                    valid_df[oof_col] = upstream_pred["P50"]
                except Exception:
                    valid_df[oof_col] = 0.0
            else:
                valid_df[oof_col] = 0.0

        self.feature_cols_map[target] = feature_cols

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target].dropna()
        X_train = X_train.loc[y_train.index]

        sample_weight = self.weight_calc.compute_for_training(
            train_df.loc[y_train.index], [target]
        )

        eval_set = None
        if target in valid_df.columns:
            y_valid = valid_df[target].dropna()
            if len(y_valid) > 0:
                X_valid = valid_df[feature_cols].fillna(0).loc[y_valid.index]
                eval_set = (X_valid, y_valid)

        model = GBDTQuantileModel(name=f"gbdt_{target}")
        model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=eval_set)
        self.gbdt_models[target] = model

        model.save(os.path.join(self.model_dir, target, f"gbdt_{target}.pkl"), feature_cols=feature_cols)
        logger.info(f"    GBDT for {target} saved, features={len(feature_cols)}")

    def train_target_moe(self, df: pd.DataFrame, target: str):
        logger.info(f"  Training MoE for target: {target}")
        train_df = self._time_split(df, "train")

        feature_cols = self._get_base_feature_cols(df, target)
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target].dropna()
        X_train = X_train.loc[y_train.index]

        sample_weight = self.weight_calc.compute_for_training(
            train_df.loc[y_train.index], [target]
        )

        moe = HolidayExpertEnsemble()
        moe.train_global(X_train, y_train, sample_weight=sample_weight)
        moe.train_residual_experts(X_train, y_train, sample_weight=sample_weight)

        moe.save(os.path.join(self.model_dir, target, "moe"))
        self.moe_models[target] = moe
        logger.info(f"    MoE for {target} saved")

    def train_all(self, df: pd.DataFrame):
        logger.info("=" * 60)
        logger.info("FULL PIPELINE TRAINING")
        logger.info("=" * 60)

        os.makedirs(self.model_dir, exist_ok=True)

        self.baseline.fit(df, TARGET_COLUMNS)

        for target in TARGET_COLUMNS:
            if target not in df.columns:
                logger.warning(f"  Target '{target}' not in dataframe, skipping")
                continue

            non_null = df[target].notna().sum()
            if non_null < 100:
                logger.warning(f"  Target '{target}' has only {non_null} non-null values, skipping")
                continue

            self.train_target_gbdt(df, target)
            self.train_target_moe(df, target)

        self._build_ensemble(df)
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)

    def _build_ensemble(self, df: pd.DataFrame):
        logger.info("  Building ensemble ...")
        valid_df = self._time_split(df, "valid")

        for target in TARGET_COLUMNS:
            if target not in valid_df.columns or target not in self.gbdt_models:
                continue

            y_valid = valid_df[target].dropna()
            if len(y_valid) < 10:
                continue

            ensemble = ModelEnsemble()

            feature_cols = self.feature_cols_map.get(target, self._get_base_feature_cols(df, target))

            for col in feature_cols:
                if col not in valid_df.columns:
                    valid_df = valid_df.copy()
                    valid_df[col] = 0.0

            X_valid = valid_df[feature_cols].fillna(0).loc[y_valid.index]

            gbdt_pred_model = self.gbdt_models[target]
            ensemble.add_model("gbdt_model", gbdt_pred_model, weight=0.5)

            baseline_wrapper = _BaselineWrapper(self.baseline, target)
            ensemble.add_model("simple_baseline", baseline_wrapper, weight=0.2)

            ensemble.optimize_weights(X_valid, y_valid, n_grid=11)

            if "is_holiday" in valid_df.columns:
                expert_labels = self.moe_router.get_expert_label(valid_df.loc[y_valid.index])
                ensemble.optimize_scenario_weights(X_valid, y_valid, expert_labels)

            self.ensemble.models[target] = ensemble
            logger.info(f"    Ensemble for {target}: weights={ensemble.weights}")

    def predict(self, df: pd.DataFrame, target: str, X: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        if target not in self.gbdt_models:
            raise ValueError(f"No model for target '{target}'")

        feature_cols = self.feature_cols_map.get(target, self._get_base_feature_cols(df, target))

        if X is None:
            X = df.copy()

        for col in feature_cols:
            if col not in X.columns:
                X = X.copy()
                X[col] = 0.0

        X = X[feature_cols].fillna(0)
        return self.gbdt_models[target].predict(X)


class _BaselineWrapper:
    def __init__(self, baseline: SimpleBaseline, target: str):
        self.baseline = baseline
        self.target = target

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        return self.baseline.predict(X, self.target)
