import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    train_start: str = "2025-08-25"
    train_end: str = "2026-03-31"
    valid_start: str = "2026-04-01"
    valid_end: str = "2026-04-20"
    test_start: str = "2026-04-21"
    test_end: str = "2026-04-27"
    backtest_start: str = "2026-04-01"
    backtest_end: str = "2026-04-27"


class ModelTrainer:
    def __init__(self, model_dir: str = "./saved_models"):
        self.model_dir = model_dir
        self.models = {}
    
    def train_gbdt(self, df: pd.DataFrame, targets: List[str]) -> Dict[str, object]:
        from models.base import GBDTQuantileModel
        from config import TARGET_COLUMNS, SPLIT_DATES, TIMEZONE
        from training.trainer import PipelineTrainer
        
        logger.info("Training GBDT models...")
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        trainer = PipelineTrainer(model_dir=self.model_dir)
        trainer.train_all(df)
        
        for target in targets:
            if target in trainer.gbdt_models:
                self.models[f"gbdt_{target}"] = trainer.gbdt_models[target]
                logger.info(f"  {target}: GBDT trained")
        
        return self.models


class ModelFusion:
    def __init__(self, model_dir: str = "./saved_models"):
        self.model_dir = model_dir
        self.fusion_weights = {}
    
    def predict(self, X: pd.DataFrame, target: str) -> Dict[str, np.ndarray]:
        from models.base import GBDTQuantileModel
        from models.moe_router import HolidayExpertEnsemble
        from models.ensemble import ModelEnsemble
        
        preds = {}
        
        gbdt_path = os.path.join(self.model_dir, target, f"gbdt_{target}.pkl")
        if os.path.exists(gbdt_path):
            model = GBDTQuantileModel(name=f"gbdt_{target}")
            model.load(gbdt_path)
            feature_cols = model.feature_cols or []
            
            for col in feature_cols:
                if col not in X.columns:
                    X = X.copy()
                    X[col] = 0.0
            
            X_feat = X[feature_cols].fillna(0)
            preds["gbdt"] = model.predict(X_feat)
            logger.info(f"  {target}: GBDT predicted")
        
        moe_path = os.path.join(self.model_dir, target, "moe", "moe_global.pkl")
        if os.path.exists(moe_path):
            moe = HolidayExpertEnsemble()
            moe.load(os.path.join(self.model_dir, target, "moe"))
            feature_cols = list(preds.keys())[0] if preds else []
            if feature_cols and feature_cols in X.columns:
                X_feat = X[[feature_cols]].fillna(0) if isinstance(feature_cols, str) else X[feature_cols].fillna(0)
                try:
                    preds["moe"] = moe.predict(X_feat)
                    logger.info(f"  {target}: MoE predicted")
                except:
                    pass
        
        if not preds:
            return {"P10": np.zeros(len(X)), "P50": np.zeros(len(X)), "P90": np.zeros(len(X))}
        
        if len(preds) == 1:
            return list(preds.values())[0]
        
        result = {"P10": np.zeros(len(X)), "P50": np.zeros(len(X)), "P90": np.zeros(len(X))}
        total_w = 0
        
        if "gbdt" in preds:
            w = 0.6
            for k in result:
                result[k] += w * preds["gbdt"][k]
            total_w += w
        
        if "moe" in preds:
            w = 0.4
            for k in result:
                result[k] += w * preds["moe"][k]
            total_w += w
        
        if total_w > 0:
            for k in result:
                result[k] /= total_w
        
        return result


class StrategyOptimizer:
    def __init__(self):
        self.config = None
    
    def optimize_scale_function(self,
                               df: pd.DataFrame,
                               targets: List[str],
                               model_predict_fn) -> dict:
        from competition.optimal_bidding import OptimalConfidenceStrategy, BacktestSettler
        from config import TIMEZONE
        
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZING SCALE FUNCTION")
        logger.info("=" * 60)
        
        settler = BacktestSettler()
        
        best_config = None
        best_profit = float("-inf")
        
        confidence_thresholds = [0.5, 0.8, 1.0, 1.2]
        max_scales_pos = [1.05, 1.10, 1.15]
        spread_weights = [0.3, 0.5, 0.7]
        
        for conf_th in confidence_thresholds:
            for max_sp in max_scales_pos:
                for sp_w in spread_weights:
                    config = {
                        "confidence_threshold": conf_th,
                        "max_scale_positive": max_sp,
                        "max_scale_negative": 2 - max_sp,
                        "spread_confidence_weight": sp_w,
                    }
                    
                    profits = []
                    
                    start_ts = pd.Timestamp("2026-04-01", tz=TIMEZONE)
                    end_ts = pd.Timestamp("2026-04-20", tz=TIMEZONE)
                    dates = pd.date_range(start=start_ts, end=end_ts, freq="D")
                    
                    for d in dates:
                        day_mask = (df.index >= d) & (df.index < d + pd.Timedelta(days=1))
                        day_df = df[day_mask]
                        
                        if len(day_df) < 96:
                            continue
                        
                        given_curve = day_df["LOAD_REAL"].values[:96] if "LOAD_REAL" in day_df.columns else np.ones(96) * 100
                        actual_load = given_curve
                        actual_da = day_df["PRICE_DAYAGO"].values[:96] if "PRICE_DAYAGO" in day_df.columns else np.ones(96) * 50
                        actual_rt = day_df["PRICE_REAL"].values[:96] if "PRICE_REAL" in day_df.columns else np.ones(96) * 55
                        
                        spread_p50 = actual_rt - actual_da
                        spread_p10 = spread_p50 * 0.8
                        spread_p90 = spread_p50 * 1.2
                        
                        strategy = OptimalConfidenceStrategy(config)
                        bids = strategy.compute_hourly_bids(
                            given_curve, spread_p50, spread_p10, spread_p90, np.ones(96)
                        )
                        
                        settlement = settler.settle_hourly(
                            bids, actual_load, actual_da, actual_rt
                        )
                        profits.append(settlement["profit"])
                    
                    total_profit = sum(profits)
                    
                    if total_profit > best_profit:
                        best_profit = total_profit
                        best_config = config.copy()
        
        logger.info(f"\nBest config: {best_config}")
        logger.info(f"Best profit: {best_profit:,.2f}")
        
        self.config = best_config
        return best_config


def main():
    logger.info("=" * 70)
    logger.info("COMPLETE PIPELINE: Training + Fusion + Strategy Optimization")
    logger.info("=" * 70)
    
    data_path = "./data/output/feature_master_table.parquet"
    model_dir = "./saved_models"
    
    logger.info("\n[1] Loading data...")
    df = pd.read_parquet(data_path)
    logger.info(f"    Data shape: {df.shape}")
    logger.info(f"    Date range: {df.index.min()} ~ {df.index.max()}")
    
    from config import TARGET_COLUMNS
    targets = ["LOAD_REAL", "PRICE_DAYAGO", "PRICE_REAL", "PRICE_R_D"]
    
    logger.info("\n[2] Training GBDT models...")
    trainer = ModelTrainer(model_dir=model_dir)
    trainer.train_gbdt(df, targets)
    
    logger.info("\n[3] Model Fusion...")
    fusion = ModelFusion(model_dir=model_dir)
    
    logger.info("\n[4] Evaluating models on validation set...")
    from config import TIMEZONE, SPLIT_DATES
    
    valid_start = pd.Timestamp(SPLIT_DATES["valid"][0], tz=TIMEZONE)
    valid_end = pd.Timestamp(SPLIT_DATES["valid"][1], tz=TIMEZONE) + pd.Timedelta(hours=23, minutes=45)
    valid_mask = (df.index >= valid_start) & (df.index <= valid_end)
    valid_df = df[valid_mask]
    
    fusion_results = {}
    for target in targets[:1]:
        if target not in valid_df.columns:
            continue
        
        y_valid = valid_df[target].dropna()
        if len(y_valid) < 100:
            continue
        
        pred = fusion.predict(valid_df.loc[y_valid.index], target)
        mae = np.mean(np.abs(y_valid.values - pred["P50"]))
        
        fusion_results[target] = {
            "mae": mae,
            "rmse": np.sqrt(np.mean((y_valid.values - pred["P50"]) ** 2)),
        }
        logger.info(f"    {target}: MAE={mae:.4f}, RMSE={fusion_results[target]['rmse']:.4f}")
    
    logger.info("\n[5] Optimizing strategy scale function...")
    optimizer = StrategyOptimizer()
    best_config = optimizer.optimize_scale_function(df, targets, fusion.predict)
    
    logger.info("\n[6] Running final backtest comparison...")
    from competition.optimal_bidding import run_strategy_comparison
    
    results_df, summary = run_strategy_comparison(
        df=df,
        start_date="2026-04-01",
        end_date="2026-04-27",
        given_curve_col="LOAD_REAL",
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    
    for name, stats in summary.items():
        logger.info(f"\n{name}:")
        logger.info(f"    Total Profit: {stats['total_profit']:>15,.2f}")
        logger.info(f"    Sharpe Ratio: {stats['sharpe_ratio']:>15.4f}")
        logger.info(f"    Win Rate: {stats['win_rate']:>15.2%}")
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
