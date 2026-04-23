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
    handlers=[logging.StreamHandler(), logging.FileHandler("./logs/full_pipeline.log", mode="a")],
)
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    q10_pred = y_pred - 0.1 * np.std(y_true)
    q90_pred = y_pred + 0.1 * np.std(y_true)

    def pinball_loss(y_true, y_pred, q):
        error = y_true - y_pred
        return np.mean(np.where(error >= 0, q * error, (q - 1) * error))

    pinball_10 = pinball_loss(y_true, q10_pred, 0.1)
    pinball_50 = pinball_loss(y_true, y_pred, 0.5)
    pinball_90 = pinball_loss(y_true, q90_pred, 0.9)

    coverage_10 = np.mean((y_true >= q10_pred))
    coverage_90 = np.mean((y_true <= q90_pred))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Pinball_P10": pinball_10,
        "Pinball_P50": pinball_50,
        "Pinball_P90": pinball_90,
        "Coverage_P10_P90": (coverage_10 + coverage_90) / 2,
    }


def main():
    logger.info("=" * 80)
    logger.info("COMPLETE PIPELINE: Training + Evaluation + Strategy + Backtest")
    logger.info("=" * 80)

    data_path = "./data/output/feature_master_table.parquet"
    contract_path = "./data/raw/cont_line_dayahead_unified.parquet"
    model_dir = "./saved_models"

    logger.info("\n[STEP 1] Loading data...")
    df = pd.read_parquet(data_path)
    logger.info(f"  Data shape: {df.shape}")
    logger.info(f"  Date range: {df.index.min()} ~ {df.index.max()}")

    contract_df = pd.read_parquet(contract_path)
    logger.info(f"  Contract data: {contract_df.shape}")

    from config import TARGET_COLUMNS, SPLIT_DATES, TIMEZONE, POINTS_PER_DAY
    from training.train_pipeline_v2 import ThreePathPipeline, ModelPathConfig

    train_start = pd.Timestamp(SPLIT_DATES["train"][0], tz=TIMEZONE)
    train_end = pd.Timestamp(SPLIT_DATES["train"][1], tz=TIMEZONE) + pd.Timedelta(hours=23, minutes=45)
    valid_start = pd.Timestamp(SPLIT_DATES["valid"][0], tz=TIMEZONE)
    valid_end = pd.Timestamp(SPLIT_DATES["valid"][1], tz=TIMEZONE) + pd.Timedelta(hours=23, minutes=45)

    train_mask = (df.index >= train_start) & (df.index <= train_end)
    valid_mask = (df.index >= valid_start) & (df.index <= valid_end)

    train_df = df[train_mask].copy()
    valid_df = df[valid_mask].copy()

    logger.info(f"  Train set: {len(train_df)} rows ({train_start.date()} ~ {train_end.date()})")
    logger.info(f"  Valid set: {len(valid_df)} rows ({valid_start.date()} ~ {valid_end.date()})")

    logger.info("\n[STEP 2] Training Three-Path Models (GBDT + MoE + Simple)...")
    logger.info("(TFT training skipped for speed - can be enabled in ModelPathConfig)")

    config = ModelPathConfig(
        use_gbdt=True,
        use_tft=False,
        use_moe=True,
        use_simple=True,
        gbdt_weight=0.50,
        moe_weight=0.35,
        simple_weight=0.15,
    )

    os.makedirs(model_dir, exist_ok=True)

    from training.trainer import PipelineTrainer
    trainer = PipelineTrainer(model_dir=model_dir)
    trainer.train_all(df)

    logger.info("\n[STEP 3] Model Evaluation on Validation Set...")
    logger.info("-" * 80)

    results = {}
    for target in TARGET_COLUMNS:
        if target not in valid_df.columns:
            continue
        if target not in trainer.gbdt_models:
            continue

        y_valid = valid_df[target].dropna()
        if len(y_valid) < 10:
            continue

        feature_cols = trainer.feature_cols_map.get(target, trainer._get_base_feature_cols(valid_df, target))
        X_valid = valid_df[feature_cols].fillna(0).loc[y_valid.index]

        pred = trainer.predict(valid_df, target, X=X_valid)
        y_pred = pred["P50"]

        valid_idx = y_valid.index
        y_true = y_valid.values

        for col in feature_cols:
            if col not in X_valid.columns:
                X_valid = X_valid.copy()
                X_valid[col] = 0.0

        metrics = compute_metrics(y_true, y_pred)
        results[target] = metrics

        logger.info(f"\n  [{target}]")
        logger.info(f"    MAE:            {metrics['MAE']:.4f}")
        logger.info(f"    RMSE:           {metrics['RMSE']:.4f}")
        logger.info(f"    Pinball P10:    {metrics['Pinball_P10']:.4f}")
        logger.info(f"    Pinball P50:    {metrics['Pinball_P50']:.4f}")
        logger.info(f"    Pinball P90:    {metrics['Pinball_P90']:.4f}")
        logger.info(f"    Coverage P10-90: {metrics['Coverage_P10_P90']:.2%}")

    logger.info("\n" + "-" * 80)
    logger.info("SUMMARY:")
    for target, metrics in results.items():
        logger.info(f"  {target}: MAE={metrics['MAE']:.2f}, Pinball_P50={metrics['Pinball_P50']:.2f}")

    logger.info("\n[STEP 4] Strategy Generation and Backtest...")
    logger.info("-" * 80)

    from strategy.settlement_simulator import SettlementSimulator, SettlementRule
    from strategy.scenario_sampler import ScenarioSampler, ScenarioConfig
    from strategy.strategy_engine import StrategyEngine, StrategyParams
    from strategy.confidence_strategy import SpreadConfidenceMeter, ConfidenceBoostedCandidateGenerator

    settlement = SettlementSimulator(rule=SettlementRule())
    sampler = ScenarioSampler(config=ScenarioConfig(n_scenarios=50))
    strategy_engine = StrategyEngine(
        params=StrategyParams(),
        settlement_rule=SettlementRule(),
        scenario_config=ScenarioConfig(n_scenarios=50)
    )
    conf_meter = SpreadConfidenceMeter()

    valid_dates = valid_df.index.normalize().unique()
    daily_results = []

    for date in sorted(valid_dates):
        date_str = str(date.date())

        day_mask = valid_df.index.normalize() == date
        day_df = valid_df[day_mask]

        if len(day_df) < POINTS_PER_DAY:
            day_df = day_df.reindex(pd.date_range(date, periods=POINTS_PER_DAY, freq="15min", tz=TIMEZONE))
            day_df = day_df.tz_convert(TIMEZONE)

        contract_day = contract_df[contract_df.index.normalize() == date]
        if len(contract_day) == 0:
            contract_curve = np.ones(POINTS_PER_DAY) * np.mean(day_df.get("LOAD_REAL", [1000])) if "LOAD_REAL" in day_df.columns else np.ones(POINTS_PER_DAY) * 1000
        else:
            contract_curve = contract_day["load"].values[:POINTS_PER_DAY]
            if len(contract_curve) < POINTS_PER_DAY:
                contract_curve = np.pad(contract_curve, (0, POINTS_PER_DAY - len(contract_curve)), mode="edge")

        feature_cols = trainer.feature_cols_map.get("LOAD_REAL", trainer._get_base_feature_cols(valid_df, "LOAD_REAL"))

        day_preds = {}
        for target in ["LOAD_REAL", "PRICE_REAL", "PRICE_DAYAGO"]:
            if target not in trainer.gbdt_models or target not in valid_df.columns:
                continue

            y_day = day_df[target].dropna() if target in day_df.columns else pd.Series()
            X_day = day_df[feature_cols].fillna(0)

            if len(X_day) > 0:
                pred = trainer.predict(valid_df, target, X=X_day)
                day_preds[target] = pred

        load_p50 = day_preds.get("LOAD_REAL", {}).get("P50", contract_curve)
        load_p10 = day_preds.get("LOAD_REAL", {}).get("P10", load_p50 * 0.95)
        load_p90 = day_preds.get("LOAD_REAL", {}).get("P90", load_p50 * 1.05)

        price_da = day_preds.get("PRICE_DAYAGO", {}).get("P50", np.ones(POINTS_PER_DAY) * 50)
        price_da_p10 = day_preds.get("PRICE_DAYAGO", {}).get("P10", price_da * 0.9)
        price_da_p90 = day_preds.get("PRICE_DAYAGO", {}).get("P90", price_da * 1.1)

        price_rt = day_preds.get("PRICE_REAL", {}).get("P50", np.ones(POINTS_PER_DAY) * 55)
        price_rt_p10 = day_preds.get("PRICE_REAL", {}).get("P10", price_rt * 0.9)
        price_rt_p90 = day_preds.get("PRICE_REAL", {}).get("P90", price_rt * 1.1)

        spread_p50 = price_rt - price_da
        spread_p10 = price_rt_p10 - price_da_p90
        spread_p90 = price_rt_p90 - price_da_p10

        is_holiday = day_df["is_holiday"].iloc[0] if "is_holiday" in day_df.columns else 0
        is_post_holiday = day_df["is_post_holiday"].iloc[0] if "is_post_holiday" in day_df.columns else 0
        mask_flag = day_df["mask_flag"].iloc[0] if "mask_flag" in day_df.columns else 0

        conf = conf_meter.compute(spread_p10, spread_p50, spread_p90)
        logger.info(f"\n  [{date_str}] confidence={conf.confidence:.3f} ({conf.level}), "
                   f"spread_mean={np.mean(spread_p50):.2f}")

        from strategy.strategy_engine import StrategyInput
        inp = StrategyInput(
            load_pred_p10=load_p10, load_pred_p50=load_p50, load_pred_p90=load_p90,
            price_da_pred_p10=price_da_p10, price_da_pred_p50=price_da, price_da_pred_p90=price_da_p90,
            price_rt_pred_p10=price_rt_p10, price_rt_pred_p50=price_rt, price_rt_pred_p90=price_rt_p90,
            spread_pred_p10=spread_p10, spread_pred_p50=spread_p50, spread_pred_p90=spread_p90,
            is_holiday=bool(is_holiday),
            is_post_holiday=bool(is_post_holiday),
            mask_flag=int(mask_flag),
            contract_curve=contract_curve,
        )

        strategy_out = strategy_engine.generate_daily_strategy(inp)
        q_final = strategy_out.q_final

        actual_load = day_df["LOAD_REAL"].values[:POINTS_PER_DAY] if "LOAD_REAL" in day_df.columns else load_p50
        actual_da = day_df["PRICE_DAYAGO"].values[:POINTS_PER_DAY] if "PRICE_DAYAGO" in day_df.columns else price_da
        actual_rt = day_df["PRICE_REAL"].values[:POINTS_PER_DAY] if "PRICE_REAL" in day_df.columns else price_rt

        result = settlement.settle_day(
            q_final=q_final,
            contract_curve=contract_curve,
            load_actual=actual_load,
            price_da_real=actual_da,
            price_rt_real=actual_rt,
        )

        daily_results.append({
            "date": date_str,
            "mode": strategy_out.mode.value,
            "selected_curve": strategy_out.selected_curve_name,
            "confidence_level": conf.level,
            "confidence": conf.confidence,
            "spread_mean": np.mean(spread_p50),
            "q_final_mean": np.mean(q_final),
            "actual_load_mean": np.mean(actual_load),
            "profit": -result.total_cost,
            "recovery_cost": result.recovery_cost,
            "over_threshold_points": result.over_threshold_points,
            "total_cost": result.total_cost,
            "da_cost": result.da_cost,
            "rt_cost": result.rt_cost,
        })

        logger.info(f"    Selected: {strategy_out.selected_curve_name}, "
                   f"profit={-result.total_cost:.2f}, "
                   f"recovery={result.recovery_cost:.2f}, "
                   f"over_threshold={result.over_threshold_points}")

    logger.info("\n" + "=" * 80)
    logger.info("DAILY BACKTEST RESULTS (2026-04-01 ~ 2026-04-20)")
    logger.info("=" * 80)

    df_results = pd.DataFrame(daily_results)

    logger.info(f"\n{'Date':<12} {'Mode':<15} {'Curve':<20} {'Profit':>12} {'Recovery':>12} {'OverTh':>8}")
    logger.info("-" * 80)
    for _, row in df_results.iterrows():
        logger.info(f"{row['date']:<12} {row['mode']:<15} {row['selected_curve']:<20} "
                   f"{row['profit']:>12.2f} {row['recovery_cost']:>12.2f} {row['over_threshold_points']:>8}")

    logger.info("-" * 80)
    logger.info(f"{'TOTAL':<12} {'':<15} {'':<20} {df_results['profit'].sum():>12.2f} "
               f"{df_results['recovery_cost'].sum():>12.2f} {df_results['over_threshold_points'].sum():>8}")

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    logger.info(f"\n  Total Profit:      {df_results['profit'].sum():>15,.2f} 元")
    logger.info(f"  Total Recovery:     {df_results['recovery_cost'].sum():>15,.2f} 元")
    logger.info(f"  Avg Daily Profit:  {df_results['profit'].mean():>15,.2f} 元")
    logger.info(f"  Std Daily Profit:  {df_results['profit'].std():>15,.2f} 元")
    logger.info(f"  Sharpe Ratio:       {df_results['profit'].mean() / (df_results['profit'].std() + 1e-9):>15.4f}")
    logger.info(f"  Win Rate:          {(df_results['profit'] > 0).mean():>15.2%}")
    logger.info(f"  Total Over Threshold Points: {df_results['over_threshold_points'].sum():>10}")

    by_mode = df_results.groupby("mode").agg({
        "profit": ["sum", "mean", "count"],
        "recovery_cost": "mean",
        "over_threshold_points": "mean",
    })
    logger.info(f"\n  By Strategy Mode:")
    logger.info(f"  {'Mode':<15} {'Count':>6} {'Total Profit':>12} {'Avg Profit':>12} {'Avg Recovery':>12}")
    logger.info("  " + "-" * 60)
    for mode, row in by_mode.iterrows():
        logger.info(f"  {mode:<15} {int(row[('profit', 'count')]):>6} "
                   f"{row[('profit', 'sum')]:>12.2f} {row[('profit', 'mean')]:>12.2f} "
                   f"{row[('recovery_cost', 'mean')]:>12.2f}")

    df_results.to_csv("./logs/daily_backtest_results.csv", index=False)
    logger.info(f"\n  Results saved to ./logs/daily_backtest_results.csv")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
