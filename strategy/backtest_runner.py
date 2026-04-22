import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from strategy.settlement_simulator import SettlementSimulator, SettlementRule, SettlementResult
from strategy.strategy_engine import StrategyEngine, StrategyInput, StrategyOutput, StrategyParams, StrategyMode
from strategy.strategy_baseline import StrategyBaseline
from config import POINTS_PER_DAY, TIMEZONE

logger = logging.getLogger(__name__)


class RollMode(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class DayBacktestInput:
    date: str
    q_final: np.ndarray
    contract_curve: np.ndarray
    load_actual: np.ndarray
    price_da_real: np.ndarray
    price_rt_real: np.ndarray
    is_holiday: bool = False
    is_post_holiday: bool = False
    mask_flag: int = 0
    info_completeness_level: float = 1.0
    strategy_name: str = ""
    strategy_output: Optional[StrategyOutput] = None


@dataclass
class DayBacktestResult:
    date: str
    strategy_name: str
    settlement: SettlementResult
    is_holiday: bool = False
    is_post_holiday: bool = False
    is_missing_info: bool = False
    is_high_load: bool = False
    is_high_price: bool = False
    is_high_deviation: bool = False
    scenario_type: str = "normal_workday"


@dataclass
class BacktestResult:
    strategy_name: str
    daily_results: Dict[str, DayBacktestResult]
    summary: Dict[str, float]
    scenario_summaries: Dict[str, Dict[str, float]]


class BacktestRunner:
    def __init__(self,
                 settlement_rule: SettlementRule = None,
                 strategy_params: StrategyParams = None):
        self.settlement = SettlementSimulator(rule=settlement_rule or SettlementRule())
        self.strategy_engine = StrategyEngine(params=strategy_params or StrategyParams())
        self.baseline = StrategyBaseline()

    def run_rolling_backtest(self,
                             df: pd.DataFrame,
                             start_date: str,
                             end_date: str,
                             strategy_name: str = "strategy_engine",
                             roll_mode: str = "daily",
                             lookback_days: int = 30,
                             contract_curve_col: str = "contract_curve",
                             load_actual_col: str = "LOAD_REAL",
                             price_da_col: str = "PRICE_DAYAGO",
                             price_rt_col: str = "PRICE_REAL",
                             prediction_prefix: str = "") -> BacktestResult:
        logger.info("=" * 60)
        logger.info(f"ROLLING BACKTEST: {start_date} ~ {end_date}, strategy={strategy_name}, mode={roll_mode}")
        logger.info("=" * 60)

        start = pd.Timestamp(start_date, tz=TIMEZONE)
        end = pd.Timestamp(end_date, tz=TIMEZONE)

        daily_results = {}
        current = start

        while current <= end:
            day_data = self._extract_day_data(
                df, current, contract_curve_col, load_actual_col,
                price_da_col, price_rt_col, prediction_prefix
            )

            if day_data is None:
                logger.warning(f"  No data for {current.date()}, skipping")
                current += pd.Timedelta(days=1)
                continue

            q_final = self._get_strategy_curve(
                strategy_name, day_data, df, current, lookback_days
            )

            if q_final is not None:
                day_input = DayBacktestInput(
                    date=str(current.date()),
                    q_final=q_final,
                    contract_curve=day_data["contract_curve"],
                    load_actual=day_data["load_actual"],
                    price_da_real=day_data["price_da_real"],
                    price_rt_real=day_data["price_rt_real"],
                    is_holiday=day_data.get("is_holiday", False),
                    is_post_holiday=day_data.get("is_post_holiday", False),
                    mask_flag=day_data.get("mask_flag", 0),
                    info_completeness_level=day_data.get("info_completeness_level", 1.0),
                    strategy_name=strategy_name,
                )

                settlement_result = self.settlement.settle_day(
                    q_final=day_input.q_final,
                    contract_curve=day_input.contract_curve,
                    load_actual=day_input.load_actual,
                    price_da_real=day_input.price_da_real,
                    price_rt_real=day_input.price_rt_real,
                )

                day_result = self._classify_day(day_input, settlement_result)
                daily_results[str(current.date())] = day_result

                logger.info(f"  {current.date()}: profit={settlement_result.total_profit:.2f}, "
                           f"recovery={settlement_result.recovery_cost:.2f}, "
                           f"over_threshold={settlement_result.deviation_over_threshold_count}")

            if roll_mode == RollMode.WEEKLY.value:
                current += pd.Timedelta(days=7)
            else:
                current += pd.Timedelta(days=1)

        summary = self.settlement.summarize(
            {d: r.settlement for d, r in daily_results.items()}
        )
        scenario_summaries = self._compute_scenario_summaries(daily_results)

        result = BacktestResult(
            strategy_name=strategy_name,
            daily_results=daily_results,
            summary=summary,
            scenario_summaries=scenario_summaries,
        )

        logger.info(f"\n  Summary for {strategy_name}:")
        logger.info(f"    Total profit: {summary.get('total_profit', 0):.2f}")
        logger.info(f"    Avg daily profit: {summary.get('avg_daily_profit', 0):.2f}")
        logger.info(f"    Total recovery: {summary.get('total_recovery', 0):.2f}")
        logger.info(f"    Over threshold: {summary.get('total_over_threshold', 0)}")

        return result

    def run_multi_strategy_backtest(self,
                                    df: pd.DataFrame,
                                    start_date: str,
                                    end_date: str,
                                    strategy_names: List[str] = None,
                                    **kwargs) -> Dict[str, BacktestResult]:
        if strategy_names is None:
            strategy_names = ["contract_curve", "p50", "conservative", "aggressive", "scenario_optimized", "moe", "strategy_engine"]

        results = {}
        for name in strategy_names:
            logger.info(f"\nRunning backtest for strategy: {name}")
            results[name] = self.run_rolling_backtest(
                df, start_date, end_date, strategy_name=name, **kwargs
            )

        return results

    def _extract_day_data(self,
                          df: pd.DataFrame,
                          date: pd.Timestamp,
                          contract_col: str,
                          load_col: str,
                          pda_col: str,
                          prt_col: str,
                          pred_prefix: str) -> Optional[Dict]:
        day_mask = df.index.date == date.date()
        if day_mask.sum() != POINTS_PER_DAY:
            if day_mask.sum() == 0:
                return None
            logger.warning(f"  {date.date()}: {day_mask.sum()} points (expected {POINTS_PER_DAY})")

        day_df = df[day_mask]

        result = {}

        if contract_col in day_df.columns:
            result["contract_curve"] = day_df[contract_col].values
        else:
            if load_col in day_df.columns:
                result["contract_curve"] = day_df[load_col].values
            else:
                return None

        if load_col in day_df.columns:
            result["load_actual"] = day_df[load_col].values
        else:
            return None

        if pda_col in day_df.columns:
            result["price_da_real"] = day_df[pda_col].values
        else:
            result["price_da_real"] = np.zeros(POINTS_PER_DAY)

        if prt_col in day_df.columns:
            result["price_rt_real"] = day_df[prt_col].values
        else:
            result["price_rt_real"] = np.zeros(POINTS_PER_DAY)

        for pred_col, key in [
            (f"{pred_prefix}load_pred_p10", "load_pred_p10"),
            (f"{pred_prefix}load_pred_p50", "load_pred_p50"),
            (f"{pred_prefix}load_pred_p90", "load_pred_p90"),
            (f"{pred_prefix}price_da_pred_p10", "price_da_pred_p10"),
            (f"{pred_prefix}price_da_pred_p50", "price_da_pred_p50"),
            (f"{pred_prefix}price_da_pred_p90", "price_da_pred_p90"),
            (f"{pred_prefix}price_rt_pred_p10", "price_rt_pred_p10"),
            (f"{pred_prefix}price_rt_pred_p50", "price_rt_pred_p50"),
            (f"{pred_prefix}price_rt_pred_p90", "price_rt_pred_p90"),
            (f"{pred_prefix}spread_pred_p10", "spread_pred_p10"),
            (f"{pred_prefix}spread_pred_p50", "spread_pred_p50"),
            (f"{pred_prefix}spread_pred_p90", "spread_pred_p90"),
        ]:
            if pred_col in day_df.columns:
                result[key] = day_df[pred_col].values

        result["is_holiday"] = bool(day_df["is_holiday"].max()) if "is_holiday" in day_df.columns else False
        result["is_post_holiday"] = bool(day_df["is_post_holiday"].max()) if "is_post_holiday" in day_df.columns else False
        result["mask_flag"] = int(day_df["mask_flag"].max()) if "mask_flag" in day_df.columns else 0
        result["info_completeness_level"] = float(day_df.get("info_completeness_level", pd.Series([1.0])).iloc[0])

        return result

    def _get_strategy_curve(self,
                            strategy_name: str,
                            day_data: Dict,
                            df: pd.DataFrame,
                            date: pd.Timestamp,
                            lookback_days: int) -> Optional[np.ndarray]:
        if strategy_name == "strategy_engine":
            return self._get_engine_curve(day_data)
        elif strategy_name in self.baseline.list_baselines():
            inp = self._build_strategy_input(day_data)
            return self.baseline.generate_single(strategy_name, inp)
        else:
            logger.warning(f"  Unknown strategy: {strategy_name}")
            return day_data.get("contract_curve", day_data.get("load_actual"))

    def _get_engine_curve(self, day_data: Dict) -> Optional[np.ndarray]:
        inp = self._build_strategy_input(day_data)
        try:
            output = self.strategy_engine.generate_daily_strategy(inp)
            return output.q_final
        except Exception as e:
            logger.error(f"  Strategy engine failed: {e}")
            return day_data.get("contract_curve", day_data.get("load_actual"))

    def _build_strategy_input(self, day_data: Dict) -> StrategyInput:
        load_p50 = day_data.get("load_pred_p50", day_data.get("load_actual", np.zeros(96)))
        load_p10 = day_data.get("load_pred_p10", load_p50 * 0.95)
        load_p90 = day_data.get("load_pred_p90", load_p50 * 1.05)

        pda_p50 = day_data.get("price_da_pred_p50", day_data.get("price_da_real", np.zeros(96)))
        pda_p10 = day_data.get("price_da_pred_p10", pda_p50 * 0.9)
        pda_p90 = day_data.get("price_da_pred_p90", pda_p50 * 1.1)

        prt_p50 = day_data.get("price_rt_pred_p50", day_data.get("price_rt_real", np.zeros(96)))
        prt_p10 = day_data.get("price_rt_pred_p10", prt_p50 * 0.9)
        prt_p90 = day_data.get("price_rt_pred_p90", prt_p50 * 1.1)

        spread_p50 = day_data.get("spread_pred_p50", prt_p50 - pda_p50)
        spread_p10 = day_data.get("spread_pred_p10", prt_p10 - pda_p10)
        spread_p90 = day_data.get("spread_pred_p90", prt_p90 - pda_p90)

        contract = day_data.get("contract_curve", load_p50)

        return StrategyInput(
            load_pred_p10=load_p10,
            load_pred_p50=load_p50,
            load_pred_p90=load_p90,
            price_da_pred_p10=pda_p10,
            price_da_pred_p50=pda_p50,
            price_da_pred_p90=pda_p90,
            price_rt_pred_p10=prt_p10,
            price_rt_pred_p50=prt_p50,
            price_rt_pred_p90=prt_p90,
            spread_pred_p10=spread_p10,
            spread_pred_p50=spread_p50,
            spread_pred_p90=spread_p90,
            is_holiday=day_data.get("is_holiday", False),
            is_post_holiday=day_data.get("is_post_holiday", False),
            mask_flag=day_data.get("mask_flag", 0),
            info_completeness_level=day_data.get("info_completeness_level", 1.0),
            contract_curve=contract,
        )

    def _classify_day(self, day_input: DayBacktestInput,
                      settlement: SettlementResult) -> DayBacktestResult:
        is_holiday = day_input.is_holiday
        is_post_holiday = day_input.is_post_holiday
        is_missing = day_input.mask_flag == 1

        load_mean = np.mean(day_input.load_actual)
        is_high_load = False
        if load_mean > 0:
            is_high_load = load_mean > np.percentile(day_input.load_actual, 80)

        price_mean = np.mean(day_input.price_rt_real)
        is_high_price = price_mean > np.percentile(day_input.price_rt_real, 80) if price_mean > 0 else False

        is_high_deviation = settlement.deviation_over_threshold_count > 10

        if is_holiday:
            scenario_type = "holiday"
        elif is_post_holiday:
            scenario_type = "post_holiday"
        elif is_missing:
            scenario_type = "missing_info"
        elif is_high_price:
            scenario_type = "high_price"
        elif is_high_load:
            scenario_type = "high_load"
        elif is_high_deviation:
            scenario_type = "high_deviation"
        else:
            scenario_type = "normal_workday"

        return DayBacktestResult(
            date=day_input.date,
            strategy_name=day_input.strategy_name,
            settlement=settlement,
            is_holiday=is_holiday,
            is_post_holiday=is_post_holiday,
            is_missing_info=is_missing,
            is_high_load=is_high_load,
            is_high_price=is_high_price,
            is_high_deviation=is_high_deviation,
            scenario_type=scenario_type,
        )

    def _compute_scenario_summaries(self,
                                    daily_results: Dict[str, DayBacktestResult]) -> Dict[str, Dict[str, float]]:
        scenario_groups = {}
        for date, result in daily_results.items():
            st = result.scenario_type
            if st not in scenario_groups:
                scenario_groups[st] = []
            scenario_groups[st].append(result)

        summaries = {}
        for scenario, results in scenario_groups.items():
            profits = [r.settlement.total_profit for r in results]
            recoveries = [r.settlement.recovery_cost for r in results]
            over_counts = [r.settlement.deviation_over_threshold_count for r in results]

            summaries[scenario] = {
                "avg_profit": np.mean(profits),
                "worst_day_profit": min(profits),
                "best_day_profit": max(profits),
                "std_profit": np.std(profits),
                "total_recovery": sum(recoveries),
                "avg_recovery": np.mean(recoveries),
                "total_over_threshold": sum(over_counts),
                "avg_over_threshold": np.mean(over_counts),
                "n_days": len(results),
                "stability": np.mean(profits) / (np.std(profits) + 1e-9),
            }

        return summaries

    def simulate_may_missing(self,
                             df: pd.DataFrame,
                             strategy_name: str = "strategy_engine",
                             may_start: str = "2026-05-01",
                             may_end: str = "2026-05-10",
                             missing_dates: List[str] = None,
                             **kwargs) -> BacktestResult:
        logger.info("=" * 60)
        logger.info(f"MAY MISSING SIMULATION: {may_start} ~ {may_end}")
        logger.info("=" * 60)

        if missing_dates is None:
            missing_dates = ["2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05", "2026-05-06"]

        df_sim = df.copy()
        for date_str in missing_dates:
            try:
                mask_date = df_sim.index.date == pd.Timestamp(date_str).date()
                for col in ["PRICE_DAYAGO", "WATER_DAYAGO", "NOMARKET_DAYAGO", "LINE_DAYAGO"]:
                    if col in df_sim.columns:
                        df_sim.loc[mask_date, col] = np.nan
                if "mask_flag" in df_sim.columns:
                    df_sim.loc[mask_date, "mask_flag"] = 1
                if "info_completeness_level" in df_sim.columns:
                    df_sim.loc[mask_date, "info_completeness_level"] = 0.2
            except Exception as e:
                logger.warning(f"  Failed to simulate missing for {date_str}: {e}")

        return self.run_rolling_backtest(
            df_sim, may_start, may_end, strategy_name=strategy_name, **kwargs
        )
