import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from itertools import product

from strategy.strategy_engine import StrategyEngine, StrategyParams, StrategyInput, StrategyOutput
from strategy.backtest_runner import BacktestRunner, BacktestResult, DayBacktestResult
from strategy.settlement_simulator import SettlementRule

logger = logging.getLogger(__name__)


@dataclass
class ParamSearchSpace:
    lambda_1: Tuple[float, float, int] = (0.1, 2.0, 5)
    lambda_2: Tuple[float, float, int] = (0.5, 3.0, 5)
    lambda_3: Tuple[float, float, int] = (0.01, 0.5, 5)
    lambda_4: Tuple[float, float, int] = (0.1, 1.0, 5)
    holiday_conservative_factor: Tuple[float, float, int] = (0.3, 0.8, 5)
    post_holiday_ramp_rate: Tuple[float, float, int] = (0.5, 1.0, 5)
    missing_info_shrink: Tuple[float, float, int] = (0.1, 0.5, 5)
    spread_positive_aggression: Tuple[float, float, int] = (0.8, 1.5, 5)
    spread_negative_shrink: Tuple[float, float, int] = (0.3, 0.8, 5)


@dataclass
class StrategyRanking:
    name: str
    total_profit: float
    avg_daily_profit: float
    risk_adjusted_profit: float
    sharpe_ratio: float
    max_drawdown: float
    total_recovery: float
    over_threshold_count: int
    holiday_profit: float
    missing_info_profit: float
    composite_score: float
    rank: int


@dataclass
class SelectionResult:
    best_strategy: str
    best_params: Optional[StrategyParams]
    rankings: List[StrategyRanking]
    scenario_best: Dict[str, str]
    param_search_history: List[Dict]


class BestStrategySelector:
    def __init__(self,
                 settlement_rule: SettlementRule = None,
                 search_space: ParamSearchSpace = None):
        self.settlement_rule = settlement_rule or SettlementRule()
        self.search_space = search_space or ParamSearchSpace()
        self._search_history = []

    def select_best_strategy(self,
                             backtest_results: Dict[str, BacktestResult],
                             weights: Dict[str, float] = None) -> SelectionResult:
        logger.info("=" * 60)
        logger.info("STRATEGY SELECTION")
        logger.info("=" * 60)

        if weights is None:
            weights = {
                "total_profit": 0.3,
                "risk_adjusted_profit": 0.25,
                "sharpe_ratio": 0.15,
                "recovery_penalty": 0.1,
                "holiday_stability": 0.1,
                "missing_info_stability": 0.1,
            }

        rankings = []
        for name, result in backtest_results.items():
            ranking = self._compute_ranking(name, result)
            rankings.append(ranking)

        rankings.sort(key=lambda r: r.composite_score, reverse=True)
        for i, r in enumerate(rankings):
            r.rank = i + 1

        for r in rankings:
            logger.info(f"  #{r.rank} {r.name}: score={r.composite_score:.4f}, "
                       f"profit={r.total_profit:.2f}, sharpe={r.sharpe_ratio:.4f}")

        scenario_best = self._find_scenario_best(backtest_results)

        best = rankings[0] if rankings else None
        best_name = best.name if best else "contract_curve"

        return SelectionResult(
            best_strategy=best_name,
            best_params=None,
            rankings=rankings,
            scenario_best=scenario_best,
            param_search_history=self._search_history,
        )

    def optimize_params(self,
                        df: pd.DataFrame,
                        start_date: str,
                        end_date: str,
                        method: str = "grid",
                        n_bayes_iter: int = 30) -> Tuple[StrategyParams, Dict]:
        logger.info("=" * 60)
        logger.info(f"PARAMETER OPTIMIZATION: method={method}")
        logger.info("=" * 60)

        if method == "grid":
            return self._grid_search(df, start_date, end_date)
        elif method == "bayes":
            return self._bayesian_search(df, start_date, end_date, n_bayes_iter)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _grid_search(self, df: pd.DataFrame,
                     start_date: str, end_date: str) -> Tuple[StrategyParams, Dict]:
        space = self.search_space
        param_ranges = {
            "lambda_1": np.linspace(*space.lambda_1),
            "lambda_2": np.linspace(*space.lambda_2),
            "lambda_3": np.linspace(*space.lambda_3),
            "lambda_4": np.linspace(*space.lambda_4),
        }

        best_score = float("-inf")
        best_params = StrategyParams()
        all_results = []

        keys = list(param_ranges.keys())
        total_combos = 1
        for k in keys:
            total_combos *= len(param_ranges[k])

        logger.info(f"  Grid search: {total_combos} combinations")

        count = 0
        for values in product(*[param_ranges[k] for k in keys]):
            params = StrategyParams(
                lambda_1=values[0],
                lambda_2=values[1],
                lambda_3=values[2],
                lambda_4=values[3],
            )

            try:
                runner = BacktestRunner(
                    settlement_rule=self.settlement_rule,
                    strategy_params=params,
                )
                result = runner.run_rolling_backtest(
                    df, start_date, end_date, strategy_name="strategy_engine"
                )

                score = self._compute_objective(result)
                all_results.append({
                    "lambda_1": values[0],
                    "lambda_2": values[1],
                    "lambda_3": values[2],
                    "lambda_4": values[3],
                    "score": score,
                    "total_profit": result.summary.get("total_profit", 0),
                    "sharpe": result.summary.get("profit_sharpe", 0),
                })

                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(f"    New best: score={score:.4f}, params={values}")

            except Exception as e:
                logger.warning(f"    Failed for params {values}: {e}")

            count += 1
            if count % 50 == 0:
                logger.info(f"  Progress: {count}/{total_combos}")

        self._search_history.extend(all_results)

        logger.info(f"\n  Best params: lambda_1={best_params.lambda_1}, "
                   f"lambda_2={best_params.lambda_2}, "
                   f"lambda_3={best_params.lambda_3}, "
                   f"lambda_4={best_params.lambda_4}")
        logger.info(f"  Best score: {best_score:.4f}")

        return best_params, {"all_results": all_results, "best_score": best_score}

    def _bayesian_search(self, df: pd.DataFrame,
                         start_date: str, end_date: str,
                         n_iter: int = 30) -> Tuple[StrategyParams, Dict]:
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.warning("scipy not available, falling back to grid search")
            return self._grid_search(df, start_date, end_date)

        space = self.search_space
        bounds = [
            (space.lambda_1[0], space.lambda_1[1]),
            (space.lambda_2[0], space.lambda_2[1]),
            (space.lambda_3[0], space.lambda_3[1]),
            (space.lambda_4[0], space.lambda_4[1]),
        ]

        best_score = float("-inf")
        best_params = StrategyParams()
        all_results = []

        def objective(x):
            nonlocal best_score, best_params
            params = StrategyParams(
                lambda_1=x[0], lambda_2=x[1],
                lambda_3=x[2], lambda_4=x[3],
            )
            try:
                runner = BacktestRunner(
                    settlement_rule=self.settlement_rule,
                    strategy_params=params,
                )
                result = runner.run_rolling_backtest(
                    df, start_date, end_date, strategy_name="strategy_engine"
                )
                score = self._compute_objective(result)
                all_results.append({
                    "lambda_1": x[0], "lambda_2": x[1],
                    "lambda_3": x[2], "lambda_4": x[3],
                    "score": score,
                })
                if score > best_score:
                    best_score = score
                    best_params = params
                return -score
            except Exception:
                return 0.0

        n_initial = min(10, n_iter)
        for _ in range(n_initial):
            x0 = np.array([b[0] + np.random.random() * (b[1] - b[0]) for b in bounds])
            objective(x0)

        try:
            x_best = np.array([best_params.lambda_1, best_params.lambda_2,
                              best_params.lambda_3, best_params.lambda_4])
            result = minimize(objective, x_best, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": n_iter - n_initial})
        except Exception as e:
            logger.warning(f"  Bayesian optimization failed: {e}")

        self._search_history.extend(all_results)
        return best_params, {"all_results": all_results, "best_score": best_score}

    def _compute_ranking(self, name: str, result: BacktestResult) -> StrategyRanking:
        summary = result.summary
        scenario_summaries = result.scenario_summaries

        total_profit = summary.get("total_profit", 0)
        avg_profit = summary.get("avg_daily_profit", 0)
        sharpe = summary.get("profit_sharpe", 0)
        total_recovery = summary.get("total_recovery", 0)
        over_threshold = summary.get("total_over_threshold", 0)

        std_profit = summary.get("std_daily_profit", 1)
        risk_adjusted = avg_profit / (std_profit + 1e-9)

        holiday_profit = 0
        if "holiday" in scenario_summaries:
            holiday_profit = scenario_summaries["holiday"].get("avg_profit", 0)

        missing_profit = 0
        if "missing_info" in scenario_summaries:
            missing_profit = scenario_summaries["missing_info"].get("avg_profit", 0)

        max_drawdown = 0
        profits = [r.settlement.total_profit for r in result.daily_results.values()]
        if len(profits) > 1:
            cumsum = np.cumsum(profits)
            running_max = np.maximum.accumulate(cumsum)
            drawdowns = cumsum - running_max
            max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0

        composite = (
            0.3 * np.sign(total_profit) * np.log1p(abs(total_profit))
            + 0.25 * risk_adjusted
            + 0.15 * sharpe
            - 0.1 * np.log1p(total_recovery)
            + 0.1 * np.sign(holiday_profit) * np.log1p(abs(holiday_profit))
            + 0.1 * np.sign(missing_profit) * np.log1p(abs(missing_profit))
        )

        return StrategyRanking(
            name=name,
            total_profit=total_profit,
            avg_daily_profit=avg_profit,
            risk_adjusted_profit=risk_adjusted,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            total_recovery=total_recovery,
            over_threshold_count=over_threshold,
            holiday_profit=holiday_profit,
            missing_info_profit=missing_profit,
            composite_score=composite,
            rank=0,
        )

    def _compute_objective(self, result: BacktestResult) -> float:
        summary = result.summary
        total_profit = summary.get("total_profit", 0)
        sharpe = summary.get("profit_sharpe", 0)
        total_recovery = summary.get("total_recovery", 0)
        over_threshold = summary.get("total_over_threshold", 0)

        scenario_summaries = result.scenario_summaries
        holiday_stability = 0
        if "holiday" in scenario_summaries:
            holiday_stability = scenario_summaries["holiday"].get("stability", 0)

        missing_stability = 0
        if "missing_info" in scenario_summaries:
            missing_stability = scenario_summaries["missing_info"].get("stability", 0)

        score = (
            total_profit
            + 100 * sharpe
            - 2 * total_recovery
            - 5 * over_threshold
            + 50 * holiday_stability
            + 50 * missing_stability
        )
        return score

    def _find_scenario_best(self,
                            backtest_results: Dict[str, BacktestResult]) -> Dict[str, str]:
        scenario_best = {}
        scenarios = set()
        for result in backtest_results.values():
            scenarios.update(result.scenario_summaries.keys())

        for scenario in scenarios:
            best_name = None
            best_profit = float("-inf")
            for name, result in backtest_results.items():
                if scenario in result.scenario_summaries:
                    profit = result.scenario_summaries[scenario].get("avg_profit", float("-inf"))
                    if profit > best_profit:
                        best_profit = profit
                        best_name = name
            if best_name is not None:
                scenario_best[scenario] = best_name

        for scenario, name in scenario_best.items():
            logger.info(f"  Best for {scenario}: {name}")

        return scenario_best

    def generate_feedback(self,
                          backtest_results: Dict[str, BacktestResult],
                          selection: SelectionResult) -> Dict:
        feedback = {
            "best_strategy": selection.best_strategy,
            "scenario_best": selection.scenario_best,
            "worst_periods": [],
            "model_recommendations": [],
        }

        best_result = backtest_results.get(selection.best_strategy)
        if best_result is None:
            return feedback

        sorted_days = sorted(
            best_result.daily_results.items(),
            key=lambda x: x[1].settlement.total_profit
        )

        for date, day_result in sorted_days[:5]:
            feedback["worst_periods"].append({
                "date": date,
                "profit": day_result.settlement.total_profit,
                "recovery": day_result.settlement.recovery_cost,
                "over_threshold": day_result.settlement.deviation_over_threshold_count,
                "scenario": day_result.scenario_type,
            })

        for scenario, best_name in selection.scenario_best.items():
            feedback["model_recommendations"].append({
                "scenario": scenario,
                "recommended_strategy": best_name,
            })

        return feedback
