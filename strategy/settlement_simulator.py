import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SettlementRule:
    deviation_threshold: float = 0.10
    mu: float = 1.05
    negative_spread_no_recovery: bool = True
    settlement_mode: str = "double_deviation"


@dataclass
class SettlementResult:
    day_ahead_cost: float = 0.0
    real_time_cost: float = 0.0
    recovery_cost: float = 0.0
    total_cost: float = 0.0
    total_profit: float = 0.0
    recovery_ratio: float = 0.0
    deviation_over_threshold_count: int = 0
    per_point_detail: Optional[pd.DataFrame] = None


class SettlementSimulator:
    def __init__(self, rule: SettlementRule = None):
        self.rule = rule or SettlementRule()

    def settle_day(self,
                   q_final: np.ndarray,
                   contract_curve: np.ndarray,
                   load_actual: np.ndarray,
                   price_da_real: np.ndarray,
                   price_rt_real: np.ndarray) -> SettlementResult:
        n = len(q_final)
        if not (len(contract_curve) == len(load_actual) == len(price_da_real) == len(price_rt_real) == n):
            raise ValueError(
                f"Length mismatch: q={len(q_final)}, contract={len(contract_curve)}, "
                f"load={len(load_actual)}, pda={len(price_da_real)}, prt={len(price_rt_real)}"
            )

        c_da = np.sum((q_final - contract_curve) * price_da_real)

        c_rt = np.sum((load_actual - q_final) * price_rt_real)

        recovery, over_count = self._compute_recovery(
            q_final, contract_curve, load_actual, price_da_real, price_rt_real
        )

        total_cost = c_da + c_rt + recovery
        total_profit = -total_cost
        recovery_ratio = recovery / (abs(c_da) + abs(c_rt) + 1e-9)

        per_point = pd.DataFrame({
            "q_final": q_final,
            "contract": contract_curve,
            "load_actual": load_actual,
            "price_da": price_da_real,
            "price_rt": price_rt_real,
            "spread": price_rt_real - price_da_real,
            "da_cost": (q_final - contract_curve) * price_da_real,
            "rt_cost": (load_actual - q_final) * price_rt_real,
            "deviation_ratio": (q_final - load_actual) / (load_actual + 1e-9),
            "recovery": np.zeros(n),
        })

        upper_mask = q_final > (1 + self.rule.deviation_threshold) * load_actual
        lower_mask = q_final < (1 - self.rule.deviation_threshold) * load_actual
        per_point_recovery = self._per_point_recovery(
            q_final, contract_curve, load_actual, price_da_real, price_rt_real,
            upper_mask, lower_mask
        )
        combined_mask = upper_mask | lower_mask
        per_point_values = per_point["recovery"].values.copy()
        per_point_values[combined_mask] = per_point_recovery[combined_mask]
        per_point["recovery"] = per_point_values

        result = SettlementResult(
            day_ahead_cost=c_da,
            real_time_cost=c_rt,
            recovery_cost=recovery,
            total_cost=total_cost,
            total_profit=total_profit,
            recovery_ratio=recovery_ratio,
            deviation_over_threshold_count=over_count,
            per_point_detail=per_point,
        )
        return result

    def _compute_recovery(self,
                          q_final: np.ndarray,
                          contract_curve: np.ndarray,
                          load_actual: np.ndarray,
                          price_da_real: np.ndarray,
                          price_rt_real: np.ndarray) -> tuple:
        threshold = self.rule.deviation_threshold
        mu = self.rule.mu
        total_recovery = 0.0
        over_count = 0

        for t in range(len(q_final)):
            lt = load_actual[t]
            if lt <= 0:
                continue
            qt = q_final[t]
            spread_t = price_rt_real[t] - price_da_real[t]

            upper_bound = (1 + threshold) * lt
            lower_bound = (1 - threshold) * lt

            if qt > upper_bound:
                excess = qt - upper_bound
                if spread_t > 0:
                    recovery_t = excess * spread_t * mu
                else:
                    if self.rule.negative_spread_no_recovery:
                        recovery_t = 0.0
                    else:
                        recovery_t = abs(excess * spread_t * mu)
                total_recovery += recovery_t
                over_count += 1

            elif qt < lower_bound:
                deficit = lower_bound - qt
                if spread_t > 0:
                    recovery_t = deficit * spread_t * mu
                else:
                    if self.rule.negative_spread_no_recovery:
                        recovery_t = 0.0
                    else:
                        recovery_t = abs(deficit * spread_t * mu)
                total_recovery += recovery_t
                over_count += 1

        return total_recovery, over_count

    def _per_point_recovery(self,
                            q_final: np.ndarray,
                            contract_curve: np.ndarray,
                            load_actual: np.ndarray,
                            price_da_real: np.ndarray,
                            price_rt_real: np.ndarray,
                            upper_mask: np.ndarray,
                            lower_mask: np.ndarray) -> np.ndarray:
        threshold = self.rule.deviation_threshold
        mu = self.rule.mu
        n = len(q_final)
        recovery = np.zeros(n)

        for t in range(n):
            lt = load_actual[t]
            if lt <= 0:
                continue
            qt = q_final[t]
            spread_t = price_rt_real[t] - price_da_real[t]

            if upper_mask[t]:
                excess = qt - (1 + threshold) * lt
                if spread_t > 0:
                    recovery[t] = excess * spread_t * mu
                elif not self.rule.negative_spread_no_recovery:
                    recovery[t] = abs(excess * spread_t * mu)

            elif lower_mask[t]:
                deficit = (1 - threshold) * lt - qt
                if spread_t > 0:
                    recovery[t] = deficit * spread_t * mu
                elif not self.rule.negative_spread_no_recovery:
                    recovery[t] = abs(deficit * spread_t * mu)

        return recovery

    def settle_scenario(self,
                        q_final: np.ndarray,
                        contract_curve: np.ndarray,
                        scenarios: Dict[str, np.ndarray]) -> Dict[str, SettlementResult]:
        results = {}
        for scenario_name, scenario_data in scenarios.items():
            load_actual = scenario_data.get("load_actual")
            price_da_real = scenario_data.get("price_da_real")
            price_rt_real = scenario_data.get("price_rt_real")

            if load_actual is None or price_da_real is None or price_rt_real is None:
                logger.warning(f"Scenario '{scenario_name}' missing required fields, skipping")
                continue

            results[scenario_name] = self.settle_day(
                q_final, contract_curve, load_actual, price_da_real, price_rt_real
            )

        return results

    def batch_settle(self,
                     daily_data: Dict[str, Dict]) -> Dict[str, SettlementResult]:
        results = {}
        for date_str, data in daily_data.items():
            try:
                result = self.settle_day(
                    q_final=data["q_final"],
                    contract_curve=data["contract_curve"],
                    load_actual=data["load_actual"],
                    price_da_real=data["price_da_real"],
                    price_rt_real=data["price_rt_real"],
                )
                results[date_str] = result
            except Exception as e:
                logger.error(f"Settlement failed for {date_str}: {e}")

        return results

    def summarize(self, results: Dict[str, SettlementResult]) -> Dict[str, float]:
        if not results:
            return {}

        profits = [r.total_profit for r in results.values()]
        costs = [r.total_cost for r in results.values()]
        recoveries = [r.recovery_cost for r in results.values()]
        over_counts = [r.deviation_over_threshold_count for r in results.values()]

        summary = {
            "total_profit": sum(profits),
            "avg_daily_profit": np.mean(profits),
            "std_daily_profit": np.std(profits),
            "worst_day_profit": min(profits),
            "best_day_profit": max(profits),
            "total_recovery": sum(recoveries),
            "avg_recovery": np.mean(recoveries),
            "total_over_threshold": sum(over_counts),
            "avg_over_threshold_per_day": np.mean(over_counts),
            "profit_sharpe": np.mean(profits) / (np.std(profits) + 1e-9),
            "recovery_ratio_avg": np.mean([r.recovery_ratio for r in results.values()]),
            "n_days": len(results),
        }
        return summary
