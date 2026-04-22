import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from strategy.settlement_simulator import SettlementSimulator, SettlementRule, SettlementResult
from strategy.scenario_sampler import ScenarioSampler, Scenario, ScenarioConfig

logger = logging.getLogger(__name__)


class StrategyMode(Enum):
    NORMAL = "normal"
    HOLIDAY = "holiday"
    POST_HOLIDAY = "post_holiday"
    MISSING_INFO = "missing_info"


@dataclass
class StrategyInput:
    load_pred_p10: np.ndarray
    load_pred_p50: np.ndarray
    load_pred_p90: np.ndarray
    price_da_pred_p10: np.ndarray
    price_da_pred_p50: np.ndarray
    price_da_pred_p90: np.ndarray
    price_rt_pred_p10: np.ndarray
    price_rt_pred_p50: np.ndarray
    price_rt_pred_p90: np.ndarray
    spread_pred_p10: np.ndarray
    spread_pred_p50: np.ndarray
    spread_pred_p90: np.ndarray
    renewable_pred_p10: Optional[np.ndarray] = None
    renewable_pred_p50: Optional[np.ndarray] = None
    renewable_pred_p90: Optional[np.ndarray] = None
    is_holiday: bool = False
    is_post_holiday: bool = False
    holiday_day_index: int = 0
    info_completeness_level: float = 1.0
    mask_flag: int = 0
    missing_length: int = 0
    price_da_available: bool = True
    water_dayago_available: bool = True
    nomarket_dayago_available: bool = True
    line_dayago_available: bool = True
    contract_curve: Optional[np.ndarray] = None
    contract_total: float = 0.0
    contract_deviation_bounds: Optional[np.ndarray] = None


@dataclass
class CandidateCurve:
    name: str
    curve: np.ndarray
    description: str = ""
    objective_value: float = 0.0
    expected_cost: float = 0.0
    cost_variance: float = 0.0
    cvar_95: float = 0.0
    smooth_penalty: float = 0.0
    contract_deviation_penalty: float = 0.0


@dataclass
class StrategyOutput:
    q_final: np.ndarray
    mode: StrategyMode
    selected_curve_name: str
    candidate_curves: List[CandidateCurve]
    selection_reason: str
    referenced_quantiles: List[str]
    affected_by_missing_info: bool
    conservative_enabled: bool
    objective_values: Dict[str, float]


@dataclass
class StrategyParams:
    lambda_1: float = 0.5
    lambda_2: float = 1.0
    lambda_3: float = 0.1
    lambda_4: float = 0.3
    ramp_limit: float = 0.05
    max_daily_deviation_ratio: float = 0.15
    holiday_conservative_factor: float = 0.6
    post_holiday_ramp_rate: float = 0.7
    missing_info_shrink: float = 0.3
    spread_positive_aggression: float = 1.2
    spread_negative_shrink: float = 0.5
    smooth_window: int = 5


class StrategyEngine:
    def __init__(self,
                 params: StrategyParams = None,
                 settlement_rule: SettlementRule = None,
                 scenario_config: ScenarioConfig = None):
        self.params = params or StrategyParams()
        self.settlement = SettlementSimulator(rule=settlement_rule or SettlementRule())
        self.sampler = ScenarioSampler(config=scenario_config or ScenarioConfig())

    def generate_daily_strategy(self, inp: StrategyInput) -> StrategyOutput:
        mode = self._determine_mode(inp)
        logger.info(f"  [StrategyEngine] Mode: {mode.value}")

        candidates = self._generate_candidates(inp, mode)
        logger.info(f"    Generated {len(candidates)} candidate curves")

        scenarios = self._generate_scenarios(inp)
        logger.info(f"    Generated {len(scenarios)} scenarios")

        for candidate in candidates:
            obj = self._evaluate_candidate(candidate, scenarios, inp)
            candidate.objective_value = obj["total"]
            candidate.expected_cost = obj["expected_cost"]
            candidate.cost_variance = obj["cost_variance"]
            candidate.cvar_95 = obj["cvar_95"]
            candidate.smooth_penalty = obj["smooth_penalty"]
            candidate.contract_deviation_penalty = obj["contract_deviation_penalty"]

        best = self._select_best(candidates, mode)
        logger.info(f"    Selected: {best.name} (obj={best.objective_value:.4f})")

        reason = self._build_selection_reason(best, mode, inp)
        referenced_q = self._get_referenced_quantiles(best, mode)

        output = StrategyOutput(
            q_final=best.curve,
            mode=mode,
            selected_curve_name=best.name,
            candidate_curves=candidates,
            selection_reason=reason,
            referenced_quantiles=referenced_q,
            affected_by_missing_info=(mode == StrategyMode.MISSING_INFO),
            conservative_enabled=(mode in [StrategyMode.HOLIDAY, StrategyMode.MISSING_INFO]),
            objective_values={
                "total": best.objective_value,
                "expected_cost": best.expected_cost,
                "cost_variance": best.cost_variance,
                "cvar_95": best.cvar_95,
                "smooth_penalty": best.smooth_penalty,
                "contract_deviation_penalty": best.contract_deviation_penalty,
            },
        )
        return output

    def _determine_mode(self, inp: StrategyInput) -> StrategyMode:
        if inp.mask_flag == 1 or inp.info_completeness_level < 0.5:
            return StrategyMode.MISSING_INFO
        if inp.is_holiday:
            return StrategyMode.HOLIDAY
        if inp.is_post_holiday:
            return StrategyMode.POST_HOLIDAY
        return StrategyMode.NORMAL

    def _generate_candidates(self, inp: StrategyInput, mode: StrategyMode) -> List[CandidateCurve]:
        candidates = []
        contract = inp.contract_curve if inp.contract_curve is not None else inp.load_pred_p50.copy()

        candidates.append(CandidateCurve(
            name="contract_curve",
            curve=contract.copy(),
            description="原始合同锚定曲线",
        ))

        p50_curve = self._build_curve_with_offset(contract, inp.load_pred_p50, inp, mode, aggression=1.0)
        candidates.append(CandidateCurve(
            name="p50_curve",
            curve=p50_curve,
            description="中位数预测曲线",
        ))

        p30_curve = self._build_curve_with_offset(contract, inp.load_pred_p50, inp, mode, aggression=0.6)
        candidates.append(CandidateCurve(
            name="p30_curve",
            curve=p30_curve,
            description="偏保守曲线",
        ))

        p70_curve = self._build_curve_with_offset(contract, inp.load_pred_p50, inp, mode, aggression=1.4)
        candidates.append(CandidateCurve(
            name="p70_curve",
            curve=p70_curve,
            description="偏激进曲线",
        ))

        risk_curve = self._build_risk_adjusted_curve(contract, inp, mode)
        candidates.append(CandidateCurve(
            name="risk_adjusted_curve",
            curve=risk_curve,
            description="风险调整曲线",
        ))

        if mode == StrategyMode.HOLIDAY:
            holiday_curve = self._build_holiday_conservative_curve(contract, inp)
            candidates.append(CandidateCurve(
                name="holiday_conservative",
                curve=holiday_curve,
                description="节假日保守曲线",
            ))

        if mode == StrategyMode.POST_HOLIDAY:
            post_holiday_curve = self._build_post_holiday_curve(contract, inp)
            candidates.append(CandidateCurve(
                name="post_holiday_recovery",
                curve=post_holiday_curve,
                description="节后恢复曲线",
            ))

        if mode == StrategyMode.MISSING_INFO:
            missing_curve = self._build_missing_info_curve(contract, inp)
            candidates.append(CandidateCurve(
                name="missing_info_fallback",
                curve=missing_curve,
                description="缺失信息兜底曲线",
            ))

        spread_p50 = inp.spread_pred_p50
        if spread_p50 is not None and np.mean(spread_p50) > 0:
            high_price_curve = self._build_high_price_curve(contract, inp, mode)
            candidates.append(CandidateCurve(
                name="high_price_expected",
                curve=high_price_curve,
                description="高价预期曲线",
            ))

        if spread_p50 is not None and np.mean(spread_p50) < 0:
            low_price_curve = self._build_low_price_curve(contract, inp, mode)
            candidates.append(CandidateCurve(
                name="low_price_expected",
                curve=low_price_curve,
                description="低价预期曲线",
            ))

        for c in candidates:
            c.curve = self._apply_smooth_constraints(c.curve, contract)

        return candidates

    def _build_curve_with_offset(self, contract: np.ndarray, load_pred: np.ndarray,
                                  inp: StrategyInput, mode: StrategyMode,
                                  aggression: float = 1.0) -> np.ndarray:
        delta = load_pred - contract

        if mode == StrategyMode.NORMAL:
            spread_p50 = inp.spread_pred_p50
            if spread_p50 is not None:
                positive_mask = spread_p50 > 0
                negative_mask = spread_p50 <= 0
                delta[positive_mask] *= self.params.spread_positive_aggression
                delta[negative_mask] *= self.params.spread_negative_shrink

        elif mode == StrategyMode.HOLIDAY:
            delta *= self.params.holiday_conservative_factor

        elif mode == StrategyMode.POST_HOLIDAY:
            n = len(delta)
            ramp = np.linspace(
                self.params.holiday_conservative_factor,
                self.params.post_holiday_ramp_rate,
                n
            )
            delta *= ramp

        elif mode == StrategyMode.MISSING_INFO:
            delta *= self.params.missing_info_shrink

        delta *= aggression
        curve = contract + delta
        return curve

    def _build_risk_adjusted_curve(self, contract: np.ndarray,
                                    inp: StrategyInput, mode: StrategyMode) -> np.ndarray:
        delta_p50 = inp.load_pred_p50 - contract
        spread_p50 = inp.spread_pred_p50 if inp.spread_pred_p50 is not None else np.zeros(96)

        risk_factor = np.ones(96)
        spread_std = inp.spread_pred_p90 - inp.spread_pred_p10
        high_uncertainty = spread_std > np.median(spread_std)
        risk_factor[high_uncertainty] = 0.7

        negative_spread = spread_p50 < 0
        risk_factor[negative_spread] *= 0.6

        if mode == StrategyMode.HOLIDAY:
            risk_factor *= self.params.holiday_conservative_factor
        elif mode == StrategyMode.MISSING_INFO:
            risk_factor *= self.params.missing_info_shrink

        delta = delta_p50 * risk_factor
        return contract + delta

    def _build_holiday_conservative_curve(self, contract: np.ndarray,
                                           inp: StrategyInput) -> np.ndarray:
        blend = 0.3 * inp.load_pred_p30_like() + 0.7 * contract if hasattr(inp, 'load_pred_p30_like') else None
        if inp.load_pred_p10 is not None:
            conservative_pred = 0.4 * inp.load_pred_p10 + 0.6 * inp.load_pred_p50
        else:
            conservative_pred = inp.load_pred_p50

        delta = (conservative_pred - contract) * self.params.holiday_conservative_factor
        return contract + delta

    def _build_post_holiday_curve(self, contract: np.ndarray,
                                   inp: StrategyInput) -> np.ndarray:
        delta = inp.load_pred_p50 - contract
        n = len(delta)
        morning_mask = np.zeros(n, dtype=bool)
        morning_mask[:n // 2] = True

        ramp = np.ones(n)
        ramp[morning_mask] = self.params.holiday_conservative_factor
        ramp[~morning_mask] = np.linspace(
            self.params.holiday_conservative_factor,
            self.params.post_holiday_ramp_rate,
            n - n // 2
        )
        delta *= ramp
        return contract + delta

    def _build_missing_info_curve(self, contract: np.ndarray,
                                   inp: StrategyInput) -> np.ndarray:
        delta = inp.load_pred_p50 - contract
        delta *= self.params.missing_info_shrink

        if not inp.price_da_available:
            delta *= 0.5

        return contract + delta

    def _build_high_price_curve(self, contract: np.ndarray,
                                 inp: StrategyInput, mode: StrategyMode) -> np.ndarray:
        delta = inp.load_pred_p50 - contract
        spread_p50 = inp.spread_pred_p50
        positive_spread = np.maximum(spread_p50, 0)
        aggression = 1.0 + 0.3 * (positive_spread / (np.max(np.abs(spread_p50)) + 1e-9))

        if mode == StrategyMode.HOLIDAY:
            aggression *= self.params.holiday_conservative_factor
        elif mode == StrategyMode.MISSING_INFO:
            aggression *= self.params.missing_info_shrink

        delta *= aggression
        return contract + delta

    def _build_low_price_curve(self, contract: np.ndarray,
                                inp: StrategyInput, mode: StrategyMode) -> np.ndarray:
        delta = inp.load_pred_p50 - contract
        spread_p50 = inp.spread_pred_p50
        negative_spread = np.minimum(spread_p50, 0)
        shrink = 1.0 + 0.3 * (negative_spread / (np.max(np.abs(spread_p50)) + 1e-9))

        delta *= np.clip(shrink, 0.3, 1.0)
        return contract + delta

    def _apply_smooth_constraints(self, curve: np.ndarray,
                                   contract: np.ndarray) -> np.ndarray:
        n = len(curve)
        smoothed = curve.copy()

        for t in range(1, n):
            if contract[t] > 0:
                max_change = self.params.ramp_limit * contract[t]
            else:
                max_change = self.params.ramp_limit * np.mean(np.abs(contract)) if np.mean(np.abs(contract)) > 0 else 1.0

            diff = smoothed[t] - smoothed[t - 1]
            if abs(diff) > max_change:
                smoothed[t] = smoothed[t - 1] + np.sign(diff) * max_change

        if self.params.smooth_window > 1:
            kernel = np.ones(self.params.smooth_window) / self.params.smooth_window
            deviation = smoothed - contract
            smoothed_deviation = np.convolve(deviation, kernel, mode="same")
            smoothed = contract + smoothed_deviation

        if contract.sum() > 0:
            total_deviation = abs(smoothed.sum() - contract.sum()) / contract.sum()
            if total_deviation > self.params.max_daily_deviation_ratio:
                scale = contract.sum() * (1 + np.sign(smoothed.sum() - contract.sum()) * self.params.max_daily_deviation_ratio) / smoothed.sum()
                smoothed *= scale

        return smoothed

    def _generate_scenarios(self, inp: StrategyInput) -> List[Scenario]:
        return self.sampler.sample_scenarios(
            load_pred_p10=inp.load_pred_p10,
            load_pred_p50=inp.load_pred_p50,
            load_pred_p90=inp.load_pred_p90,
            price_da_pred_p10=inp.price_da_pred_p10,
            price_da_pred_p50=inp.price_da_pred_p50,
            price_da_pred_p90=inp.price_da_pred_p90,
            price_rt_pred_p10=inp.price_rt_pred_p10,
            price_rt_pred_p50=inp.price_rt_pred_p50,
            price_rt_pred_p90=inp.price_rt_pred_p90,
            spread_pred_p10=inp.spread_pred_p10,
            spread_pred_p50=inp.spread_pred_p50,
            spread_pred_p90=inp.spread_pred_p90,
            renewable_pred_p10=inp.renewable_pred_p10,
            renewable_pred_p50=inp.renewable_pred_p50,
            renewable_pred_p90=inp.renewable_pred_p90,
            is_holiday=inp.is_holiday,
            is_post_holiday=inp.is_post_holiday,
            info_completeness_level=inp.info_completeness_level,
            mask_flag=inp.mask_flag,
            missing_length=inp.missing_length,
        )

    def _evaluate_candidate(self, candidate: CandidateCurve,
                            scenarios: List[Scenario],
                            inp: StrategyInput) -> Dict[str, float]:
        contract = inp.contract_curve if inp.contract_curve is not None else inp.load_pred_p50.copy()
        costs = []
        weights = []

        for scenario in scenarios:
            result = self.settlement.settle_day(
                q_final=candidate.curve,
                contract_curve=contract,
                load_actual=scenario.load_actual,
                price_da_real=scenario.price_da_real,
                price_rt_real=scenario.price_rt_real,
            )
            costs.append(result.total_cost)
            weights.append(scenario.weight)

        costs = np.array(costs)
        weights = np.array(weights)
        weights = weights / weights.sum()

        expected_cost = np.sum(weights * costs)
        variance = np.sum(weights * (costs - expected_cost) ** 2)

        sorted_costs = np.sort(costs)
        n_tail = max(1, int(len(sorted_costs) * 0.05))
        cvar_95 = np.mean(sorted_costs[-n_tail:])

        smooth_penalty = self._compute_smooth_penalty(candidate.curve)
        contract_dev_penalty = self._compute_contract_deviation_penalty(candidate.curve, contract)

        total = (expected_cost
                 + self.params.lambda_1 * variance
                 + self.params.lambda_2 * cvar_95
                 + self.params.lambda_3 * smooth_penalty
                 + self.params.lambda_4 * contract_dev_penalty)

        return {
            "total": total,
            "expected_cost": expected_cost,
            "cost_variance": variance,
            "cvar_95": cvar_95,
            "smooth_penalty": smooth_penalty,
            "contract_deviation_penalty": contract_dev_penalty,
        }

    def _compute_smooth_penalty(self, curve: np.ndarray) -> float:
        if len(curve) < 2:
            return 0.0
        diffs = np.diff(curve)
        ramp_limit = self.params.ramp_limit * np.mean(np.abs(curve)) if np.mean(np.abs(curve)) > 0 else 1.0
        violations = np.maximum(np.abs(diffs) - ramp_limit, 0)
        return np.sum(violations ** 2)

    def _compute_contract_deviation_penalty(self, curve: np.ndarray,
                                             contract: np.ndarray) -> float:
        if contract is None or len(contract) == 0:
            return 0.0
        deviation = curve - contract
        relative_dev = deviation / (contract + 1e-9)
        threshold = 0.10
        over = np.maximum(np.abs(relative_dev) - threshold, 0)
        return np.sum(over ** 2)

    def _select_best(self, candidates: List[CandidateCurve],
                     mode: StrategyMode) -> CandidateCurve:
        if not candidates:
            raise ValueError("No candidate curves available")

        for c in candidates:
            if not np.isfinite(c.objective_value):
                c.objective_value = float("inf")

        sorted_candidates = sorted(candidates, key=lambda c: c.objective_value)
        return sorted_candidates[0]

    def _build_selection_reason(self, best: CandidateCurve,
                                mode: StrategyMode,
                                inp: StrategyInput) -> str:
        parts = [f"选中曲线: {best.name}"]
        parts.append(f"策略模式: {mode.value}")
        parts.append(f"目标函数值: {best.objective_value:.4f}")
        parts.append(f"期望成本: {best.expected_cost:.4f}")
        parts.append(f"CVaR95: {best.cvar_95:.4f}")

        if mode == StrategyMode.HOLIDAY:
            parts.append("节假日模式: 启用保守策略")
        elif mode == StrategyMode.POST_HOLIDAY:
            parts.append("节后模式: 上午保守, 下午逐步恢复")
        elif mode == StrategyMode.MISSING_INFO:
            parts.append(f"缺失信息模式: 完整度={inp.info_completeness_level:.2f}")

        return "; ".join(parts)

    def _get_referenced_quantiles(self, best: CandidateCurve,
                                  mode: StrategyMode) -> List[str]:
        quantiles = ["P50"]
        if "p30" in best.name or "conservative" in best.name or "holiday" in best.name:
            quantiles = ["P10", "P50"]
        elif "p70" in best.name or "aggressive" in best.name or "high_price" in best.name:
            quantiles = ["P50", "P90"]
        elif "risk" in best.name:
            quantiles = ["P10", "P50", "P90"]
        elif "contract" in best.name:
            quantiles = []
        elif "missing" in best.name:
            quantiles = ["P50"]
        return quantiles

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
                logger.info(f"  [StrategyEngine] Updated {key} = {value}")
