import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod

from strategy.strategy_engine import StrategyInput, StrategyMode

logger = logging.getLogger(__name__)


class BaselineStrategy(ABC):
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def generate_curve(self, inp: StrategyInput) -> np.ndarray:
        pass

    def __repr__(self):
        return f"BaselineStrategy({self.name})"


class ContractCurveBaseline(BaselineStrategy):
    def __init__(self):
        super().__init__("contract_curve", "直接使用合同曲线申报")

    def generate_curve(self, inp: StrategyInput) -> np.ndarray:
        if inp.contract_curve is not None:
            return inp.contract_curve.copy()
        return inp.load_pred_p50.copy()


class P50Baseline(BaselineStrategy):
    def __init__(self):
        super().__init__("p50", "使用P50预测值申报")

    def generate_curve(self, inp: StrategyInput) -> np.ndarray:
        return inp.load_pred_p50.copy()


class P50FixedOffsetBaseline(BaselineStrategy):
    def __init__(self, offset_ratio: float = 0.05):
        super().__init__(f"p50_fixed_offset_{offset_ratio}", f"P50 + 固定偏移{offset_ratio*100:.0f}%")
        self.offset_ratio = offset_ratio

    def generate_curve(self, inp: StrategyInput) -> np.ndarray:
        base = inp.load_pred_p50.copy()
        offset = base * self.offset_ratio
        spread_p50 = inp.spread_pred_p50
        if spread_p50 is not None:
            direction = np.sign(np.mean(spread_p50))
            return base + direction * offset
        return base + offset


class ConservativeBaseline(BaselineStrategy):
    def __init__(self, conservative_quantile_weight: float = 0.7):
        super().__init__("conservative", "保守策略: 偏向P10和合同曲线")
        self.conservative_quantile_weight = conservative_quantile_weight

    def generate_curve(self, inp: StrategyInput) -> np.ndarray:
        contract = inp.contract_curve if inp.contract_curve is not None else inp.load_pred_p50.copy()
        p10 = inp.load_pred_p10
        p50 = inp.load_pred_p50

        blended = self.conservative_quantile_weight * p10 + (1 - self.conservative_quantile_weight) * p50
        curve = 0.5 * contract + 0.5 * blended

        if inp.is_holiday:
            curve = 0.6 * contract + 0.4 * blended
        if inp.mask_flag == 1:
            curve = 0.8 * contract + 0.2 * blended

        return curve


class AggressiveBaseline(BaselineStrategy):
    def __init__(self, aggression_factor: float = 1.3):
        super().__init__("aggressive", "激进策略: 偏向P90和价差方向")
        self.aggression_factor = aggression_factor

    def generate_curve(self, inp: StrategyInput) -> np.ndarray:
        contract = inp.contract_curve if inp.contract_curve is not None else inp.load_pred_p50.copy()
        p50 = inp.load_pred_p50
        p90 = inp.load_pred_p90

        delta = p50 - contract
        spread_p50 = inp.spread_pred_p50

        if spread_p50 is not None:
            positive_mask = spread_p50 > 0
            delta[positive_mask] *= self.aggression_factor
            delta[~positive_mask] *= 0.8

        curve = contract + delta

        if inp.is_holiday or inp.mask_flag == 1:
            curve = 0.7 * contract + 0.3 * p50

        return curve


class ScenarioOptimizedBaseline(BaselineStrategy):
    def __init__(self, n_sample_scenarios: int = 50):
        super().__init__("scenario_optimized", "场景优化策略: 基于场景评估选择最优偏移")
        self.n_sample_scenarios = n_sample_scenarios

    def generate_curve(self, inp: StrategyInput) -> np.ndarray:
        contract = inp.contract_curve if inp.contract_curve is not None else inp.load_pred_p50.copy()
        p50 = inp.load_pred_p50

        offsets = np.linspace(-0.1, 0.1, 11)
        best_offset = 0.0
        best_score = float("inf")

        spread_p50 = inp.spread_pred_p50
        if spread_p50 is None:
            return p50.copy()

        for offset_ratio in offsets:
            candidate = contract + offset_ratio * contract
            score = self._quick_evaluate(candidate, inp)
            if score < best_score:
                best_score = score
                best_offset = offset_ratio

        return contract + best_offset * contract

    def _quick_evaluate(self, curve: np.ndarray, inp: StrategyInput) -> float:
        spread_p50 = inp.spread_pred_p50
        contract = inp.contract_curve if inp.contract_curve is not None else inp.load_pred_p50.copy()

        deviation = np.abs(curve - inp.load_pred_p50) / (inp.load_pred_p50 + 1e-9)
        over_threshold = np.sum(deviation > 0.10)

        spread_benefit = np.sum((curve - contract) * spread_p50) if spread_p50 is not None else 0

        smooth_penalty = np.sum(np.diff(curve) ** 2)

        score = -spread_benefit + 10 * over_threshold + 0.001 * smooth_penalty
        return score


class MoEBaseline(BaselineStrategy):
    def __init__(self):
        super().__init__("moe", "MoE策略: 按日况选择不同专家策略")

    def generate_curve(self, inp: StrategyInput) -> np.ndarray:
        mode = self._determine_mode(inp)
        contract = inp.contract_curve if inp.contract_curve is not None else inp.load_pred_p50.copy()

        if mode == "holiday":
            return self._holiday_expert(inp, contract)
        elif mode == "post_holiday":
            return self._post_holiday_expert(inp, contract)
        elif mode == "missing":
            return self._missing_info_expert(inp, contract)
        else:
            return self._normal_expert(inp, contract)

    def _determine_mode(self, inp: StrategyInput) -> str:
        if inp.is_holiday:
            return "holiday"
        if inp.is_post_holiday:
            return "post_holiday"
        if inp.mask_flag == 1 or inp.info_completeness_level < 0.5:
            return "missing"
        return "normal"

    def _normal_expert(self, inp: StrategyInput, contract: np.ndarray) -> np.ndarray:
        delta = inp.load_pred_p50 - contract
        spread_p50 = inp.spread_pred_p50
        if spread_p50 is not None:
            positive = spread_p50 > 0
            delta[positive] *= 1.1
            delta[~positive] *= 0.7
        return contract + delta

    def _holiday_expert(self, inp: StrategyInput, contract: np.ndarray) -> np.ndarray:
        blended = 0.4 * inp.load_pred_p10 + 0.6 * inp.load_pred_p50
        delta = (blended - contract) * 0.5
        return contract + delta

    def _post_holiday_expert(self, inp: StrategyInput, contract: np.ndarray) -> np.ndarray:
        delta = inp.load_pred_p50 - contract
        n = len(delta)
        ramp = np.linspace(0.5, 0.9, n)
        delta *= ramp
        return contract + delta

    def _missing_info_expert(self, inp: StrategyInput, contract: np.ndarray) -> np.ndarray:
        delta = (inp.load_pred_p50 - contract) * 0.2
        return contract + delta


class StrategyBaseline:
    def __init__(self):
        self.baselines: Dict[str, BaselineStrategy] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.baselines["contract_curve"] = ContractCurveBaseline()
        self.baselines["p50"] = P50Baseline()
        self.baselines["p50_fixed_offset"] = P50FixedOffsetBaseline()
        self.baselines["conservative"] = ConservativeBaseline()
        self.baselines["aggressive"] = AggressiveBaseline()
        self.baselines["scenario_optimized"] = ScenarioOptimizedBaseline()
        self.baselines["moe"] = MoEBaseline()

    def add_baseline(self, strategy: BaselineStrategy):
        self.baselines[strategy.name] = strategy
        logger.info(f"  [StrategyBaseline] Added baseline: {strategy.name}")

    def generate_all(self, inp: StrategyInput) -> Dict[str, np.ndarray]:
        results = {}
        for name, strategy in self.baselines.items():
            try:
                curve = strategy.generate_curve(inp)
                results[name] = curve
            except Exception as e:
                logger.warning(f"  [StrategyBaseline] {name} failed: {e}")
        return results

    def generate_single(self, name: str, inp: StrategyInput) -> Optional[np.ndarray]:
        if name not in self.baselines:
            logger.warning(f"  [StrategyBaseline] Unknown baseline: {name}")
            return None
        return self.baselines[name].generate_curve(inp)

    def list_baselines(self) -> List[str]:
        return list(self.baselines.keys())
