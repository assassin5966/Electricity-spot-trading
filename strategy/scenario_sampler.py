import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    n_scenarios: int = 100
    min_scenarios: int = 50
    include_extreme_price: bool = True
    include_missing_continuation: bool = True
    extreme_price_quantile: float = 0.95
    missing_continuation_days: int = 5
    correlation_method: str = "copula"
    bootstrap_residuals: bool = True
    random_seed: Optional[int] = None


@dataclass
class Scenario:
    name: str
    load_actual: np.ndarray
    price_da_real: np.ndarray
    price_rt_real: np.ndarray
    spread: np.ndarray
    renewable_actual: np.ndarray
    weight: float = 1.0
    is_extreme: bool = False
    is_holiday_specific: bool = False
    is_missing_scenario: bool = False


class ScenarioSampler:
    def __init__(self, config: ScenarioConfig = None):
        self.config = config or ScenarioConfig()
        self._residual_cache = {}
        self._corr_matrix = None

    def fit(self,
            historical_load: np.ndarray,
            historical_price_da: np.ndarray,
            historical_price_rt: np.ndarray,
            historical_renewable: np.ndarray = None):
        logger.info("  [ScenarioSampler] Fitting on historical data ...")

        self._historical = {
            "load": historical_load,
            "price_da": historical_price_da,
            "price_rt": historical_price_rt,
        }
        if historical_renewable is not None:
            self._historical["renewable"] = historical_renewable

        self._compute_residuals()
        self._compute_correlation()
        logger.info(f"    Residuals computed, correlation matrix shape: {self._corr_matrix.shape}")

    def _compute_residuals(self):
        for key, values in self._historical.items():
            if len(values) < 10:
                self._residual_cache[key] = np.zeros(len(values))
                continue
            rolling_mean = pd.Series(values).rolling(96, min_periods=1).mean().values
            residuals = values - rolling_mean
            self._residual_cache[key] = residuals

    def _compute_correlation(self):
        keys = list(self._residual_cache.keys())
        n = len(keys)
        self._corr_matrix = np.eye(n)
        residuals_aligned = []
        min_len = min(len(self._residual_cache[k]) for k in keys)
        for k in keys:
            residuals_aligned.append(self._residual_cache[k][:min_len])

        if min_len > 10:
            corr_df = pd.DataFrame({k: r for k, r in zip(keys, residuals_aligned)})
            self._corr_matrix = corr_df.corr().values

    def sample_scenarios(self,
                         load_pred_p10: np.ndarray,
                         load_pred_p50: np.ndarray,
                         load_pred_p90: np.ndarray,
                         price_da_pred_p10: np.ndarray,
                         price_da_pred_p50: np.ndarray,
                         price_da_pred_p90: np.ndarray,
                         price_rt_pred_p10: np.ndarray,
                         price_rt_pred_p50: np.ndarray,
                         price_rt_pred_p90: np.ndarray,
                         spread_pred_p10: np.ndarray,
                         spread_pred_p50: np.ndarray,
                         spread_pred_p90: np.ndarray,
                         renewable_pred_p10: np.ndarray = None,
                         renewable_pred_p50: np.ndarray = None,
                         renewable_pred_p90: np.ndarray = None,
                         is_holiday: bool = False,
                         is_post_holiday: bool = False,
                         info_completeness_level: float = 1.0,
                         mask_flag: int = 0,
                         missing_length: int = 0) -> List[Scenario]:
        rng = np.random.RandomState(self.config.random_seed)
        n = len(load_pred_p50)
        n_scenarios = self.config.n_scenarios

        scenarios = []

        n_quantile = int(n_scenarios * 0.4)
        n_bootstrap = int(n_scenarios * 0.3)
        n_correlated = int(n_scenarios * 0.2)
        n_extreme = n_scenarios - n_quantile - n_bootstrap - n_correlated

        for i in range(n_quantile):
            alpha = rng.uniform(0.05, 0.95)
            load_s = self._interpolate_quantile(load_pred_p10, load_pred_p50, load_pred_p90, alpha)
            pda_s = self._interpolate_quantile(price_da_pred_p10, price_da_pred_p50, price_da_pred_p90, alpha)
            prt_s = self._interpolate_quantile(price_rt_pred_p10, price_rt_pred_p50, price_rt_pred_p90, alpha)
            spread_s = self._interpolate_quantile(spread_pred_p10, spread_pred_p50, spread_pred_p90, alpha)

            renewable_s = None
            if renewable_pred_p50 is not None:
                renewable_s = self._interpolate_quantile(
                    renewable_pred_p10, renewable_pred_p50, renewable_pred_p90, alpha
                )

            scenarios.append(Scenario(
                name=f"quantile_{i}",
                load_actual=load_s,
                price_da_real=pda_s,
                price_rt_real=prt_s,
                spread=spread_s,
                renewable_actual=renewable_s if renewable_s is not None else np.zeros(n),
                weight=1.0 / n_scenarios,
            ))

        for i in range(n_bootstrap):
            load_s = self._bootstrap_sample(load_pred_p50, "load", rng)
            pda_s = self._bootstrap_sample(price_da_pred_p50, "price_da", rng)
            prt_s = self._bootstrap_sample(price_rt_pred_p50, "price_rt", rng)
            spread_s = prt_s - pda_s

            renewable_s = None
            if renewable_pred_p50 is not None:
                renewable_s = self._bootstrap_sample(renewable_pred_p50, "renewable", rng)

            scenarios.append(Scenario(
                name=f"bootstrap_{i}",
                load_actual=load_s,
                price_da_real=pda_s,
                price_rt_real=prt_s,
                spread=spread_s,
                renewable_actual=renewable_s if renewable_s is not None else np.zeros(n),
                weight=1.0 / n_scenarios,
            ))

        for i in range(n_correlated):
            correlated = self._correlated_sample(
                load_pred_p50, price_da_pred_p50, price_rt_pred_p50, rng
            )
            load_s, pda_s, prt_s = correlated
            spread_s = prt_s - pda_s

            renewable_s = np.zeros(n)
            if renewable_pred_p50 is not None:
                renewable_s = renewable_pred_p50.copy()

            scenarios.append(Scenario(
                name=f"correlated_{i}",
                load_actual=load_s,
                price_da_real=pda_s,
                price_rt_real=prt_s,
                spread=spread_s,
                renewable_actual=renewable_s,
                weight=1.0 / n_scenarios,
            ))

        if self.config.include_extreme_price and n_extreme > 0:
            for i in range(n_extreme):
                load_s = load_pred_p50.copy()
                pda_s = price_da_pred_p90.copy() if rng.random() > 0.5 else price_da_pred_p10.copy()
                prt_s = price_rt_pred_p90.copy() if rng.random() > 0.5 else price_rt_pred_p10.copy()
                spread_s = prt_s - pda_s

                renewable_s = np.zeros(n)
                if renewable_pred_p50 is not None:
                    renewable_s = renewable_pred_p50.copy()

                scenarios.append(Scenario(
                    name=f"extreme_price_{i}",
                    load_actual=load_s,
                    price_da_real=pda_s,
                    price_rt_real=prt_s,
                    spread=spread_s,
                    renewable_actual=renewable_s,
                    weight=1.0 / n_scenarios,
                    is_extreme=True,
                ))

        if is_holiday:
            scenarios = self._adjust_for_holiday(scenarios, rng)
        if is_post_holiday:
            scenarios = self._adjust_for_post_holiday(scenarios, rng)
        if mask_flag == 1 or info_completeness_level < 1.0:
            scenarios = self._adjust_for_missing(scenarios, info_completeness_level, missing_length, rng)

        if self.config.include_missing_continuation:
            missing_scenarios = self._generate_missing_continuation_scenarios(
                load_pred_p50, price_da_pred_p50, price_rt_pred_p50,
                spread_pred_p50, renewable_pred_p50, rng
            )
            scenarios.extend(missing_scenarios)

        total_weight = sum(s.weight for s in scenarios)
        for s in scenarios:
            s.weight /= total_weight

        logger.info(f"  [ScenarioSampler] Generated {len(scenarios)} scenarios "
                    f"(holiday={is_holiday}, post_holiday={is_post_holiday}, "
                    f"missing={mask_flag})")
        return scenarios

    def _interpolate_quantile(self, p10: np.ndarray, p50: np.ndarray,
                               p90: np.ndarray, alpha: float) -> np.ndarray:
        if alpha <= 0.1:
            return p10.copy()
        elif alpha <= 0.5:
            t = (alpha - 0.1) / 0.4
            return p10 + t * (p50 - p10)
        elif alpha <= 0.9:
            t = (alpha - 0.5) / 0.4
            return p50 + t * (p90 - p50)
        else:
            return p90.copy()

    def _bootstrap_sample(self, pred_p50: np.ndarray, key: str,
                          rng: np.random.RandomState) -> np.ndarray:
        n = len(pred_p50)
        if key in self._residual_cache and len(self._residual_cache[key]) > 0:
            residuals = self._residual_cache[key]
            indices = rng.choice(len(residuals), size=n, replace=True)
            sampled_residual = residuals[indices]
            scale = rng.uniform(0.8, 1.2)
            return pred_p50 + sampled_residual * scale
        else:
            noise_std = np.std(pred_p50) * 0.05 if np.std(pred_p50) > 0 else 1.0
            return pred_p50 + rng.normal(0, noise_std, n)

    def _correlated_sample(self,
                           load_p50: np.ndarray,
                           price_da_p50: np.ndarray,
                           price_rt_p50: np.ndarray,
                           rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(load_p50)
        n_vars = 3

        if self._corr_matrix is not None and self._corr_matrix.shape[0] >= n_vars:
            corr = self._corr_matrix[:n_vars, :n_vars]
        else:
            corr = np.array([
                [1.0, 0.3, 0.2],
                [0.3, 1.0, 0.7],
                [0.2, 0.7, 1.0],
            ])

        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            L = np.eye(n_vars)

        z = rng.randn(n_vars, n)
        correlated = L @ z

        load_std = np.std(load_p50) * 0.1 if np.std(load_p50) > 0 else 1.0
        pda_std = np.std(price_da_p50) * 0.15 if np.std(price_da_p50) > 0 else 1.0
        prt_std = np.std(price_rt_p50) * 0.15 if np.std(price_rt_p50) > 0 else 1.0

        load_s = load_p50 + correlated[0] * load_std
        pda_s = price_da_p50 + correlated[1] * pda_std
        prt_s = price_rt_p50 + correlated[2] * prt_std

        return load_s, pda_s, prt_s

    def _adjust_for_holiday(self, scenarios: List[Scenario],
                            rng: np.random.RandomState) -> List[Scenario]:
        for s in scenarios:
            s.is_holiday_specific = True
            shrink = rng.uniform(0.7, 0.9)
            s.load_actual *= shrink
            s.weight *= 1.2
        return scenarios

    def _adjust_for_post_holiday(self, scenarios: List[Scenario],
                                  rng: np.random.RandomState) -> List[Scenario]:
        for s in scenarios:
            ramp = np.linspace(0.85, 1.0, len(s.load_actual))
            s.load_actual *= ramp
        return scenarios

    def _adjust_for_missing(self, scenarios: List[Scenario],
                            info_level: float, missing_length: int,
                            rng: np.random.RandomState) -> List[Scenario]:
        for s in scenarios:
            s.is_missing_scenario = True
            uncertainty_factor = 1.0 + (1.0 - info_level) * 0.3
            noise = rng.normal(0, 0.05 * uncertainty_factor, len(s.load_actual))
            s.load_actual *= (1 + noise)
        return scenarios

    def _generate_missing_continuation_scenarios(self,
                                                  load_p50, price_da_p50, price_rt_p50,
                                                  spread_p50, renewable_p50,
                                                  rng: np.random.RandomState) -> List[Scenario]:
        n = len(load_p50)
        scenarios = []
        n_missing = 3

        for i in range(n_missing):
            load_s = load_p50.copy() * rng.uniform(0.9, 1.1)
            pda_s = price_da_p50.copy() * rng.uniform(0.85, 1.15)
            prt_s = price_rt_p50.copy() * rng.uniform(0.85, 1.15)
            spread_s = prt_s - pda_s
            renewable_s = renewable_p50.copy() if renewable_p50 is not None else np.zeros(n)

            scenarios.append(Scenario(
                name=f"missing_continuation_{i}",
                load_actual=load_s,
                price_da_real=pda_s,
                price_rt_real=prt_s,
                spread=spread_s,
                renewable_actual=renewable_s,
                weight=0.02,
                is_missing_scenario=True,
            ))

        return scenarios

    def scenarios_to_dict(self, scenarios: List[Scenario]) -> Dict[str, Dict[str, np.ndarray]]:
        result = {}
        for s in scenarios:
            result[s.name] = {
                "load_actual": s.load_actual,
                "price_da_real": s.price_da_real,
                "price_rt_real": s.price_rt_real,
                "spread": s.spread,
                "renewable_actual": s.renewable_actual,
                "weight": s.weight,
                "is_extreme": s.is_extreme,
                "is_holiday_specific": s.is_holiday_specific,
                "is_missing_scenario": s.is_missing_scenario,
            }
        return result
