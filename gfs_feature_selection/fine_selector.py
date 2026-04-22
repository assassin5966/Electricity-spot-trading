import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FineSelectionConfig:
    correlation_threshold: float = 0.95
    importance_zero_threshold: float = 0.001
    gbdt_top_k: int = 60
    shap_top_k: int = 60
    permutation_top_k: int = 60
    n_redundancy_check_samples: int = 10000
    target_wind: str = "WIND_REAL"
    target_solar: str = "LIGHT_REAL"
    target_renewable: str = "GREEN_REAL"


@dataclass
class FeatureImportanceResult:
    feature_name: str
    gbdt_importance: float = 0.0
    shap_importance: float = 0.0
    permutation_importance: float = 0.0
    combined_score: float = 0.0
    is_redundant: bool = False
    redundancy_with: str = ""
    final_retained: bool = True


class GFSFineSelector:
    def __init__(self, config: FineSelectionConfig = None):
        self.config = config or FineSelectionConfig()
        self.importance_results: Dict[str, FeatureImportanceResult] = {}
        self.wind_features: List[str] = []
        self.solar_features: List[str] = []
        self.renewable_features: List[str] = []

    def run_fine_selection(self,
                           df: pd.DataFrame,
                           candidate_features: List[str],
                           target_wind: str = None,
                           target_solar: str = None,
                           target_renewable: str = None) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        logger.info("=" * 60)
        logger.info("GFS FINE FEATURE SELECTION")
        logger.info("=" * 60)

        target_wind = target_wind or self.config.target_wind
        target_solar = target_solar or self.config.target_solar
        target_renewable = target_renewable or self.config.target_renewable

        available_features = [f for f in candidate_features if f in df.columns]
        logger.info(f"  Candidate features available in data: {len(available_features)}/{len(candidate_features)}")

        logger.info("  Step 1: GBDT feature importance ...")
        gbdt_imp = self._compute_gbdt_importance(df, available_features, target_renewable)

        logger.info("  Step 2: SHAP importance ...")
        shap_imp = self._compute_shap_importance(df, available_features, target_renewable)

        logger.info("  Step 3: Permutation importance ...")
        perm_imp = self._compute_permutation_importance(df, available_features, target_renewable)

        logger.info("  Step 4: Combining importance scores ...")
        self._combine_importance(available_features, gbdt_imp, shap_imp, perm_imp)

        logger.info("  Step 5: Redundancy filtering ...")
        self._filter_redundancy(df, available_features)

        logger.info("  Step 6: Model-specific selection ...")
        feature_sets = self._model_specific_selection(df, available_features, target_wind, target_solar, target_renewable)

        result_df = self._build_result_dataframe()

        n_final = sum(1 for r in self.importance_results.values() if r.final_retained)
        logger.info(f"  Final retained features: {n_final}")
        logger.info(f"  Wind features: {len(self.wind_features)}")
        logger.info(f"  Solar features: {len(self.solar_features)}")
        logger.info(f"  Renewable features: {len(self.renewable_features)}")

        return result_df, feature_sets

    def _compute_gbdt_importance(self, df: pd.DataFrame, features: List[str],
                                 target: str) -> Dict[str, float]:
        if target not in df.columns:
            logger.warning(f"    Target '{target}' not found, using dummy importance")
            return {f: 0.0 for f in features}

        valid_mask = df[target].notna()
        for f in features:
            valid_mask = valid_mask & df[f].notna()

        X = df.loc[valid_mask, features].fillna(0)
        y = df.loc[valid_mask, target]

        if len(X) < 100:
            logger.warning(f"    Not enough valid samples ({len(X)}), skipping GBDT importance")
            return {f: 0.0 for f in features}

        n_sample = min(self.config.n_redundancy_check_samples, len(X))
        if n_sample < len(X):
            sample_idx = np.random.choice(len(X), n_sample, replace=False)
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]

        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                verbose=-1, random_state=42,
            )
            model.fit(X, y)
            importances = model.feature_importances_
            result = dict(zip(features, importances))
            logger.info(f"    GBDT importance computed for {len(features)} features")
            return result
        except ImportError:
            logger.warning("    lightgbm not available, using correlation-based importance")
            return self._correlation_importance(X, y, features)

    def _correlation_importance(self, X: pd.DataFrame, y: pd.Series,
                                features: List[str]) -> Dict[str, float]:
        result = {}
        for f in features:
            if X[f].std() > 0:
                corr = np.abs(np.corrcoef(X[f], y)[0, 1])
                result[f] = corr if np.isfinite(corr) else 0.0
            else:
                result[f] = 0.0
        return result

    def _compute_shap_importance(self, df: pd.DataFrame, features: List[str],
                                 target: str) -> Dict[str, float]:
        try:
            import shap
        except ImportError:
            logger.warning("    shap not available, using GBDT importance as proxy")
            return self._compute_gbdt_importance(df, features, target)

        if target not in df.columns:
            return {f: 0.0 for f in features}

        valid_mask = df[target].notna()
        for f in features:
            valid_mask = valid_mask & df[f].notna()

        X = df.loc[valid_mask, features].fillna(0)
        y = df.loc[valid_mask, target]

        if len(X) < 100:
            return {f: 0.0 for f in features}

        n_sample = min(5000, len(X))
        if n_sample < len(X):
            sample_idx = np.random.choice(len(X), n_sample, replace=False)
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]

        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                verbose=-1, random_state=42,
            )
            model.fit(X, y)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            result = dict(zip(features, mean_abs_shap))
            logger.info(f"    SHAP importance computed for {len(features)} features")
            return result
        except Exception as e:
            logger.warning(f"    SHAP computation failed: {e}, using GBDT as proxy")
            return self._compute_gbdt_importance(df, features, target)

    def _compute_permutation_importance(self, df: pd.DataFrame, features: List[str],
                                        target: str) -> Dict[str, float]:
        if target not in df.columns:
            return {f: 0.0 for f in features}

        valid_mask = df[target].notna()
        for f in features:
            valid_mask = valid_mask & df[f].notna()

        X = df.loc[valid_mask, features].fillna(0)
        y = df.loc[valid_mask, target]

        if len(X) < 100:
            return {f: 0.0 for f in features}

        n_sample = min(5000, len(X))
        if n_sample < len(X):
            sample_idx = np.random.choice(len(X), n_sample, replace=False)
            X = X.iloc[sample_idx]
            y = y.iloc[sample_idx]

        try:
            from sklearn.inspection import permutation_importance as sk_perm_imp
            import lightgbm as lgb

            model = lgb.LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                verbose=-1, random_state=42,
            )
            model.fit(X, y)

            result_sk = sk_perm_imp(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
            result = dict(zip(features, result_sk.importances_mean))
            logger.info(f"    Permutation importance computed for {len(features)} features")
            return result
        except ImportError:
            logger.warning("    sklearn not available, using manual permutation")
            return self._manual_permutation_importance(X, y, features)

    def _manual_permutation_importance(self, X: pd.DataFrame, y: pd.Series,
                                       features: List[str]) -> Dict[str, float]:
        try:
            import lightgbm as lgb
        except ImportError:
            return {f: 0.0 for f in features}

        model = lgb.LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            verbose=-1, random_state=42,
        )
        model.fit(X, y)
        baseline_score = np.mean((model.predict(X) - y.values) ** 2)

        result = {}
        for f in features:
            X_perm = X.copy()
            X_perm[f] = np.random.permutation(X_perm[f].values)
            perm_score = np.mean((model.predict(X_perm) - y.values) ** 2)
            result[f] = perm_score - baseline_score

        return result

    def _combine_importance(self, features: List[str],
                            gbdt_imp: Dict[str, float],
                            shap_imp: Dict[str, float],
                            perm_imp: Dict[str, float]):
        for f in features:
            gbdt_val = gbdt_imp.get(f, 0.0)
            shap_val = shap_imp.get(f, 0.0)
            perm_val = perm_imp.get(f, 0.0)

            gbdt_rank = self._rank_value(gbdt_val, gbdt_imp)
            shap_rank = self._rank_value(shap_val, shap_imp)
            perm_rank = self._rank_value(perm_val, perm_imp)

            combined = 0.4 * gbdt_rank + 0.35 * shap_rank + 0.25 * perm_rank

            self.importance_results[f] = FeatureImportanceResult(
                feature_name=f,
                gbdt_importance=gbdt_val,
                shap_importance=shap_val,
                permutation_importance=perm_val,
                combined_score=combined,
            )

    def _rank_value(self, value: float, all_values: Dict[str, float]) -> float:
        vals = sorted(all_values.values(), reverse=True)
        if not vals or vals[0] == vals[-1]:
            return 0.5
        return (value - vals[-1]) / (vals[0] - vals[-1])

    def _filter_redundancy(self, df: pd.DataFrame, features: List[str]):
        n_sample = min(self.config.n_redundancy_check_samples, len(df))
        if n_sample < len(df):
            sample_idx = np.random.choice(len(df), n_sample, replace=False)
            X = df[features].iloc[sample_idx].fillna(0)
        else:
            X = df[features].fillna(0)

        corr_matrix = X.corr().abs()
        sorted_features = sorted(features, key=lambda f: self.importance_results.get(f, FeatureImportanceResult()).combined_score, reverse=True)

        redundant_set = set()
        for i, f1 in enumerate(sorted_features):
            if f1 in redundant_set:
                continue
            for f2 in sorted_features[i + 1:]:
                if f2 in redundant_set:
                    continue
                if f1 in corr_matrix.columns and f2 in corr_matrix.columns:
                    corr_val = corr_matrix.loc[f1, f2]
                    if pd.notna(corr_val) and corr_val > self.config.correlation_threshold:
                        redundant_set.add(f2)
                        if f2 in self.importance_results:
                            self.importance_results[f2].is_redundant = True
                            self.importance_results[f2].redundancy_with = f1

        for f in features:
            if f in self.importance_results:
                imp = self.importance_results[f]
                if imp.is_redundant:
                    imp.final_retained = False
                elif imp.combined_score < self.config.importance_zero_threshold:
                    imp.final_retained = False

        n_redundant = sum(1 for r in self.importance_results.values() if r.is_redundant)
        n_zero_imp = sum(1 for r in self.importance_results.values() if not r.is_redundant and r.combined_score < self.config.importance_zero_threshold)
        logger.info(f"    Redundant features (corr > {self.config.correlation_threshold}): {n_redundant}")
        logger.info(f"    Zero importance features: {n_zero_imp}")

    def _model_specific_selection(self, df: pd.DataFrame, features: List[str],
                                  target_wind: str, target_solar: str,
                                  target_renewable: str) -> Dict[str, List[str]]:
        retained = [f for f in features if f in self.importance_results and self.importance_results[f].final_retained]

        wind_features = self._select_for_target(df, retained, target_wind, "wind")
        solar_features = self._select_for_target(df, retained, target_solar, "solar")
        renewable_features = list(set(wind_features + solar_features))

        renewable_features = sorted(set(renewable_features),
                                     key=lambda f: self.importance_results.get(f, FeatureImportanceResult()).combined_score,
                                     reverse=True)[:self.config.gbdt_top_k]

        self.wind_features = wind_features
        self.solar_features = solar_features
        self.renewable_features = renewable_features

        return {
            "wind": wind_features,
            "solar": solar_features,
            "renewable": renewable_features,
        }

    def _select_for_target(self, df: pd.DataFrame, features: List[str],
                           target: str, model_type: str) -> List[str]:
        if target not in df.columns:
            logger.warning(f"    Target '{target}' not found for {model_type}, using combined importance")
            return sorted(features, key=lambda f: self.importance_results.get(f, FeatureImportanceResult()).combined_score, reverse=True)[:self.config.gbdt_top_k]

        valid_mask = df[target].notna()
        for f in features:
            if f in df.columns:
                valid_mask = valid_mask & df[f].notna()

        X = df.loc[valid_mask, features].fillna(0)
        y = df.loc[valid_mask, target]

        if len(X) < 100:
            return sorted(features, key=lambda f: self.importance_results.get(f, FeatureImportanceResult()).combined_score, reverse=True)[:self.config.gbdt_top_k]

        try:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                verbose=-1, random_state=42,
            )
            model.fit(X, y)
            importances = dict(zip(features, model.feature_importances_))
            sorted_feats = sorted(features, key=lambda f: importances.get(f, 0), reverse=True)
            top_k = min(self.config.gbdt_top_k, len(sorted_feats))
            selected = sorted_feats[:top_k]
            logger.info(f"    {model_type} model: {len(selected)} features selected from {len(features)}")
            return selected
        except ImportError:
            return sorted(features, key=lambda f: self.importance_results.get(f, FeatureImportanceResult()).combined_score, reverse=True)[:self.config.gbdt_top_k]

    def _build_result_dataframe(self) -> pd.DataFrame:
        rows = []
        for name, result in self.importance_results.items():
            rows.append({
                "feature_name": name,
                "gbdt_importance": result.gbdt_importance,
                "shap_importance": result.shap_importance,
                "permutation_importance": result.permutation_importance,
                "combined_score": result.combined_score,
                "is_redundant": result.is_redundant,
                "redundancy_with": result.redundancy_with,
                "final_retained": result.final_retained,
            })

        return pd.DataFrame(rows).sort_values("combined_score", ascending=False)
