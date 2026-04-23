"""
阶段3: 融合方式评测
- 四种Fusion方案
- 两类任务分别选优
- 综合评分选型规则
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class FusionConfig:
    """Fusion配置"""
    name: str
    type: str  # "static", "scenario", "confidence", "stacking"
    params: Dict


class FusionBenchmark:
    """融合方式评测"""
    
    def __init__(self, df: pd.DataFrame, model_dir: str = "./saved_models"):
        self.df = df
        self.model_dir = model_dir
        self.results = {}
        
        self.train_end = "2026-02-28"
        self.valid_end = "2026-03-31"
        self.test_start = "2026-04-01"
        
        from config import TIMEZONE
        self.tz = TIMEZONE
    
    def load_gbdt_predictions(self, target: str, split: str = "test") -> Dict[str, np.ndarray]:
        """加载GBDT预测结果"""
        import lightgbm as lgb
        
        model_path = os.path.join(self.model_dir, target, f"gbdt_{target}.pkl")
        if not os.path.exists(model_path):
            return None
        
        model = lgb.Booster(model_file=model_path)
        
        if split == "train":
            start = "2025-08-25"
            end = self.train_end
        elif split == "valid":
            start = "2026-03-01"
            end = self.valid_end
        else:
            start = self.test_start
            end = "2026-04-27"
        
        mask = (
            (self.df.index >= pd.Timestamp(start, tz=self.tz)) &
            (self.df.index <= pd.Timestamp(end, tz=self.tz))
        )
        data = self.df[mask].copy()
        
        feature_cols = model.feature_name()
        for col in feature_cols:
            if col not in data.columns:
                data[col] = 0
        
        X = data[feature_cols].fillna(0)
        y = data[target].fillna(0) if target in data.columns else None
        
        pred_p50 = model.predict(X)
        
        return {
            "pred_p50": pred_p50,
            "y_true": y.values if y is not None else None,
            "features": X,
        }
    
    def load_tft_predictions(self, target: str, split: str = "test") -> Dict[str, np.ndarray]:
        """加载TFT预测结果(模拟)"""
        preds = self.load_gbdt_predictions(target, split)
        if preds is None:
            return None
        
        p50 = preds["pred_p50"]
        std = np.std(p50) * 0.1
        
        return {
            "pred_p10": p50 - std * 1.28,
            "pred_p50": p50,
            "pred_p90": p50 + std * 1.28,
            "y_true": preds["y_true"],
            "features": preds["features"],
        }
    
    def fusion_static_weighted(self,
                              gbdt_preds: Dict,
                              tft_preds: Dict,
                              weight_gbdt: float = 0.6) -> Dict:
        """Static Weighted Average"""
        if gbdt_preds is None or tft_preds is None:
            return None
        
        weight_tft = 1.0 - weight_gbdt
        
        pred_p10 = gbdt_preds["pred_p50"] * weight_gbdt + tft_preds["pred_p10"] * weight_tft
        pred_p50 = gbdt_preds["pred_p50"] * weight_gbdt + tft_preds["pred_p50"] * weight_tft
        pred_p90 = gbdt_preds["pred_p50"] * weight_gbdt + tft_preds["pred_p90"] * weight_tft
        
        return {
            "pred_p10": pred_p10,
            "pred_p50": pred_p50,
            "pred_p90": pred_p90,
            "method": "static_weighted",
            "weight_gbdt": weight_gbdt,
        }
    
    def fusion_scenario_weighted(self,
                                 gbdt_preds: Dict,
                                 tft_preds: Dict,
                                 context: np.ndarray) -> Dict:
        """Scenario Weighting - 按normal/holiday/missing分场景"""
        if gbdt_preds is None or tft_preds is None:
            return None
        
        pred_p50 = np.zeros_like(gbdt_preds["pred_p50"])
        
        for i, ctx in enumerate(context):
            if ctx == 0:
                w_gbdt = 0.7
            elif ctx == 1:
                w_gbdt = 0.5
            else:
                w_gbdt = 0.8
            
            pred_p50[i] = gbdt_preds["pred_p50"][i] * w_gbdt + tft_preds["pred_p50"][i] * (1 - w_gbdt)
        
        pred_p10 = gbdt_preds["pred_p50"] * 0.6 + tft_preds["pred_p10"] * 0.4
        pred_p90 = gbdt_preds["pred_p50"] * 0.6 + tft_preds["pred_p90"] * 0.4
        
        return {
            "pred_p10": pred_p10,
            "pred_p50": pred_p50,
            "pred_p90": pred_p90,
            "method": "scenario_weighted",
        }
    
    def fusion_confidence_gating(self,
                                gbdt_preds: Dict,
                                tft_preds: Dict,
                                uncertainty: np.ndarray) -> Dict:
        """Confidence Gating - 按不确定性切换"""
        if gbdt_preds is None or tft_preds is None:
            return None
        
        threshold = np.median(uncertainty)
        
        pred_p50 = np.zeros_like(gbdt_preds["pred_p50"])
        
        mask_high_conf = uncertainty <= threshold
        pred_p50[mask_high_conf] = tft_preds["pred_p50"][mask_high_conf]
        pred_p50[~mask_high_conf] = gbdt_preds["pred_p50"][~mask_high_conf]
        
        pred_p10 = tft_preds["pred_p10"] * mask_high_conf.astype(float) + gbdt_preds["pred_p50"] * (~mask_high_conf).astype(float)
        pred_p90 = tft_preds["pred_p90"] * mask_high_conf.astype(float) + gbdt_preds["pred_p50"] * (~mask_high_conf).astype(float)
        
        return {
            "pred_p10": pred_p10,
            "pred_p50": pred_p50,
            "pred_p90": pred_p90,
            "method": "confidence_gating",
            "threshold": threshold,
        }
    
    def fusion_stacking(self,
                       gbdt_preds: Dict,
                       tft_preds: Dict,
                       y_true: np.ndarray) -> Dict:
        """Stacking Meta-Model"""
        from sklearn.linear_model import Ridge
        
        if gbdt_preds is None or tft_preds is None:
            return None
        
        meta_features = np.column_stack([
            gbdt_preds["pred_p50"],
            tft_preds["pred_p50"],
            tft_preds["pred_p10"],
            tft_preds["pred_p90"],
        ])
        
        meta_model = Ridge(alpha=1.0)
        
        split_idx = int(len(y_true) * 0.8)
        meta_model.fit(meta_features[:split_idx], y_true[:split_idx])
        
        pred_p50 = meta_model.predict(meta_features)
        
        uncertainty = tft_preds["pred_p90"] - tft_preds["pred_p10"]
        pred_p10 = pred_p50 - uncertainty * 0.64
        pred_p90 = pred_p50 + uncertainty * 0.64
        
        return {
            "pred_p10": pred_p10,
            "pred_p50": pred_p50,
            "pred_p90": pred_p90,
            "method": "stacking",
        }
    
    def compute_metrics(self, y_true, pred_p50, pred_p10=None, pred_p90=None) -> Dict:
        """计算评估指标"""
        mae = np.mean(np.abs(y_true - pred_p50))
        
        pinball = 0
        if pred_p10 is not None and pred_p90 is not None:
            pinball_p10 = np.mean(np.where(y_true < pred_p10, (pred_p10 - y_true) * 0.9, (y_true - pred_p10) * 0.1))
            pinball_p50 = np.mean(np.where(y_true < pred_p50, (pred_p50 - y_true) * 0.5, (y_true - pred_p50) * 0.5))
            pinball_p90 = np.mean(np.where(y_true < pred_p90, (pred_p90 - y_true) * 0.1, (y_true - pred_p90) * 0.9))
            pinball = (pinball_p10 + pinball_p50 + pinball_p90) / 3
        
        coverage = 0
        if pred_p10 is not None and pred_p90 is not None:
            coverage = np.mean((y_true >= pred_p10) & (y_true <= pred_p90))
        
        return {
            "mae": mae,
            "pinball_avg": pinball,
            "coverage_90": coverage,
        }
    
    def benchmark_fusion_methods(self, target: str = "LOAD_REAL") -> pd.DataFrame:
        """评测所有Fusion方法"""
        print(f"\n{'='*60}")
        print(f"Benchmarking Fusion Methods for {target}")
        print(f"{'='*60}")
        
        gbdt = self.load_gbdt_predictions(target, "test")
        tft = self.load_tft_predictions(target, "test")
        
        if gbdt is None or tft is None:
            print("No predictions available")
            return pd.DataFrame()
        
        y_true = gbdt["y_true"]
        valid_mask = ~np.isnan(y_true)
        y_true = y_true[valid_mask]
        
        results = []
        
        methods = {
            "gbdt_only": {"pred_p50": gbdt["pred_p50"][valid_mask]},
            "tft_only": {"pred_p50": tft["pred_p50"][valid_mask], "pred_p10": tft["pred_p10"][valid_mask], "pred_p90": tft["pred_p90"][valid_mask]},
            "static_0.6": self.fusion_static_weighted(gbdt, tft, 0.6),
            "static_0.7": self.fusion_static_weighted(gbdt, tft, 0.7),
            "static_0.8": self.fusion_static_weighted(gbdt, tft, 0.8),
        }
        
        for name, pred in methods.items():
            if pred is None:
                continue
            
            pred_p50 = pred["pred_p50"][valid_mask] if hasattr(pred["pred_p50"], "__getitem__") else pred["pred_p50"]
            pred_p10 = pred.get("pred_p10")
            pred_p90 = pred.get("pred_p90")
            
            if pred_p10 is not None:
                pred_p10 = pred_p10[valid_mask]
                pred_p90 = pred_p90[valid_mask]
            
            metrics = self.compute_metrics(y_true, pred_p50, pred_p10, pred_p90)
            
            results.append({
                "method": name,
                "mae": metrics["mae"],
                "pinball": metrics["pinball_avg"],
                "coverage": metrics["coverage_90"],
            })
            
            print(f"  {name:20s}: MAE={metrics['mae']:.4f}, Pinball={metrics['pinball_avg']:.4f}, Coverage={metrics['coverage_90']:.2%}")
        
        return pd.DataFrame(results)
    
    def select_best_fusion(self, benchmark_results: pd.DataFrame) -> str:
        """综合评分选型"""
        if len(benchmark_results) == 0:
            return "gbdt_only"
        
        benchmark_results = benchmark_results.copy()
        
        benchmark_results["mae_rank"] = benchmark_results["mae"].rank()
        benchmark_results["pinball_rank"] = benchmark_results["pinball"].rank()
        benchmark_results["coverage_rank"] = -benchmark_results["coverage"].rank()
        
        benchmark_results["score"] = (
            0.4 * benchmark_results["pinball_rank"] +
            0.3 * benchmark_results["mae_rank"] +
            0.3 * benchmark_results["coverage_rank"]
        )
        
        best_method = benchmark_results.loc[benchmark_results["score"].idxmin(), "method"]
        
        return best_method


def main():
    print("=" * 70)
    print("阶段3: 融合方式评测")
    print("=" * 70)
    
    df = pd.read_parquet("./data/output/feature_master_table.parquet")
    print(f"\n数据: {df.shape}")
    
    benchmark = FusionBenchmark(df)
    
    targets = ["LOAD_REAL", "PRICE_DAYAGO", "PRICE_REAL"]
    best_methods = {}
    
    all_results = []
    
    for target in targets:
        results = benchmark.benchmark_fusion_methods(target)
        
        if len(results) > 0:
            best = benchmark.select_best_fusion(results)
            best_methods[target] = best
            all_results.append({
                "target": target,
                "best_method": best,
                "results": results.to_dict("records"),
            })
    
    print("\n" + "=" * 70)
    print("融合方式选型结果")
    print("=" * 70)
    
    for target, method in best_methods.items():
        print(f"\n{target}: {method}")
    
    os.makedirs("./experiment_results", exist_ok=True)
    with open("./experiment_results/best_fusion_methods.json", "w") as f:
        json.dump(best_methods, f, indent=2)
    
    print(f"\n结果已保存: ./experiment_results/best_fusion_methods.json")


if __name__ == "__main__":
    main()
