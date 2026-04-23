"""
阶段5: 策略函数优化
- 方向+置信度 → 动态缩放倍率
- 以收益最大化为目标优化
- 输出策略收益、回收、超阈值等报告
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class SettlementResult:
    """结算结果"""
    profit: float
    da_cost: float
    rt_cost: float
    recovery: float
    over_threshold_points: int
    over_threshold_ratio: float


class SettlementSimulator:
    """结算模拟器"""
    
    DEVIATION_THRESHOLD = 0.10
    MU = 1.05
    
    def settle(self, 
               q_96: np.ndarray,
               L_actual: np.ndarray,
               p_da_96: np.ndarray,
               p_rt_96: np.ndarray) -> SettlementResult:
        """
        结算计算
        
        Args:
            q_96: 申报量(96点)
            L_actual: 实际负荷(96点)
            p_da_96: 日前电价(96点)
            p_rt_96: 实时电价(96点)
        """
        n_hours = len(q_96) // 4
        
        q_h = np.array([np.mean(q_96[h*4:(h+1)*4]) for h in range(n_hours)])
        L_h = np.array([np.mean(L_actual[h*4:(h+1)*4]) for h in range(n_hours)])
        p_da_h = np.array([np.mean(p_da_96[h*4:(h+1)*4]) for h in range(n_hours)])
        p_rt_h = np.array([np.mean(p_rt_96[h*4:(h+1)*4]) for h in range(n_hours)])
        
        da_cost = np.sum((q_h - L_h) * p_da_h)
        rt_cost = np.sum((L_h - q_h) * p_rt_h)
        
        recovery = 0.0
        over_count = 0
        
        for h in range(n_hours):
            dev = abs(q_h[h] - L_h[h])
            thresh = self.DEVIATION_THRESHOLD * abs(L_h[h])
            excess = max(0, dev - thresh)
            spread_h = p_rt_h[h] - p_da_h[h]
            
            if excess > 0 and spread_h > 0:
                recovery += excess * L_h[h] * spread_h * self.MU
            
            if excess > 0:
                over_count += 1
        
        total_cost = da_cost + rt_cost + recovery
        
        return SettlementResult(
            profit=-total_cost,
            da_cost=da_cost,
            rt_cost=rt_cost,
            recovery=recovery,
            over_threshold_points=over_count,
            over_threshold_ratio=over_count / n_hours,
        )


class StrategyScalerFunction:
    """
    策略缩放函数
    
    ratio_t = f(direction_t, confidence_t, context_t; θ)
    
    可学习参数:
    - conf_threshold: 置信度阈值
    - max_scale_pos: 正向最大缩放
    - max_scale_neg: 负向最大缩放
    - weight: 置信度权重
    """
    
    def __init__(self, params: Dict = None):
        self.params = params or {
            "conf_threshold": 0.8,
            "max_scale_pos": 1.10,
            "max_scale_neg": 0.90,
            "weight": 0.5,
        }
    
    def compute_scale(self, 
                     direction: int,
                     confidence: float) -> float:
        """
        计算缩放因子
        
        Args:
            direction: 方向 (1=RT>DA, -1=RT<DA)
            confidence: 置信度
            
        Returns:
            scale: 缩放因子
        """
        p = self.params
        
        if confidence < p["conf_threshold"]:
            return 1.0
        
        if direction > 0:
            base = 1.0 + confidence * p["weight"]
            return min(base, p["max_scale_pos"])
        else:
            base = 1.0 - confidence * p["weight"]
            return max(base, p["max_scale_neg"])
    
    def apply_to_predictions(self,
                           given_curve: np.ndarray,
                           predictions: pd.DataFrame) -> np.ndarray:
        """
        将缩放函数应用到预测结果
        
        Args:
            given_curve: 给定曲线(96点)
            predictions: 预测DataFrame，包含direction, confidence等
            
        Returns:
            q_96: 申报量(96点)
        """
        q_96 = np.zeros(96)
        
        for h in range(24):
            row = predictions[predictions["hour"] == h]
            if len(row) == 0:
                q_96[h*4:(h+1)*4] = given_curve[h*4:(h+1)*4]
                continue
            
            row = row.iloc[0]
            direction = row.get("direction", 1)
            confidence = row.get("spread_confidence", 0)
            
            scale = self.compute_scale(direction, confidence)
            
            base_q = np.mean(given_curve[h*4:(h+1)*4])
            q_96[h*4:(h+1)*4] = base_q * scale
        
        return q_96


class StrategyOptimizer:
    """
    策略优化器
    用历史回放期最大化收益
    """
    
    def __init__(self, df: pd.DataFrame, predictions: pd.DataFrame):
        self.df = df
        self.predictions = predictions
        self.settler = SettlementSimulator()
        
        from config import TIMEZONE
        self.tz = TIMEZONE
    
    def _prepare_day_data(self, date_str: str) -> Dict:
        """准备某一天的数据"""
        d = pd.Timestamp(date_str, tz=self.tz)
        day_mask = (self.df.index >= d) & (self.df.index < d + pd.Timedelta(days=1))
        day_df = self.df[day_mask]
        
        if len(day_df) < 96:
            return None
        
        L_actual = day_df["LOAD_REAL"].values[:96]
        p_da = day_df["PRICE_DAYAGO"].values[:96]
        p_rt = day_df["PRICE_REAL"].values[:96]
        given_curve = day_df["LOAD_DAYAGO"].values[:96] if "LOAD_DAYAGO" in day_df.columns else L_actual
        
        L_actual = np.nan_to_num(L_actual, nan=np.nanmean(L_actual))
        p_da = np.nan_to_num(p_da, nan=np.nanmean(p_da))
        p_rt = np.nan_to_num(p_rt, nan=np.nanmean(p_rt))
        given_curve = np.nan_to_num(given_curve, nan=np.nanmean(given_curve))
        
        day_preds = self.predictions[self.predictions["date"] == date_str]
        
        return {
            "L_actual": L_actual,
            "p_da": p_da,
            "p_rt": p_rt,
            "given_curve": given_curve,
            "predictions": day_preds,
        }
    
    def evaluate_params(self, params: Dict) -> float:
        """
        评估一组参数的性能
        
        Returns:
            总收益(负值用于最小化)
        """
        scaler = StrategyScalerFunction(params)
        
        total_profit = 0.0
        total_recovery = 0.0
        total_over = 0
        
        dates = self.predictions["date"].unique()
        
        for date_str in dates:
            data = self._prepare_day_data(date_str)
            if data is None:
                continue
            
            q_96 = scaler.apply_to_predictions(
                data["given_curve"],
                data["predictions"]
            )
            
            result = self.settler.settle(
                q_96,
                data["L_actual"],
                data["p_da"],
                data["p_rt"]
            )
            
            total_profit += result.profit
            total_recovery += result.recovery
            total_over += result.over_threshold_points
        
        obj = -(total_profit - 0.1 * total_recovery - 0.05 * total_over)
        
        return obj
    
    def optimize(self, method: str = "grid") -> Dict:
        """
        优化参数
        
        使用Grid Search或Bayesian Optimization
        """
        print("\n" + "=" * 70)
        print("阶段5: 策略函数优化")
        print("=" * 70)
        
        if method == "grid":
            return self._optimize_grid()
        elif method == "bayesian":
            return self._optimize_bayesian()
        else:
            return self._optimize_grid()
    
    def _optimize_grid(self) -> Dict:
        """网格搜索优化"""
        print("\n使用网格搜索优化参数...")
        
        best_params = None
        best_score = float("inf")
        
        conf_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        max_scales = [1.05, 1.10, 1.15]
        weights = [0.3, 0.4, 0.5, 0.6]
        
        total = len(conf_thresholds) * len(max_scales) * len(weights)
        count = 0
        
        for conf_th in conf_thresholds:
            for max_s in max_scales:
                for w in weights:
                    params = {
                        "conf_threshold": conf_th,
                        "max_scale_pos": max_s,
                        "max_scale_neg": 2 - max_s,
                        "weight": w,
                    }
                    
                    score = self.evaluate_params(params)
                    count += 1
                    
                    if score < best_score:
                        best_score = score
                        best_params = params.copy()
                        
                        if count % 20 == 0:
                            print(f"  Progress: {count}/{total}, Best score: {-best_score:.4f}")
        
        print(f"\n最优参数: {best_params}")
        print(f"最优分数: {-best_score:.4f}")
        
        self.best_params = best_params
        return best_params
    
    def _optimize_bayesian(self) -> Dict:
        """贝叶斯优化"""
        print("\n使用贝叶斯优化参数...")
        
        bounds = [
            (0.3, 1.2),    # conf_threshold
            (1.02, 1.20),  # max_scale_pos
            (0.3, 0.7),    # weight
        ]
        
        result = differential_evolution(
            self._evaluate_continuous,
            bounds,
            maxiter=50,
            seed=42,
        )
        
        best_params = {
            "conf_threshold": result.x[0],
            "max_scale_pos": result.x[1],
            "max_scale_neg": 2 - result.x[1],
            "weight": result.x[2],
        }
        
        print(f"\n最优参数: {best_params}")
        print(f"最优分数: {-result.fun:.4f}")
        
        self.best_params = best_params
        return best_params
    
    def _evaluate_continuous(self, x: np.ndarray) -> float:
        """评估连续参数"""
        params = {
            "conf_threshold": x[0],
            "max_scale_pos": x[1],
            "max_scale_neg": 2 - x[1],
            "weight": x[2],
        }
        return self.evaluate_params(params)
    
    def final_evaluation(self, params: Dict) -> Dict:
        """最终评估报告"""
        scaler = StrategyScalerFunction(params)
        
        total_profit = 0.0
        total_recovery = 0.0
        total_over = 0
        daily_profits = []
        
        dates = self.predictions["date"].unique()
        
        for date_str in dates:
            data = self._prepare_day_data(date_str)
            if data is None:
                continue
            
            q_96 = scaler.apply_to_predictions(
                data["given_curve"],
                data["predictions"]
            )
            
            result = self.settler.settle(
                q_96,
                data["L_actual"],
                data["p_da"],
                data["p_rt"]
            )
            
            total_profit += result.profit
            total_recovery += result.recovery
            total_over += result.over_threshold_points
            daily_profits.append(result.profit)
        
        daily_profits = np.array(daily_profits)
        
        return {
            "total_profit": total_profit,
            "avg_daily_profit": np.mean(daily_profits),
            "std_daily_profit": np.std(daily_profits),
            "sharpe_ratio": np.mean(daily_profits) / (np.std(daily_profits) + 1e-9),
            "win_rate": np.mean(daily_profits > 0),
            "total_recovery": total_recovery,
            "total_over_threshold": total_over,
            "avg_over_threshold_ratio": total_over / (len(dates) * 24),
        }


def compare_with_baselines(optimizer: StrategyOptimizer, 
                          best_params: Dict) -> pd.DataFrame:
    """与基线策略对比"""
    print("\n" + "=" * 70)
    print("策略对比报告")
    print("=" * 70)
    
    results = []
    
    baselines = {
        "conservative": {"conf_threshold": 999, "max_scale_pos": 1.0, "max_scale_neg": 1.0, "weight": 0},
        "fixed_0.9_1.1": {"conf_threshold": 0.0, "max_scale_pos": 1.1, "max_scale_neg": 0.9, "weight": 1.0},
        "optimized": best_params,
    }
    
    for name, params in baselines.items():
        report = optimizer.final_evaluation(params)
        report["strategy"] = name
        results.append(report)
        
        print(f"\n{name}:")
        print(f"  Total Profit: {report['total_profit']:,.2f}")
        print(f"  Sharpe Ratio: {report['sharpe_ratio']:.4f}")
        print(f"  Win Rate: {report['win_rate']:.2%}")
        print(f"  Recovery: {report['total_recovery']:,.2f}")
        print(f"  Over Threshold: {report['total_over_threshold']}")
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("阶段5: 策略函数优化")
    print("=" * 70)
    
    df = pd.read_parquet("./data/output/feature_master_table.parquet")
    print(f"\n数据: {df.shape}")
    
    pred_path = "./experiment_results/rolling_predictions.parquet"
    if os.path.exists(pred_path):
        predictions = pd.read_parquet(pred_path)
    else:
        print("运行阶段4生成预测...")
        import phase4_rolling_prediction as p4
        engine = p4.RollingPredictorEngine(df)
        predictions = engine.run()
    
    print(f"预测数据: {len(predictions)} 条")
    
    optimizer = StrategyOptimizer(df, predictions)
    
    best_params = optimizer.optimize(method="grid")
    
    comparison = compare_with_baselines(optimizer, best_params)
    
    print("\n" + "=" * 70)
    print("优化结论")
    print("=" * 70)
    
    opt_result = comparison[comparison["strategy"] == "optimized"].iloc[0]
    fixed_result = comparison[comparison["strategy"] == "fixed_0.9_1.1"].iloc[0]
    
    profit_improvement = opt_result["total_profit"] - fixed_result["total_profit"]
    
    print(f"\n优化后 vs 固定0.9/1.1:")
    print(f"  收益提升: {profit_improvement:,.2f} ({profit_improvement/fixed_result['total_profit']*100:+.1f}%)")
    print(f"  夏普比率: {opt_result['sharpe_ratio']:.4f} vs {fixed_result['sharpe_ratio']:.4f}")
    print(f"  胜率: {opt_result['win_rate']:.2%} vs {fixed_result['win_rate']:.2%}")
    
    os.makedirs("./experiment_results", exist_ok=True)
    
    with open("./experiment_results/optimal_strategy_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    comparison.to_csv("./experiment_results/strategy_comparison.csv", index=False)
    
    print(f"\n最优参数已保存: ./experiment_results/optimal_strategy_params.json")
    print(f"对比结果已保存: ./experiment_results/strategy_comparison.csv")


if __name__ == "__main__":
    main()
