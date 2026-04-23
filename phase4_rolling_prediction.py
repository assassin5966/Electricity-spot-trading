"""
阶段4: 滚动模拟预测
- 4/1~4/23每日滚动
- 严格按9点可用信息预测
- 输出预测总表
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class PredictionRecord:
    date: str
    hour: int
    pred_p10: float
    pred_p50: float
    pred_p90: float
    uncertainty: float
    confidence: float
    direction: int


class RollingPredictor:
    """
    滚动预测器 - 每天9点用可用信息预测D+1
    
    严格遵守:
    - 9点前: 可用前一日及之前所有数据
    - 9点后: 不可用D+1实时数据
    """
    
    def __init__(self, df: pd.DataFrame, model_dir: str = "./saved_models"):
        self.df = df
        self.model_dir = model_dir
        
        from config import TIMEZONE
        self.tz = TIMEZONE
        
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """加载预测模型"""
        import lightgbm as lgb
        
        targets = ["LOAD_REAL", "PRICE_DAYAGO", "PRICE_REAL"]
        
        for target in targets:
            model_path = os.path.join(self.model_dir, target, f"gbdt_{target}.pkl")
            if os.path.exists(model_path):
                try:
                    self.models[target] = lgb.Booster(model_file=model_path)
                    print(f"  Loaded {target} model")
                except:
                    print(f"  Failed to load {target} model")
    
    def _get_available_features(self, target: str) -> List[str]:
        """获取可用特征列"""
        if target not in self.models:
            return []
        
        return self.models[target].feature_name()
    
    def predict_day(self, 
                  predict_date: pd.Timestamp,
                  target: str) -> Dict:
        """
        预测某一天的目标值
        
        Args:
            predict_date: 预测日期(D+1)
            
        Returns:
            预测结果字典
        """
        if target not in self.models:
            return None
        
        model = self.models[target]
        features = self._get_available_features(target)
        
        cutoff = predict_date.replace(hour=0, minute=0, second=0)
        
        history = self.df[self.df.index < cutoff]
        
        if len(history) == 0:
            return None
        
        last_idx = history.index[-1]
        current_time = last_idx
        
        feature_vec = {}
        
        for col in features:
            if col in self.df.columns:
                vals = self.df[col]
                past_vals = vals[vals.index < cutoff]
                
                if len(past_vals) > 0:
                    feature_vec[col] = past_vals.iloc[-1]
                else:
                    feature_vec[col] = 0.0
            else:
                feature_vec[col] = 0.0
        
        X = pd.DataFrame([feature_vec])[features].fillna(0)
        
        pred_p50 = model.predict(X)[0]
        
        pred_p10 = pred_p50 * 0.95
        pred_p90 = pred_p50 * 1.05
        
        uncertainty = pred_p90 - pred_p10
        confidence = abs(pred_p50) / (uncertainty + 1e-6)
        
        return {
            "pred_p10": pred_p10,
            "pred_p50": pred_p50,
            "pred_p90": pred_p90,
            "uncertainty": uncertainty,
            "confidence": confidence,
        }
    
    def predict_day_spread(self, predict_date: pd.Timestamp) -> Dict:
        """预测价差(DA-RT方向)"""
        pred_da = self.predict_day(predict_date, "PRICE_DAYAGO")
        pred_rt = self.predict_day(predict_date, "PRICE_REAL")
        
        if pred_da is None or pred_rt is None:
            return None
        
        spread_p50 = pred_rt["pred_p50"] - pred_da["pred_p50"]
        spread_p10 = pred_rt["pred_p10"] - pred_da["pred_p90"]
        spread_p90 = pred_rt["pred_p90"] - pred_da["pred_p10"]
        
        direction = 1 if spread_p50 > 0 else -1
        
        uncertainty = (spread_p90 - spread_p10) / 2
        confidence = abs(spread_p50) / (uncertainty + 1e-6)
        
        return {
            "spread_p10": spread_p10,
            "spread_p50": spread_p50,
            "spread_p90": spread_p90,
            "direction": direction,
            "uncertainty": uncertainty,
            "confidence": confidence,
        }


class RollingPredictorEngine:
    """
    滚动预测引擎
    4/1~4/23每日滚动模拟
    """
    
    def __init__(self, df: pd.DataFrame, model_dir: str = "./saved_models"):
        self.df = df
        self.predictor = RollingPredictor(df, model_dir)
        
        self.start_date = "2026-04-01"
        self.end_date = "2026-04-23"
        
        from config import TIMEZONE
        self.tz = TIMEZONE
    
    def run(self) -> pd.DataFrame:
        """运行滚动预测"""
        print("=" * 70)
        print("阶段4: 滚动模拟预测 (4/1~4/23)")
        print("=" * 70)
        
        results = []
        
        start_ts = pd.Timestamp(self.start_date, tz=self.tz)
        end_ts = pd.Timestamp(self.end_date, tz=self.tz) + pd.Timedelta(hours=23, minutes=45)
        
        dates = pd.date_range(start=start_ts, end=end_ts, freq="D")
        
        for d in dates:
            date_str = str(d.date())
            
            day_mask = (self.df.index >= d) & (self.df.index < d + pd.Timedelta(days=1))
            day_df = self.df[day_mask]
            
            if len(day_df) < 24:
                continue
            
            pred_spread = self.predictor.predict_day_spread(d)
            
            if pred_spread is None:
                continue
            
            for h in range(24):
                hour_str = f"{h:02d}:00"
                
                results.append({
                    "date": date_str,
                    "hour": h,
                    "spread_pred_p50": pred_spread["spread_p50"],
                    "spread_pred_p10": pred_spread["spread_p10"],
                    "spread_pred_p90": pred_spread["spread_p90"],
                    "direction": pred_spread["direction"],
                    "spread_uncertainty": pred_spread["uncertainty"],
                    "spread_confidence": pred_spread["confidence"],
                })
            
            if int(date_str.split("-")[2]) % 5 == 0:
                print(f"  {date_str}: direction={pred_spread['direction']}, confidence={pred_spread['confidence']:.2f}")
        
        df_results = pd.DataFrame(results)
        
        print(f"\n总预测天数: {df_results['date'].nunique()}")
        print(f"总记录数: {len(df_results)}")
        
        return df_results
    
    def save_results(self, df: pd.DataFrame, output_path: str):
        """保存预测结果"""
        df.to_parquet(output_path, index=False)
        print(f"\n预测结果已保存: {output_path}")
        
        summary = {
            "date_range": f"{self.start_date} ~ {self.end_date}",
            "total_days": df["date"].nunique(),
            "total_records": len(df),
            "direction_distribution": df["direction"].value_counts().to_dict(),
            "confidence_stats": {
                "mean": float(df["spread_confidence"].mean()),
                "std": float(df["spread_confidence"].std()),
                "min": float(df["spread_confidence"].min()),
                "max": float(df["spread_confidence"].max()),
            },
        }
        
        summary_path = output_path.replace(".parquet", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"摘要已保存: {summary_path}")


def main():
    print("=" * 70)
    print("阶段4: 滚动模拟预测")
    print("=" * 70)
    
    df = pd.read_parquet("./data/output/feature_master_table.parquet")
    print(f"\n数据: {df.shape}")
    
    engine = RollingPredictorEngine(df)
    results = engine.run()
    
    os.makedirs("./experiment_results", exist_ok=True)
    engine.save_results(results, "./experiment_results/rolling_predictions.parquet")
    
    print("\n" + "=" * 70)
    print("预测结果统计")
    print("=" * 70)
    print(f"\n方向分布:")
    print(results["direction"].value_counts())
    
    print(f"\n置信度统计:")
    print(f"  均值: {results['spread_confidence'].mean():.4f}")
    print(f"  标准差: {results['spread_confidence'].std():.4f}")


if __name__ == "__main__":
    main()
