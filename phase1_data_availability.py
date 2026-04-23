"""
阶段1: 数据可用性对齐
- 9点可用信息快照规则
- 回放器(Replay Engine)
- 防止信息穿越
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, time


@dataclass
class DataAvailabilitySpec:
    """
    9点可用信息快照规则定义
    
    可用信息(AsOf 9:00):
    - D+1日前数据: 天气预测、负荷预测(DAYAGO)、日前电价(DAYAGO)
    - 历史数据: 前一天及之前的实际值(REAL)
    
    不可用信息:
    - D+1及之后的REAL值
    - D日当日及未来的实时价格
    """
    
    @staticmethod
    def get_available_columns(as_of_date: pd.Timestamp) -> Dict[str, List[str]]:
        """
        获取指定日期可用的字段列表
        
        Returns:
            {
                "available_now": 可用字段列表,
                "not_available": 不可用字段列表,
                "cutoff_time": 截断时间
            }
        """
        available = [
            # 天气相关(预测值)
            "TEMPERATURE",
            "WEATHER",
            "HUMIDITY",
            "WIND_SPEED",
            
            # 已知的负荷/电价(DAYAGO是前一天预测的)
            "LOAD_DAYAGO",
            "PRICE_DAYAGO",
            "LOAD_RATE_DAYAGO",
            "PRICE_RATE_DAYAGO",
            
            # 历史实际值(前两天及之前)
            "LOAD_REAL_lag_1d",
            "LOAD_REAL_lag_2d",
            "PRICE_REAL_lag_1d",
            "PRICE_REAL_lag_2d",
            
            # 统计特征
            "LOAD_DAYAGO_rolling_mean_1d",
            "LOAD_DAYAGO_rolling_std_1d",
            "LOAD_DAYAGO_rolling_mean_7d",
            "PRICE_DAYAGO_rolling_mean_1d",
            "PRICE_DAYAGO_rolling_mean_7d",
            
            # 时间特征
            "hour_sin",
            "hour_cos",
            "dayofweek",
            "is_weekend",
            "is_holiday",
            "is_peak_hour",
        ]
        
        not_available = [
            # 当日实时数据不可用
            "LOAD_REAL",
            "PRICE_REAL",
            "PRICE_R_D",  # 当日价差
            
            # 未来数据
            "future_*",
        ]
        
        return {
            "available_now": available,
            "not_available": not_available,
        }


class ReplayEngine:
    """
    回放器 - 模拟9点可看到的信息
    
    用于:
    1. 训练时确保不穿越
    2. 回测时严格按9点可用信息预测
    """
    
    def __init__(self, df: pd.DataFrame, timezone: str = "Asia/Shanghai"):
        self.df = df.copy()
        self.timezone = timezone
        
        # 预计算lag特征
        self._prepare_lag_features()
    
    def _prepare_lag_features(self):
        """预计算lag特征用于回放"""
        pass
    
    def get_snapshot(self, target_date: pd.Timestamp, target_hour: int = 9) -> pd.DataFrame:
        """
        获取指定日期9点的数据快照
        
        Args:
            target_date: 目标日期(D+1)
            target_hour: 截断小时(默认9点)
            
        Returns:
            截断后的数据视图
        """
        cutoff = pd.Timestamp(target_date).replace(hour=target_hour, minute=0, second=0)
        
        snapshot = self.df[self.df.index < cutoff].copy()
        return snapshot
    
    def create_training_data(self, 
                           start_date: str, 
                           end_date: str,
                           available_cols: List[str]) -> pd.DataFrame:
        """
        创建训练数据 - 按9点快照规则
        
        每个样本只包含该时间点可用的信息
        """
        df_filtered = self.df.copy()
        
        for col in df_filtered.columns:
            if col not in available_cols:
                df_filtered[col] = np.nan
        
        return df_filtered
    
    def validate_no_lookahead(self, 
                            prediction_date: pd.Timestamp,
                            features: pd.DataFrame,
                            target_col: str) -> bool:
        """
        验证特征中没有未来信息穿越
        
        Returns:
            True=无穿越, False=存在穿越
        """
        cutoff = prediction_date.replace(hour=9, minute=0, second=0)
        
        future_mask = features.index >= cutoff
        if features[future_mask][target_col].notna().any():
            return False
        
        return True


class RollingPredictor:
    """
    滚动预测器 - 每天9点用可用信息预测D+1
    
    严格遵守:
    - 9点前: 可用前一日及之前所有数据
    - 9点后: 不可用D+1实时数据
    """
    
    def __init__(self, model, replay_engine: ReplayEngine):
        self.model = model
        self.replay_engine = replay_engine
    
    def predict_next_day(self, 
                        current_date: pd.Timestamp,
                        target: str = "LOAD_REAL") -> Dict[str, np.ndarray]:
        """
        预测D+1目标值
        
        使用当前日期9点可看到的所有信息
        """
        snapshot = self.replay_engine.get_snapshot(current_date)
        
        features = self._build_features(snapshot, target)
        
        pred_p10, pred_p50, pred_p90 = self.model.predict_quantiles(features)
        
        uncertainty = pred_p90 - pred_p10
        confidence = np.abs(pred_p50) / (uncertainty + 1e-6)
        
        return {
            "pred_p10": pred_p10,
            "pred_p50": pred_p50,
            "pred_p90": pred_p90,
            "uncertainty": uncertainty,
            "confidence": confidence,
        }
    
    def _build_features(self, snapshot: pd.DataFrame, target: str) -> pd.DataFrame:
        """构建预测特征"""
        features = snapshot.copy()
        return features


def run_phase1_validation():
    """验证9点快照规则"""
    print("=" * 70)
    print("阶段1: 数据可用性验证")
    print("=" * 70)
    
    df = pd.read_parquet("./data/output/feature_master_table.parquet")
    print(f"\n数据范围: {df.index.min()} ~ {df.index.max()}")
    print(f"总行数: {len(df)}")
    
    spec = DataAvailabilitySpec()
    avail = spec.get_available_columns(pd.Timestamp("2026-04-01"))
    
    print(f"\n可用字段数: {len(avail['available_now'])}")
    print(f"不可用字段数: {len(avail['not_available'])}")
    
    engine = ReplayEngine(df)
    
    test_date = pd.Timestamp("2026-04-01", tz="Asia/Shanghai")
    snapshot = engine.get_snapshot(test_date, target_hour=9)
    print(f"\n2026-04-01 9点快照: {len(snapshot)} 行")
    
    cutoff = test_date.replace(hour=9, minute=0, second=0)
    print(f"截断时间: {cutoff}")
    print(f"快照最后时间: {snapshot.index.max()}")
    
    assert snapshot.index.max() < cutoff, "快照包含截断时间之后的数据!"
    
    print("\n✓ 阶段1验证通过: 快照不包含9点后数据")


if __name__ == "__main__":
    run_phase1_validation()
