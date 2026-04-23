import os
import sys
import numpy as np
import pandas as pd
from typing import Dict

class SettlementCalculator:
    DEVIATION_THRESHOLD = 0.10
    MU = 1.05
    
    def settle(self, q_96, L_96, p_da_96, p_rt_96) -> Dict:
        n_hours = len(q_96) // 4
        q_h = np.array([np.mean(q_96[h*4:(h+1)*4]) for h in range(n_hours)])
        L_h = np.array([np.mean(L_96[h*4:(h+1)*4]) for h in range(n_hours)])
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
            spread = p_rt_h[h] - p_da_h[h]
            if excess > 0 and spread > 0:
                recovery += excess * L_h[h] * spread * self.MU
            if excess > 0:
                over_count += 1
        
        total_cost = da_cost + rt_cost + recovery
        return {
            "profit": -total_cost,
            "da_cost": da_cost,
            "rt_cost": rt_cost,
            "recovery": recovery,
            "over_count": over_count,
        }


class NormalizedBacktest:
    def __init__(self):
        self.settle = SettlementCalculator()
    
    def normalize_data(self, L, p_da, p_rt):
        L_norm = L / np.mean(L)
        p_da_norm = p_da / np.mean(p_da)
        p_rt_norm = p_rt / np.mean(p_rt)
        return L_norm, p_da_norm, p_rt_norm
    
    def scale_factor(self, confidence, direction, conf_th, max_scale, spread_w):
        if confidence < conf_th:
            return 1.0
        if direction > 0:
            base = 1.0 + confidence * spread_w
            base = min(max_scale, base)
        else:
            base = 1.0 - confidence * spread_w
            base = max(2 - max_scale, base)
        return np.clip(base, 0.85, 1.15)
    
    def run_comparison(self, df, start, end):
        from config import TIMEZONE
        
        start_ts = pd.Timestamp(start, tz=TIMEZONE)
        end_ts = pd.Timestamp(end, tz=TIMEZONE)
        
        all_dates = df.index.normalize().unique()
        all_dates = [d for d in all_dates if start_ts <= d <= end_ts]
        
        strategies = {
            "conservative": lambda L, s: L.copy(),
            "fixed_0.95": lambda L, s: L * 0.95,
            "fixed_0.9_1.1": lambda L, s: L * (1.1 if np.mean(s) > 0 else 0.9),
            "conf_adaptive_0.8": self._conf_adaptive(0.8, 1.10),
            "conf_adaptive_1.0": self._conf_adaptive(1.0, 1.10),
            "conf_adaptive_1.2": self._conf_adaptive(1.2, 1.15),
        }
        
        results = {name: [] for name in strategies.keys()}
        results["date"] = []
        
        for d in all_dates:
            d_ts = d.tz_localize(None) if d.tzinfo is None else d.tz_convert(TIMEZONE)
            day_mask = (df.index >= d_ts) & (df.index < d_ts + pd.Timedelta(days=1))
            day_df = df[day_mask]
            
            if len(day_df) < 48:
                continue
            
            L_orig = day_df["LOAD_REAL"].values[:96] if "LOAD_REAL" in day_df.columns else None
            p_da = day_df["PRICE_DAYAGO"].values[:96] if "PRICE_DAYAGO" in day_df.columns else None
            p_rt = day_df["PRICE_REAL"].values[:96] if "PRICE_REAL" in day_df.columns else None
            
            if L_orig is None or p_da is None or p_rt is None:
                continue
            
            actual_len = min(len(L_orig), len(p_da), len(p_rt), 96)
            if np.sum(np.isnan(L_orig[:actual_len])) > actual_len * 0.5:
                continue
            
            L_orig = np.nan_to_num(L_orig[:actual_len], nan=np.nanmean(L_orig[:actual_len]))
            p_da = np.nan_to_num(p_da[:actual_len], nan=np.nanmean(p_da[:actual_len]))
            p_rt = np.nan_to_num(p_rt[:actual_len], nan=np.nanmean(p_rt[:actual_len]))
            
            if actual_len < 48:
                continue
            
            L = np.tile(L_orig, (96 // actual_len + 1))[:96]
            p_da = np.tile(p_da, (96 // actual_len + 1))[:96]
            p_rt = np.tile(p_rt, (96 // actual_len + 1))[:96]
            
            L_n, p_da_n, p_rt_n = self.normalize_data(L, p_da, p_rt)
            spread = p_rt_n - p_da_n
            
            for name, strategy_fn in strategies.items():
                q = strategy_fn(L_n, spread)
                result = self.settle.settle(q, L_n, p_da_n, p_rt_n)
                results[name].append(result["profit"])
            
            results["date"].append(str(d.date()) if hasattr(d, 'date') else str(d)[:10])
        
        return pd.DataFrame(results)
    
    def _conf_adaptive(self, conf_th, max_scale):
        spread_w = 0.5
        
        def apply(L, spread):
            direction = 1 if np.mean(spread) > 0 else -1
            q = np.zeros_like(L)
            n_hours = len(L) // 4
            
            for h in range(n_hours):
                s_p50 = spread[h*4:(h+1)*4]
                s_std = np.std(s_p50) + 1e-6
                conf = abs(np.mean(s_p50)) / s_std
                scale = self.scale_factor(conf, direction, conf_th, max_scale, spread_w)
                
                for i in range(h*4, min((h+1)*4, len(L))):
                    q[i] = L[i] * scale
            
            return q
        
        return apply
    
    def print_summary(self, df_results):
        print("\n" + "=" * 75)
        print("STRATEGY COMPARISON (Normalized Data)")
        print("=" * 75)
        
        strategies = [c for c in df_results.columns if c != "date"]
        
        print(f"\n{'Strategy':<20} {'Total Profit':>15} {'Avg Daily':>12} {'Sharpe':>10} {'WinRate':>10}")
        print("-" * 70)
        
        sorted_strategies = []
        for s in strategies:
            profits = df_results[s].dropna()
            if len(profits) == 0:
                continue
            sorted_strategies.append({
                "name": s,
                "total": profits.sum(),
                "avg": profits.mean(),
                "std": profits.std(),
                "sharpe": profits.mean() / (profits.std() + 1e-9),
                "win_rate": (profits > 0).mean(),
            })
        
        sorted_strategies.sort(key=lambda x: x["total"], reverse=True)
        
        for s in sorted_strategies:
            print(f"{s['name']:<20} {s['total']:>15.4f} {s['avg']:>12.4f} {s['sharpe']:>10.4f} {s['win_rate']:>10.2%}")
        
        print("-" * 70)
        
        best = sorted_strategies[0]
        worst = sorted_strategies[-1]
        
        print(f"\nBEST: {best['name']} with total profit {best['total']:.4f}")
        print(f"WORST: {worst['name']} with total profit {worst['total']:.4f}")
        
        better_than_fixed = any(
            s["total"] > sorted_strategies[[x["name"] for x in sorted_strategies].index("fixed_0.9_1.1")]["total"]
            for s in sorted_strategies if "conf" in s["name"]
        )
        print(f"\nConfidence strategy beats fixed_0.9_1.1: {better_than_fixed}")
        
        return sorted_strategies


def main():
    print("=" * 75)
    print("COMPLETE PIPELINE: Strategy Comparison (Normalized Backtest)")
    print("=" * 75)
    
    df = pd.read_parquet("./data/output/feature_master_table.parquet")
    print(f"\nData loaded: {df.shape}")
    print(f"Range: {df.index.min()} ~ {df.index.max()}")
    
    backtest = NormalizedBacktest()
    results = backtest.run_comparison(df, "2026-04-01", "2026-04-27")
    sorted_results = backtest.print_summary(results)
    
    print("\n" + "=" * 75)
    print("KEY INSIGHTS")
    print("=" * 75)
    
    fixed_0911_idx = [i for i, s in enumerate(sorted_results) if s["name"] == "fixed_0.9_1.1"]
    if fixed_0911_idx:
        fixed_total = sorted_results[fixed_0911_idx[0]]["total"]
        conf_best = max([s for s in sorted_results if "conf" in s["name"]], key=lambda x: x["total"])
        conf_total = conf_best["total"]
        
        print(f"\nfixed_0.9_1.1: {fixed_total:.4f}")
        print(f"Best confidence: {conf_best['name']} = {conf_total:.4f}")
        print(f"Difference: {conf_total - fixed_total:.4f} ({'+' if conf_total > fixed_total else ''}{(conf_total/fixed_total - 1)*100:.2f}%)")
    
    print("\n" + "=" * 75)
    print("CONCLUSION")
    print("=" * 75)
    if conf_total > fixed_total:
        print("✓ Confidence-based strategy OUTPERFORMS fixed 0.9/1.1 strategy")
    else:
        print("✗ Fixed 0.9/1.1 strategy outperforms confidence-based strategy")
        print("  → Consider adjusting confidence thresholds or scale function")


if __name__ == "__main__":
    main()
