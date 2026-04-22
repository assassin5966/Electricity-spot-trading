import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class MissingSimulator:
    def __init__(self, mask_columns=None, mask_lengths=None, mask_prob: float = 0.3):
        from config import MISSING_MASK_COLUMNS, MISSING_MASK_LENGTHS
        self.mask_columns = mask_columns or MISSING_MASK_COLUMNS
        self.mask_lengths = mask_lengths or MISSING_MASK_LENGTHS
        self.mask_prob = mask_prob

    def simulate_missing(self, df: pd.DataFrame,
                         force_mask: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        n = len(df)
        mask_info = pd.DataFrame(index=df.index)
        mask_info["mask_flag"] = 0
        mask_info["missing_length"] = 0

        if not force_mask and np.random.random() > self.mask_prob:
            return df, mask_info

        mask_length = np.random.choice(self.mask_lengths, p=[0.2, 0.3, 0.5])

        points_per_day = 96
        total_mask_points = mask_length * points_per_day

        if total_mask_points >= n:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, n - total_mask_points)

        end_idx = min(start_idx + total_mask_points, n)

        for col in self.mask_columns:
            if col in df.columns:
                df.iloc[start_idx:end_idx, df.columns.get_loc(col)] = np.nan

        mask_info.iloc[start_idx:end_idx, mask_info.columns.get_loc("mask_flag")] = 1
        mask_info.iloc[start_idx:end_idx, mask_info.columns.get_loc("missing_length")] = mask_length

        logger.info(f"    Simulated missing: {mask_length} days, columns={self.mask_columns}, rows {start_idx}-{end_idx}")
        return df, mask_info

    def simulate_batch(self, df: pd.DataFrame, n_augments: int = 3) -> list:
        augmented = []
        for i in range(n_augments):
            df_masked, mask_info = self.simulate_missing(df)
            augmented.append((df_masked, mask_info))
        return augmented

    def create_scenario(self, df: pd.DataFrame,
                        scenario: str = "5day_missing") -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        n = len(df)
        mask_info = pd.DataFrame(index=df.index)
        mask_info["mask_flag"] = 0
        mask_info["missing_length"] = 0

        scenarios = {
            "1day_missing": 1,
            "3day_missing": 3,
            "5day_missing": 5,
            "full_available": 0,
        }

        mask_length = scenarios.get(scenario, 5)

        if mask_length > 0:
            total_mask_points = mask_length * 96
            start_idx = max(0, n - total_mask_points)
            for col in self.mask_columns:
                if col in df.columns:
                    df.iloc[start_idx:, df.columns.get_loc(col)] = np.nan
            mask_info.iloc[start_idx:, mask_info.columns.get_loc("mask_flag")] = 1
            mask_info.iloc[start_idx:, mask_info.columns.get_loc("missing_length")] = mask_length

        return df, mask_info


class MissingAugmentor:
    def __init__(self, simulator: MissingSimulator = None):
        self.simulator = simulator or MissingSimulator()

    def augment_training_data(self, X: pd.DataFrame, y: pd.DataFrame,
                              n_copies: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        all_X = [X]
        all_y = [y]

        for i in range(n_copies):
            X_masked, _ = self.simulator.simulate_missing(X.copy())
            all_X.append(X_masked)
            all_y.append(y.copy())

        X_aug = pd.concat(all_X, ignore_index=True)
        y_aug = pd.concat(all_y, ignore_index=True)

        logger.info(f"  Augmented training data: {len(X)} -> {len(X_aug)} samples ({n_copies} copies with missing simulation)")
        return X_aug, y_aug

    def create_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask_cols = [c for c in df.columns if any(m in c for m in self.simulator.mask_columns)]

        df["mask_flag"] = 0
        df["missing_length"] = 0

        for col in mask_cols:
            if col in df.columns:
                is_missing = df[col].isna()
                df.loc[is_missing, "mask_flag"] = 1

        missing_groups = df["mask_flag"].groupby((df["mask_flag"] != df["mask_flag"].shift()).cumsum())
        for name, group in missing_groups:
            if group.iloc[0] == 1:
                length = len(group) // 96 + (1 if len(group) % 96 > 0 else 0)
                df.loc[group.index, "missing_length"] = length

        return df
