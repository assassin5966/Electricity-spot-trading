import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import TIMEZONE, POINTS_PER_DAY

logger = logging.getLogger(__name__)


@dataclass
class AlignConfig:
    fill_mode: str = "duplicate"
    interpolate: bool = False
    enforce_no_future_leak: bool = True
    decision_cutoff_hour: int = 9
    decision_cutoff_minute: int = 30
    gfs_run_cycle_hours: List[int] = field(default_factory=lambda: [0, 6, 12, 18])
    max_lead_hours: int = 384


@dataclass
class AlignDiagnostics:
    total_samples: int = 0
    samples_with_weather: int = 0
    samples_without_weather: int = 0
    future_leak_detected: int = 0
    lead_hour_stats: Dict[str, float] = field(default_factory=dict)
    same_hour_consistency_check: bool = False
    run_time_distribution: Dict[str, int] = field(default_factory=dict)


class GFSAligner:
    def __init__(self, config: AlignConfig = None):
        self.config = config or AlignConfig()
        self.diagnostics = AlignDiagnostics()

    def build_gfs_hourly_table(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("  Building GFS hourly standard table ...")

        df = weather_df.copy()

        df["TIME_FCST"] = pd.to_datetime(df["TIME_FCST"], errors="coerce")
        df["TIME_FORECAST"] = pd.to_datetime(df["TIME_FORECAST"], errors="coerce")
        df = df.dropna(subset=["TIME_FCST", "TIME_FORECAST"])

        if df["TIME_FCST"].dt.tz is None:
            df["TIME_FCST"] = df["TIME_FCST"].dt.tz_localize(TIMEZONE)
        else:
            df["TIME_FCST"] = df["TIME_FCST"].dt.tz_convert(TIMEZONE)

        if df["TIME_FORECAST"].dt.tz is None:
            df["TIME_FORECAST"] = df["TIME_FORECAST"].dt.tz_localize(TIMEZONE)
        else:
            df["TIME_FORECAST"] = df["TIME_FORECAST"].dt.tz_convert(TIMEZONE)

        df["gfs_run_time"] = df["TIME_FCST"]
        df["gfs_valid_time"] = df["TIME_FORECAST"]

        meta_cols = {"TIME_FCST", "TIME_FORECAST", "CITY_NAME", "CITY_CODE", "LON", "LAT",
                     "gfs_run_time", "gfs_valid_time"}
        weather_cols = [c for c in df.columns if c not in meta_cols]

        agg = df.groupby(["gfs_run_time", "gfs_valid_time"])[weather_cols].mean().reset_index()

        agg["lead_hour"] = (
            (agg["gfs_valid_time"] - agg["gfs_run_time"]).dt.total_seconds() / 3600
        ).astype(int)

        agg = agg[agg["lead_hour"] >= 0].copy()
        agg = agg[agg["lead_hour"] <= self.config.max_lead_hours].copy()

        agg = agg.sort_values(["gfs_run_time", "gfs_valid_time"]).reset_index(drop=True)

        logger.info(f"    GFS hourly table: {len(agg)} rows, "
                    f"run_time range: {agg['gfs_run_time'].min()} ~ {agg['gfs_run_time'].max()}, "
                    f"valid_time range: {agg['gfs_valid_time'].min()} ~ {agg['gfs_valid_time'].max()}")
        return agg

    def align_to_15min(self,
                       main_df: pd.DataFrame,
                       gfs_hourly: pd.DataFrame,
                       weather_features: List[str] = None) -> pd.DataFrame:
        logger.info("  Aligning GFS hourly to 15-minute resolution ...")

        main = main_df.copy()

        if not isinstance(main.index, pd.DatetimeIndex):
            if "datetime" in main.columns:
                main["datetime"] = pd.to_datetime(main["datetime"], errors="coerce")
                if main["datetime"].dt.tz is None:
                    main["datetime"] = main["datetime"].dt.tz_localize(TIMEZONE)
                main = main.set_index("datetime")

        if weather_features is None:
            meta_exclude = {"TIME_FCST", "TIME_FORECAST", "CITY_NAME", "CITY_CODE",
                           "LON", "LAT", "gfs_run_time", "gfs_valid_time", "lead_hour"}
            weather_features = [c for c in gfs_hourly.columns if c not in meta_exclude]

        gfs_indexed = gfs_hourly.set_index("gfs_valid_time").sort_index()
        gfs_valid_times = gfs_indexed.index.unique()

        main["target_time"] = main.index
        main["gfs_valid_time"] = main.index.floor("h")
        main["gfs_run_time"] = pd.Series(pd.NaT, index=main.index, dtype="datetime64[ns, Asia/Shanghai]")
        main["lead_hour"] = np.nan
        main["quarter_in_hour"] = (main.index.minute // 15).fillna(0).astype(int)
        main["weather_fill_mode"] = self.config.fill_mode

        run_time_lookup = self._build_run_time_lookup(gfs_hourly)

        for col in weather_features:
            main[col] = np.nan

        n_total = len(main)
        n_aligned = 0
        n_missing = 0

        for idx, row in main.iterrows():
            target_time = row["target_time"]
            valid_hour = row["gfs_valid_time"]

            best_run = self._find_latest_available_run(
                target_time, valid_hour, run_time_lookup
            )

            if best_run is not None:
                main.at[idx, "gfs_run_time"] = best_run
                lead_h = int((valid_hour - best_run).total_seconds() / 3600)
                main.at[idx, "lead_hour"] = lead_h

                mask = (gfs_hourly["gfs_run_time"] == best_run) & (gfs_hourly["gfs_valid_time"] == valid_hour)
                matching = gfs_hourly.loc[mask]

                if len(matching) > 0:
                    weather_vals = matching[weather_features].iloc[0]
                    for col in weather_features:
                        main.at[idx, col] = weather_vals[col]
                    n_aligned += 1
                else:
                    n_missing += 1
            else:
                n_missing += 1

        if self.config.interpolate and self.config.fill_mode == "interpolate":
            main = self._interpolate_weather(main, weather_features)

        self.diagnostics.total_samples = n_total
        self.diagnostics.samples_with_weather = n_aligned
        self.diagnostics.samples_without_weather = n_missing

        lead_hours = main["lead_hour"].dropna()
        if len(lead_hours) > 0:
            self.diagnostics.lead_hour_stats = {
                "min": float(lead_hours.min()),
                "max": float(lead_hours.max()),
                "mean": float(lead_hours.mean()),
                "median": float(lead_hours.median()),
            }

        self._check_future_leak(main)
        self._check_same_hour_consistency(main, weather_features)

        logger.info(f"    Aligned: {n_aligned}/{n_total} samples have weather data")
        logger.info(f"    Missing weather: {n_missing}")
        logger.info(f"    Lead hour stats: {self.diagnostics.lead_hour_stats}")

        return main

    def _build_run_time_lookup(self, gfs_hourly: pd.DataFrame) -> Dict[pd.Timestamp, List[pd.Timestamp]]:
        lookup = {}
        for _, row in gfs_hourly.iterrows():
            valid_time = row["gfs_valid_time"]
            run_time = row["gfs_run_time"]
            if valid_time not in lookup:
                lookup[valid_time] = []
            if run_time not in lookup[valid_time]:
                lookup[valid_time].append(run_time)

        for valid_time in lookup:
            lookup[valid_time] = sorted(lookup[valid_time])

        return lookup

    def _find_latest_available_run(self,
                                   target_time: pd.Timestamp,
                                   valid_hour: pd.Timestamp,
                                   run_time_lookup: Dict) -> Optional[pd.Timestamp]:
        if valid_hour not in run_time_lookup:
            return None

        available_runs = run_time_lookup[valid_hour]

        if self.config.enforce_no_future_leak:
            cutoff = target_time.replace(
                hour=self.config.decision_cutoff_hour,
                minute=self.config.decision_cutoff_minute,
                second=0, microsecond=0,
            )

            if target_time.hour < self.config.decision_cutoff_hour or (
                target_time.hour == self.config.decision_cutoff_hour and
                target_time.minute < self.config.decision_cutoff_minute
            ):
                cutoff = cutoff - pd.Timedelta(days=1)
                cutoff = cutoff.replace(hour=self.config.decision_cutoff_hour,
                                       minute=self.config.decision_cutoff_minute)

            valid_runs = [r for r in available_runs if r <= cutoff]
        else:
            valid_runs = available_runs

        if not valid_runs:
            return None

        return valid_runs[-1]

    def _interpolate_weather(self, df: pd.DataFrame,
                             weather_features: List[str]) -> pd.DataFrame:
        logger.info("    Interpolating weather to 15-min ...")

        for col in weather_features:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear", limit=4)

        df["weather_fill_mode"] = "interpolate"
        return df

    def _check_future_leak(self, df: pd.DataFrame):
        if "gfs_run_time" not in df.columns or "target_time" not in df.columns:
            return

        valid = df["gfs_run_time"].notna()
        if valid.sum() == 0:
            return

        run_times = df.loc[valid, "gfs_run_time"]
        target_times = df.loc[valid, "target_time"]

        future_leak = run_times > target_times
        n_leak = future_leak.sum()

        self.diagnostics.future_leak_detected = int(n_leak)

        if n_leak > 0:
            logger.error(f"    FUTURE LEAK DETECTED: {n_leak} samples use GFS run from the future!")
        else:
            logger.info(f"    No future leak detected")

    def _check_same_hour_consistency(self, df: pd.DataFrame,
                                     weather_features: List[str]):
        if "gfs_valid_time" not in df.columns:
            return

        groups = df.groupby("gfs_valid_time")
        inconsistent = 0

        for valid_time, group in groups:
            if len(group) < 2:
                continue
            for col in weather_features:
                if col in group.columns:
                    vals = group[col].dropna()
                    if len(vals) > 1 and vals.nunique() > 1:
                        inconsistent += 1
                        break

        self.diagnostics.same_hour_consistency_check = (inconsistent == 0)

        if inconsistent > 0:
            logger.warning(f"    Same-hour consistency: {inconsistent} hours have inconsistent weather")
        else:
            logger.info(f"    Same-hour consistency: OK")

    def build_training_dataset(self,
                               aligned_df: pd.DataFrame,
                               target_col: str = "GREEN_REAL",
                               feature_cols: List[str] = None,
                               lag_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"  Building training dataset for target: {target_col} ...")

        time_features = self._build_time_features(aligned_df)

        if feature_cols is None:
            meta_exclude = {"target_time", "gfs_valid_time", "gfs_run_time",
                           "lead_hour", "quarter_in_hour", "weather_fill_mode",
                           "FORECAST_DATE", "TIME_POINT"}
            target_exclude = {"GREEN_REAL", "WIND_REAL", "LIGHT_REAL", "LOAD_REAL",
                             "PRICE_DAYAGO", "PRICE_REAL", "PRICE_R_D"}
            feature_cols = [c for c in aligned_df.columns
                           if c not in meta_exclude and c not in target_exclude
                           and aligned_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        X = aligned_df[feature_cols].copy()

        for col_name, col_data in time_features.items():
            X[col_name] = col_data

        if lag_cols is not None:
            for col in lag_cols:
                if col in aligned_df.columns:
                    for lag_name, shift_val in [("lag_1h", 4), ("lag_1d", 96)]:
                        X[f"{col}_{lag_name}"] = aligned_df[col].shift(shift_val)

        y = aligned_df[target_col] if target_col in aligned_df.columns else pd.Series(dtype=float)

        valid_mask = y.notna()
        for col in X.columns:
            valid_mask = valid_mask & X[col].notna()

        X = X[valid_mask].fillna(0)
        y = y[valid_mask]

        logger.info(f"    Training dataset: X={X.shape}, y={y.shape}")
        return X, y

    def _build_time_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        features = {}

        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx_local = idx.tz_convert(TIMEZONE)
        else:
            idx_local = idx

        features["quarter_in_hour"] = (idx_local.minute // 15).fillna(0).astype(int)
        features["hour_of_day"] = idx_local.hour.fillna(0).astype(int)
        features["sin_hour"] = np.sin(2 * np.pi * features["hour_of_day"] / 24)
        features["cos_hour"] = np.cos(2 * np.pi * features["hour_of_day"] / 24)
        features["quarter_index"] = (features["hour_of_day"] * 4 + features["quarter_in_hour"]).astype(int)
        features["sin_quarter"] = np.sin(2 * np.pi * features["quarter_index"] / 96)
        features["cos_quarter"] = np.cos(2 * np.pi * features["quarter_index"] / 96)
        features["day_of_year"] = idx_local.day_of_year.fillna(1).astype(int)
        features["sin_doy"] = np.sin(2 * np.pi * features["day_of_year"] / 365)
        features["cos_doy"] = np.cos(2 * np.pi * features["day_of_year"] / 365)
        features["month"] = idx_local.month.fillna(1).astype(int)
        features["weekday"] = idx_local.weekday.fillna(0).astype(int)

        if "lead_hour" in df.columns:
            features["lead_hour"] = df["lead_hour"].fillna(0).astype(int)

        return features

    def export_training_sets(self,
                             aligned_df: pd.DataFrame,
                             output_dir: str = "./data/output"):
        import os
        os.makedirs(output_dir, exist_ok=True)

        for target, name in [("WIND_REAL", "wind"), ("LIGHT_REAL", "solar"), ("GREEN_REAL", "renewable")]:
            if target in aligned_df.columns:
                X, y = self.build_training_dataset(aligned_df, target_col=target)

                dataset = X.copy()
                dataset["target"] = y.values

                path = os.path.join(output_dir, f"gfs_aligned_{name}_train.parquet")
                dataset.to_parquet(path)
                logger.info(f"    Exported {name} training set: {dataset.shape} -> {path}")

    def get_diagnostics(self) -> AlignDiagnostics:
        return self.diagnostics

    def print_diagnostics(self):
        d = self.diagnostics
        logger.info("=" * 60)
        logger.info("GFS ALIGNMENT DIAGNOSTICS")
        logger.info("=" * 60)
        logger.info(f"  Total samples: {d.total_samples}")
        logger.info(f"  Samples with weather: {d.samples_with_weather}")
        logger.info(f"  Samples without weather: {d.samples_without_weather}")
        logger.info(f"  Future leak detected: {d.future_leak_detected}")
        logger.info(f"  Same-hour consistency: {'OK' if d.same_hour_consistency_check else 'FAILED'}")
        if d.lead_hour_stats:
            logger.info(f"  Lead hour stats: {d.lead_hour_stats}")
