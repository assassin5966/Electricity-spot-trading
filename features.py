import os
import logging
import pandas as pd
import numpy as np

from config import (
    RAW_DATA_DIR, OUTPUT_DIR, TIMEZONE, POINTS_PER_DAY,
    LAG_STEPS, LAG_APPLY_TO, ROLLING_WINDOWS, ROLLING_APPLY_TO,
    MISSING_MASK_COLUMNS, SPLIT_DATES, TARGET_COLUMNS,
)
from gfs_features import (
    WIND_RAW, SOLAR_RAW, TEMP_RAW,
    PLEVEL_TEMP, PLEVEL_RH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data():
    main_path = os.path.join(RAW_DATA_DIR, "main_table.parquet")
    weather_path = os.path.join(RAW_DATA_DIR, "weather.parquet")
    main_df = pd.read_parquet(main_path) if os.path.exists(main_path) else None
    weather_df = pd.read_parquet(weather_path) if os.path.exists(weather_path) else None
    return main_df, weather_df


def unify_time_index(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  Unifying time index ...")
    df = df.dropna(subset=["datetime"]).copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(TIMEZONE)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(TIMEZONE)

    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    full_range = pd.date_range(
        start=df.index.min().normalize(),
        end=df.index.max().normalize() + pd.Timedelta(hours=23, minutes=45),
        freq="15min", tz=TIMEZONE,
    )
    df = df.reindex(full_range)

    missing_mask = df.isnull().all(axis=1)
    n_missing = missing_mask.sum()
    if n_missing > 0:
        logger.info(f"    {n_missing} missing timestamps, forward-filling non-target columns")
        target_set = set(TARGET_COLUMNS)
        non_target = [c for c in df.columns if c not in target_set]
        df[non_target] = df[non_target].ffill()

    logger.info(f"    After alignment: {len(df)} rows")
    return df


def aggregate_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  Aggregating weather data (full GFS) ...")
    if weather_df is None or weather_df.empty:
        logger.warning("    No weather data, returning empty")
        return pd.DataFrame()

    df = weather_df.copy()
    df["TIME_FORECAST"] = pd.to_datetime(df["TIME_FORECAST"], errors="coerce")
    df = df.dropna(subset=["TIME_FORECAST"])

    if df["TIME_FORECAST"].dt.tz is None:
        df["TIME_FORECAST"] = df["TIME_FORECAST"].dt.tz_localize(TIMEZONE)
    else:
        df["TIME_FORECAST"] = df["TIME_FORECAST"].dt.tz_convert(TIMEZONE)

    meta_cols = {"TIME_FCST", "TIME_FORECAST", "CITY_NAME", "CITY_CODE", "LON", "LAT"}
    numeric_cols = [c for c in df.columns if c not in meta_cols]

    logger.info(f"    Raw GFS columns: {len(numeric_cols)}, aggregating by city (mean only) ...")

    agg = df.groupby("TIME_FORECAST")[numeric_cols].mean().reset_index()
    agg = agg.rename(columns={"TIME_FORECAST": "datetime"})
    agg = agg.set_index("datetime").sort_index()

    for name, (u_col, v_col) in WIND_RAW.items():
        if u_col in agg.columns and v_col in agg.columns:
            agg[f"ws_{name}"] = np.sqrt(agg[u_col]**2 + agg[v_col]**2)
            agg[f"wd_{name}"] = np.degrees(np.arctan2(agg[v_col], agg[u_col])) % 360

    if "ws_10m" in agg.columns and "ws_100m" in agg.columns:
        agg["wind_shear_100m_10m"] = agg["ws_100m"] - agg["ws_10m"]

    if "ws_850mb" in agg.columns and "ws_10m" in agg.columns:
        agg["wind_shear_850mb_surface"] = agg["ws_850mb"] - agg["ws_10m"]

    if "ws_10m" in agg.columns:
        agg["ws_10m_sq"] = agg["ws_10m"] ** 2
        agg["ws_10m_cu"] = agg["ws_10m"] ** 3

    if "ws_100m" in agg.columns:
        agg["ws_100m_sq"] = agg["ws_100m"] ** 2
        agg["ws_100m_cu"] = agg["ws_100m"] ** 3

    if "wd_10m" in agg.columns:
        agg["wd_10m_change"] = agg["wd_10m"].diff()
        agg["wd_10m_change"] = ((agg["wd_10m_change"] + 180) % 360) - 180

    for name, col in SOLAR_RAW.items():
        if col in agg.columns:
            agg[name] = agg[col]

    if "dswrf" in agg.columns:
        agg["dswrf_sq"] = agg["dswrf"] ** 2

    if "tcdc_total" in agg.columns:
        agg["clear_sky_index"] = 1 - agg["tcdc_total"] / 100

    for name, col in TEMP_RAW.items():
        if col in agg.columns:
            agg[name] = agg[col]

    for level, col in PLEVEL_TEMP.items():
        if col in agg.columns:
            agg[f"tmp_{level}"] = agg[col]

    for level, col in PLEVEL_RH.items():
        if col in agg.columns:
            agg[f"rh_{level}"] = agg[col]

    if "tmp_2m" in agg.columns and "dpt_2m" in agg.columns:
        agg["temp_dew_spread"] = agg["tmp_2m"] - agg["dpt_2m"]

    if "tmax" in agg.columns and "tmin" in agg.columns:
        agg["temp_diurnal_range"] = agg["tmax"] - agg["tmin"]

    if "CAPE_624" in agg.columns:
        agg["cape_surface"] = agg["CAPE_624"]
    if "CIN_625" in agg.columns:
        agg["cin_surface"] = agg["CIN_625"]
    if "HPBL_712" in agg.columns:
        agg["pbl_height"] = agg["HPBL_712"]
    if "PRATE_593" in agg.columns:
        agg["precip_rate"] = agg["PRATE_593"]
    if "APCP_596" in agg.columns:
        agg["precip_accum"] = agg["APCP_596"]

    raw_gfs_cols = [c for c in numeric_cols if c in agg.columns]
    agg = agg.drop(columns=raw_gfs_cols)

    key_vars = ["ws_10m", "ws_100m", "dswrf", "tmp_2m", "cape_surface",
                "wind_shear_100m_10m", "clear_sky_index", "tcdc_total"]
    for var in key_vars:
        if var in agg.columns:
            agg[f"{var}_lag_1d"] = agg[var].shift(96)
            agg[f"{var}_diff_1d"] = agg[var] - agg[f"{var}_lag_1d"]
            agg[f"{var}_rolling_mean_1d"] = agg[var].rolling(96, min_periods=1).mean()
            agg[f"{var}_rolling_std_1d"] = agg[var].rolling(96, min_periods=1).std()

    for lag in [1, 3, 6]:
        for var in ["ws_10m", "ws_100m", "dswrf", "tmp_2m"]:
            if var in agg.columns:
                agg[f"{var}_lag_{lag}"] = agg[var].shift(lag)

    logger.info(f"    Weather aggregated: {len(agg)} rows, {len(agg.columns)} columns")
    return agg


def merge_weather(main_df: pd.DataFrame, weather_agg: pd.DataFrame) -> pd.DataFrame:
    logger.info("  Merging weather ...")
    if weather_agg.empty:
        logger.warning("    No weather data to merge")
        return main_df

    weather_reindexed = weather_agg.reindex(main_df.index, method="nearest", tolerance=pd.Timedelta("1h"))
    for c in weather_reindexed.columns:
        main_df[c] = weather_reindexed[c].values

    logger.info(f"    Merged weather: added {len(weather_agg.columns)} columns")
    return main_df


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  Building calendar features ...")
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx_local = idx.tz_convert(TIMEZONE)
    else:
        idx_local = idx

    df["quarter_index"] = (idx_local.hour * 4 + idx_local.minute // 15).astype(int)
    df["hour"] = idx_local.hour
    df["weekday"] = idx_local.weekday
    df["is_weekend"] = (idx_local.weekday >= 5).astype(int)
    df["month"] = idx_local.month
    df["day_of_month"] = idx_local.day
    df["day_of_year"] = idx_local.day_of_year
    df["sin_hour"] = np.sin(2 * np.pi * df["quarter_index"] / 96)
    df["cos_hour"] = np.cos(2 * np.pi * df["quarter_index"] / 96)
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    try:
        import holidays
        years = range(idx_local.year.min(), idx_local.year.max() + 1)
        cn_holidays = holidays.CountryHoliday("CN", years=years)
        dates = idx_local.normalize()
        df["is_holiday"] = dates.map(lambda x: 1 if x in cn_holidays else 0).astype(int)
    except ImportError:
        logger.warning("    holidays library not installed")
        df["is_holiday"] = 0

    holiday_dates = set()
    try:
        import holidays as h
        cn = h.CountryHoliday("CN", years=range(idx_local.year.min(), idx_local.year.max() + 1))
        for d in cn:
            holiday_dates.add(pd.Timestamp(d).date())
    except Exception:
        pass

    date_series = idx_local.date
    df["is_post_holiday"] = 0
    df["is_pre_holiday"] = 0
    df["holiday_day_index"] = 0

    prev_date = None
    holiday_streak = 0
    for i, d in enumerate(date_series):
        if d in holiday_dates:
            holiday_streak += 1
            df.iloc[i, df.columns.get_loc("holiday_day_index")] = holiday_streak
        else:
            holiday_streak = 0

        if prev_date is not None:
            if prev_date in holiday_dates and d not in holiday_dates:
                df.iloc[i, df.columns.get_loc("is_post_holiday")] = 1

        next_day = pd.Timestamp(d) + pd.Timedelta(days=1)
        if next_day.date() in holiday_dates and d not in holiday_dates:
            df.iloc[i, df.columns.get_loc("is_pre_holiday")] = 1

        prev_date = d

    logger.info(f"    Calendar features: holiday={df['is_holiday'].sum()}, "
                f"post_holiday={df['is_post_holiday'].sum()}, pre_holiday={df['is_pre_holiday'].sum()}")
    return df


def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  Building derived features ...")

    if "LOAD_DAYAGO" in df.columns and "GREEN_DAYAGO" in df.columns:
        df["net_load_dayago"] = df["LOAD_DAYAGO"] - df["GREEN_DAYAGO"]
    if "LOAD_REAL" in df.columns and "GREEN_REAL" in df.columns:
        df["net_load_real"] = df["LOAD_REAL"] - df["GREEN_REAL"]

    if "GREEN_DAYAGO" in df.columns and "LOAD_DAYAGO" in df.columns:
        df["renewable_ratio_dayago"] = df["GREEN_DAYAGO"] / df["LOAD_DAYAGO"].replace(0, np.nan)
    if "GREEN_REAL" in df.columns and "LOAD_REAL" in df.columns:
        df["renewable_ratio_real"] = df["GREEN_REAL"] / df["LOAD_REAL"].replace(0, np.nan)

    for bias_col, src_col in [
        ("load_bias", "LOAD_R_D"),
        ("renewable_bias", "GREEN_R_D"),
        ("price_bias", "PRICE_R_D"),
    ]:
        if src_col in df.columns:
            df[bias_col] = df[src_col]

    if "THERMAL_REAL" in df.columns and "THERMAL_DAYAGO" in df.columns:
        df["thermal_bias"] = df["THERMAL_REAL"] - df["THERMAL_DAYAGO"]

    if "THERMAL_REAL" in df.columns and "UNIT_CAPACITY" in df.columns:
        df["thermal_share"] = df["THERMAL_REAL"] / df["UNIT_CAPACITY"].replace(0, np.nan)

    for gap_name, real_col, dayago_col in [
        ("load_rate_gap", "LOAD_RATE_REAL", "LOAD_RATE_DAYAGO"),
        ("load_rate_water_gap", "LOAD_RATE_WATER_REAL", "LOAD_RATE_WATER_DAYAGO"),
        ("load_rate_n_gap", "LOAD_RATE_N_REAL", "LOAD_RATE_N_DAYAGO"),
        ("load_rate_n_w_gap", "LOAD_RATE_N_W_REAL", "LOAD_RATE_N_W_DAYAGO"),
    ]:
        if real_col in df.columns and dayago_col in df.columns:
            df[gap_name] = pd.to_numeric(df[real_col], errors="coerce") - pd.to_numeric(df[dayago_col], errors="coerce")

    logger.info(f"    Derived features added")
    return df


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  Building lag features ...")
    for col in LAG_APPLY_TO:
        if col not in df.columns:
            continue
        for lag_name, shift_val in LAG_STEPS.items():
            feat_name = f"{col}_{lag_name}"
            df[feat_name] = df[col].shift(shift_val)

    logger.info(f"    Lag features: {len(LAG_APPLY_TO)} columns x {len(LAG_STEPS)} lags")
    return df


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  Building rolling features ...")
    for col in ROLLING_APPLY_TO:
        if col not in df.columns:
            continue
        for win_name, (window, agg_fn) in ROLLING_WINDOWS.items():
            feat_name = f"{col}_{win_name}"
            if agg_fn == "mean":
                df[feat_name] = df[col].rolling(window=window, min_periods=1).mean()
            elif agg_fn == "std":
                df[feat_name] = df[col].rolling(window=window, min_periods=1).std()

    logger.info(f"    Rolling features: {len(ROLLING_APPLY_TO)} columns x {len(ROLLING_WINDOWS)} windows")
    return df


def build_missing_simulation_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("  Building missing simulation features ...")
    df["mask_flag"] = 0
    df["missing_length"] = 0
    df["availability_flag"] = 1

    for col in MISSING_MASK_COLUMNS:
        if col in df.columns:
            is_missing = df[col].isna()
            df.loc[is_missing, "mask_flag"] = 1
            df.loc[is_missing, "availability_flag"] = 0

    missing_groups = (df["mask_flag"] != df["mask_flag"].shift()).cumsum()
    for _, group in df.groupby(missing_groups):
        if group["mask_flag"].iloc[0] == 1:
            length = max(1, len(group) // POINTS_PER_DAY)
            df.loc[group.index, "missing_length"] = length

    logger.info(f"    mask_flag=1: {df['mask_flag'].sum()}, availability_flag=0: {(df['availability_flag']==0).sum()}")
    return df


def check_data_quality(df: pd.DataFrame):
    logger.info("  Checking data quality ...")
    dates = df.index.date
    unique_dates = sorted(set(dates))

    for d in unique_dates:
        day_count = (dates == d).sum()
        if day_count != POINTS_PER_DAY:
            logger.warning(f"    {d}: {day_count} points (expected {POINTS_PER_DAY})")

    duplicated = df.index.duplicated().sum()
    if duplicated > 0:
        logger.warning(f"    {duplicated} duplicated timestamps")

    for col in TARGET_COLUMNS:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            logger.info(f"    {col}: {null_count} nulls / {len(df)} total")

    logger.info("  Data quality check complete")


def split_and_export(df: pd.DataFrame):
    logger.info("  Splitting and exporting datasets ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    target_cols = [c for c in TARGET_COLUMNS if c in df.columns]

    df.to_parquet(os.path.join(OUTPUT_DIR, "feature_master_table.parquet"))
    logger.info(f"    feature_master_table: {df.shape}")

    for split_name, (start, end) in SPLIT_DATES.items():
        start_ts = pd.Timestamp(start, tz=TIMEZONE)
        end_ts = pd.Timestamp(end, tz=TIMEZONE) + pd.Timedelta(hours=23, minutes=45)
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        split_df = df[mask]

        if len(split_df) == 0:
            logger.warning(f"    {split_name}: no data")
            continue

        flat_path = os.path.join(OUTPUT_DIR, f"gbdt_{split_name}.parquet")
        split_df.to_parquet(flat_path)
        logger.info(f"    GBDT {split_name}: {split_df.shape} -> {flat_path}")

        if len(split_df) >= POINTS_PER_DAY * 7:
            n_days = len(split_df) // POINTS_PER_DAY - 6
            X_list, y_list = [], []
            for i in range(n_days):
                start_idx = i * POINTS_PER_DAY
                end_idx = start_idx + 7 * POINTS_PER_DAY
                if end_idx > len(split_df):
                    break
                X_list.append(split_df[feature_cols].iloc[start_idx:end_idx].values)
                y_list.append(split_df[target_cols].iloc[end_idx - POINTS_PER_DAY:end_idx].values)

            if X_list:
                X = np.stack(X_list)
                y = np.stack(y_list)
                np.save(os.path.join(OUTPUT_DIR, f"deep_{split_name}_X.npy"), X)
                np.save(os.path.join(OUTPUT_DIR, f"deep_{split_name}_y.npy"), y)
                logger.info(f"    Deep {split_name}: X={X.shape}, y={y.shape}")

        if "expert_label" not in split_df.columns and "is_holiday" in split_df.columns:
            split_df = split_df.copy()
            split_df["expert_label"] = "normal_day"
            split_df.loc[split_df["is_holiday"] == 1, "expert_label"] = "holiday"
            split_df.loc[split_df["is_post_holiday"] == 1, "expert_label"] = "post_holiday"

        if "expert_label" in split_df.columns:
            moe_path = os.path.join(OUTPUT_DIR, f"moe_{split_name}.parquet")
            split_df.to_parquet(moe_path)

    pd.DataFrame({"feature_cols": feature_cols}).to_parquet(
        os.path.join(OUTPUT_DIR, "feature_columns.parquet"))
    pd.DataFrame({"target_cols": target_cols}).to_parquet(
        os.path.join(OUTPUT_DIR, "target_columns.parquet"))


def run_feature_pipeline(main_df: pd.DataFrame = None, weather_df: pd.DataFrame = None):
    logger.info("=" * 60)
    logger.info("FEATURE PIPELINE")
    logger.info("=" * 60)

    if main_df is None or weather_df is None:
        main_df, weather_df = load_raw_data()

    if main_df is None:
        raise FileNotFoundError("No main table data found")

    df = unify_time_index(main_df)
    weather_agg = aggregate_weather(weather_df)
    df = merge_weather(df, weather_agg)
    df = build_calendar_features(df)
    df = build_derived_features(df)
    df = build_lag_features(df)
    df = build_rolling_features(df)
    df = build_missing_simulation_features(df)

    check_data_quality(df)
    split_and_export(df)

    logger.info("=" * 60)
    logger.info("FEATURE PIPELINE COMPLETE")
    logger.info(f"  Final shape: {df.shape}")
    logger.info("=" * 60)
    return df


if __name__ == "__main__":
    run_feature_pipeline()
