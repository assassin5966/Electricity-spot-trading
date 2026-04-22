PIPELINE_NAME = "power_spot_market_data_pipeline"
TIME_RESOLUTION = "15min"
POINTS_PER_DAY = 96
TIMEZONE = "Asia/Shanghai"
SNAPSHOT_CUTOFF = "D-1 09:30:00"

ORACLE_CONNECTIONS = {
    "main": "dm/ZBwis@2022@172.16.168.52:31521/zbdwpdb",
    "weather": "zb_weather/zbWeather2025@172.16.168.59:31521/zb_weather",
}

MAIN_TABLE = "BI_SX02_SNXH_ALL_DAYAGO_REAL"
WEATHER_TABLE = "DM_NOAA_GFS_SX02_CITY_TIME02_ALL"

RAW_DATA_DIR = "./data/raw"
OUTPUT_DIR = "./data/output"
MODEL_SAVE_DIR = "./saved_models"

SPLIT_DATES = {
    "train": ("2025-08-25", "2026-03-31"),
    "valid": ("2026-04-01", "2026-04-20"),
    "test": ("2026-04-21", "2026-04-27"),
    "may_focus": ("2026-05-01", "2026-05-10"),
}

QUANTILES = [0.1, 0.5, 0.9]
QUANTILE_LABELS = ["P10", "P50", "P90"]

EXPERT_TYPES = ["normal_day", "holiday", "post_holiday"]

MISSING_MASK_COLUMNS = [
    "PRICE_DAYAGO", "WATER_DAYAGO", "NOMARKET_DAYAGO", "LINE_DAYAGO",
]
MISSING_MASK_LENGTHS = [1, 3, 5]

LAG_STEPS = {
    "lag_15min": 1,
    "lag_1h": 4,
    "lag_1d": 96,
    "lag_7d": 672,
    "lag_15d": 1440,
}
LAG_APPLY_TO = ["LOAD_DAYAGO", "GREEN_DAYAGO", "PRICE_DAYAGO", "WIND_DAYAGO", "LIGHT_DAYAGO", "net_load_dayago"]

ROLLING_WINDOWS = {
    "rolling_mean_1d": (96, "mean"),
    "rolling_std_1d": (96, "std"),
    "rolling_mean_7d": (672, "mean"),
    "rolling_std_7d": (672, "std"),
}
ROLLING_APPLY_TO = ["LOAD_DAYAGO", "GREEN_DAYAGO", "PRICE_DAYAGO"]

TRAINING_WEIGHTS = {
    "recent_30d": 2.0,
    "holiday": 2.5,
    "extreme_price": 2.0,
    "extreme_load": 2.0,
    "abnormal_renewable": 2.0,
}

ENSEMBLE_WEIGHTS = {
    "deep_model": 0.4,
    "gbdt_model": 0.4,
    "simple_baseline": 0.2,
}

PRICE_LOSS_WEIGHTS = {
    "quantile_load": 0.5,
    "quantile_price": 0.3,
    "spike_classification": 0.2,
}

TARGET_COLUMNS = ["LOAD_REAL", "PRICE_DAYAGO", "PRICE_REAL", "PRICE_R_D"]

LIGHTGBM_PARAMS = {
    "objective": "quantile",
    "metric": "quantile",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 500,
}

DEEP_MODEL_PARAMS = {
    "hidden_size": 128,
    "num_heads": 4,
    "dropout": 0.1,
    "encoder_steps": 672,
    "decoder_steps": 96,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "max_epochs": 100,
    "patience": 10,
}
