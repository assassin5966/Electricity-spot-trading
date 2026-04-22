import os
import logging
import cx_Oracle
import pandas as pd
import numpy as np

from config import ORACLE_CONNECTIONS, MAIN_TABLE, WEATHER_TABLE, RAW_DATA_DIR, TIMEZONE
from gfs_features import get_download_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_main_table() -> pd.DataFrame:
    logger.info(f"Downloading {MAIN_TABLE} ...")
    conn = cx_Oracle.connect(ORACLE_CONNECTIONS["main"])
    try:
        df = pd.read_sql(f"SELECT * FROM {MAIN_TABLE}", con=conn)
        logger.info(f"  Downloaded: {len(df)} rows, {len(df.columns)} columns")
    finally:
        conn.close()

    df["FORECAST_DATE"] = pd.to_datetime(df["FORECAST_DATE"])
    df["datetime"] = pd.to_datetime(
        df["FORECAST_DATE"].dt.strftime("%Y-%m-%d") + " " + df["TIME_POINT"].astype(str),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )

    drop_cols = ["ID", "CREATE_TIME", "UPDATE_TIME"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    numeric_cols = [c for c in df.columns if c not in ["FORECAST_DATE", "TIME_POINT", "datetime"]]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    path = os.path.join(RAW_DATA_DIR, "main_table.parquet")
    df.to_parquet(path, index=False)
    logger.info(f"  Saved -> {path}")
    return df


def download_weather() -> pd.DataFrame:
    logger.info(f"Downloading weather from {WEATHER_TABLE} ...")
    download_cols = get_download_columns()

    col_str = ", ".join(download_cols)
    sql = f"""SELECT {col_str}
              FROM {WEATHER_TABLE}
              WHERE TRUNC(TIME_FCST) = TRUNC(TIME_FORECAST)-1
              ORDER BY TIME_FCST, TIME_FORECAST, CITY_NAME"""

    conn = cx_Oracle.connect(ORACLE_CONNECTIONS["weather"])
    try:
        df = pd.read_sql(sql, con=conn)
        logger.info(f"  Downloaded: {len(df)} rows, {len(df.columns)} columns")
    finally:
        conn.close()

    df["TIME_FORECAST"] = pd.to_datetime(df["TIME_FORECAST"])
    df["TIME_FCST"] = pd.to_datetime(df["TIME_FCST"])

    meta_cols = {"TIME_FCST", "TIME_FORECAST", "CITY_NAME", "CITY_CODE", "LON", "LAT"}
    numeric_cols = [c for c in df.columns if c not in meta_cols]
    drop_cols = []
    for c in numeric_cols:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except (TypeError, ValueError):
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)
        numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    n_null = df[numeric_cols].isnull().sum()
    high_null = n_null[n_null > len(df) * 0.5]
    if len(high_null) > 0:
        logger.warning(f"  Columns with >50% null: {list(high_null.index)}")
        drop_null_cols = list(high_null.index)
        df = df.drop(columns=drop_null_cols)
        logger.info(f"  Dropped {len(drop_null_cols)} high-null columns")

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    path = os.path.join(RAW_DATA_DIR, "weather.parquet")
    df.to_parquet(path, index=False)
    logger.info(f"  Saved -> {path} ({len(df.columns)} columns)")
    return df


def run_download():
    logger.info("=" * 60)
    logger.info("STEP: download_data")
    logger.info("=" * 60)
    main_df = download_main_table()
    weather_df = download_weather()
    return main_df, weather_df


if __name__ == "__main__":
    run_download()
