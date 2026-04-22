import os
import logging
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./logs/pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def run_pipeline(skip_download: bool = False):
    os.makedirs("./logs", exist_ok=True)

    if not skip_download:
        from download_data import run_download
        logger.info("\n")
        main_df, weather_df = run_download()
    else:
        logger.info("Skipping download, using existing raw data")
        main_df, weather_df = None, None

    from features import run_feature_pipeline
    logger.info("\n")
    df = run_feature_pipeline(main_df, weather_df)

    logger.info(f"\nPipeline complete. Feature master table shape: {df.shape}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    run_pipeline(skip_download=args.skip_download)
