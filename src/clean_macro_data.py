#!/usr/bin/env python3
"""
Enhanced macro-data ingestion with raw source support and backshift detection
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.raw_schema_validator import SchemaMismatchError, validate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_news_dir(news_dir: str) -> pd.DataFrame:
    """Load news data from directory of files"""
    news_dir = Path(news_dir)
    if not news_dir.exists():
        logger.warning(f"News directory not found: {news_dir}")
        return pd.DataFrame()

    dfs = []
    for file_path in news_dir.glob("*.csv"):
        try:
            df = pd.read_csv(file_path, parse_dates=["date"])
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def detect_backshift(name: str, proc_df: pd.DataFrame, raw_df: pd.DataFrame) -> bool:
    """Detect if raw data is newer than processed data"""
    if proc_df.empty or raw_df.empty:
        return False

    max_proc = proc_df["date"].max()
    max_raw = raw_df["date"].max()

    if max_raw != max_proc:
        logger.warning(f"{name} backshift: processed max={max_proc}, raw max={max_raw}")
        return True
    return False


def load_data_source(
    name: str, raw_path: str, parquet_path: str, use_raw: bool, is_json: bool = False
) -> pd.DataFrame:
    """Load data from raw or parquet source with fallback logic"""
    # Load processed data for comparison
    proc_df = pd.DataFrame()
    if os.path.exists(parquet_path):
        try:
            proc_df = pd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning(f"Failed to load processed {name}: {e}")

    # Load raw data
    raw_df = pd.DataFrame()
    if os.path.exists(raw_path):
        try:
            if is_json:
                import json

                with open(raw_path, "r") as f:
                    data = json.load(f)
                # Handle nested VIX JSON structure
                if "observations" in data:
                    raw_df = pd.DataFrame(data["observations"])
                else:
                    raw_df = pd.read_json(raw_path)
                if "date" in raw_df.columns:
                    raw_df["date"] = pd.to_datetime(raw_df["date"])
            else:
                if raw_path.endswith(".parquet"):
                    raw_df = pd.read_parquet(raw_path)
                else:
                    raw_df = pd.read_csv(raw_path, parse_dates=["date"])
        except Exception as e:
            logger.warning(f"Failed to load raw {name}: {e}")

    # Decide which source to use
    if use_raw:
        logger.info(f"Using raw {name} data (--use-raw-macro flag)")
        return raw_df

    # Auto-fallback logic
    if proc_df.empty and not raw_df.empty:
        logger.info(f"No processed {name} data, falling back to raw")
        return raw_df

    if not proc_df.empty and not raw_df.empty:
        if detect_backshift(name, proc_df, raw_df):
            logger.info(f"Backshift detected for {name}, falling back to raw")
            return raw_df

    if not proc_df.empty:
        logger.info(f"Using processed {name} data")
        return proc_df

    logger.warning(f"No data available for {name}")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Enhanced macro-data ingestion")
    parser.add_argument(
        "--use-raw-macro", action="store_true", help="Force use of raw macro data"
    )
    parser.add_argument(
        "--raw-fred-csv",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/FRED.csv",
    )
    parser.add_argument(
        "--raw-vix-json",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/vix_data.parquet",
    )
    parser.add_argument(
        "--raw-dxy-csv",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/DXY.csv",
    )
    parser.add_argument(
        "--raw-news-dir",
        default="/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data/Raw/news",
    )

    args = parser.parse_args()

    # Define output paths
    base_path = "/Users/amroheidak/Desktop/conviction-ai-clean/data/Parquet_data"
    fred_out = f"{base_path}/fred.parquet"
    vix_out = f"{base_path}/vix_data.parquet"
    dxy_out = f"{base_path}/dxy.parquet"
    news_out = f"{base_path}/news_data.parquet"

    # Load FRED data
    fred_df = load_data_source("FRED", args.raw_fred_csv, fred_out, args.use_raw_macro)

    # Load VIX data
    vix_df = load_data_source(
        "VIX", args.raw_vix_json, vix_out, args.use_raw_macro, is_json=False
    )

    # Calculate VIX MA divergence if VIX data available
    if not vix_df.empty and "date" in vix_df.columns:
        vix_df = vix_df.sort_values("date")
        if "close" in vix_df.columns:
            vix_df["vix_ma_10"] = (
                vix_df["close"].rolling(window=10, min_periods=1).mean()
            )
            vix_df["vix_ma_20"] = (
                vix_df["close"].rolling(window=20, min_periods=1).mean()
            )
            vix_df["vix_ma_divergence"] = (
                vix_df["close"] - vix_df["vix_ma_10"]
            ) / vix_df["vix_ma_10"]
        elif "value" in vix_df.columns:
            # Convert value column to numeric, handling '.' as NaN
            vix_df["value"] = pd.to_numeric(vix_df["value"], errors="coerce")
            vix_df["vix_ma_10"] = (
                vix_df["value"].rolling(window=10, min_periods=1).mean()
            )
            vix_df["vix_ma_20"] = (
                vix_df["value"].rolling(window=20, min_periods=1).mean()
            )
            vix_df["vix_ma_divergence"] = (
                vix_df["value"] - vix_df["vix_ma_10"]
            ) / vix_df["vix_ma_10"]

    # Load DXY data
    dxy_df = load_data_source("DXY", args.raw_dxy_csv, dxy_out, args.use_raw_macro)

    # Load News data
    if args.use_raw_macro or not os.path.exists(news_out):
        news_df = load_news_dir(args.raw_news_dir)
        if not news_df.empty:
            logger.info("Using raw news data")
        else:
            logger.info("Loading processed news data")
            news_df = (
                pd.read_parquet(news_out)
                if os.path.exists(news_out)
                else pd.DataFrame()
            )
    else:
        news_df = (
            pd.read_parquet(news_out) if os.path.exists(news_out) else pd.DataFrame()
        )

    # Write unified parquet files
    if not fred_df.empty:
        fred_df.to_parquet(fred_out, index=False)
        logger.info(f"Written FRED data: {len(fred_df)} rows")

    if not vix_df.empty:
        vix_df.to_parquet(vix_out, index=False)
        logger.info(f"Written VIX data: {len(vix_df)} rows")

    if not dxy_df.empty:
        dxy_df.to_parquet(dxy_out, index=False)
        logger.info(f"Written DXY data: {len(dxy_df)} rows")

    if not news_df.empty:
        news_df.to_parquet(news_out, index=False)
        logger.info(f"Written news data: {len(news_df)} rows")


if __name__ == "__main__":
    main()
