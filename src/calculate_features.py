#!/usr/bin/env python3
"""
Feature calculation module using Polars for high-performance data processing.
Optimized for Apple Silicon M2 Ultra with 24 cores and 64GB RAM.
Generates final feature matrix from daily and intraday master datasets.
"""

import argparse
import concurrent.futures
import os
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl

from gpu_utils import (gpu_rolling_mean, gpu_rolling_std, gpu_supported,
                       optimize_for_apple_silicon, optimize_for_gpu)
from utils.lineage_utils import LineageTracker

# Optimize for M2 Ultra at module import
optimize_for_apple_silicon()

# 36-Stock Training Universe + ETF Options Targets
SECTOR_MAPPING = {
    # Tech: AAPL, GOOGL, META, MSFT, NFLX, SMCI
    "AAPL": "TECH",
    "GOOGL": "TECH",
    "META": "TECH",
    "MSFT": "TECH",
    "NFLX": "TECH",
    "SMCI": "TECH",
    # Tech (High IV): AMD, NVDA, PLTR
    "AMD": "TECH",
    "NVDA": "TECH",
    "PLTR": "TECH",
    # Financial: BAC, GS, JPM, MA, MS, V
    "BAC": "FINANCIAL",
    "GS": "FINANCIAL",
    "JPM": "FINANCIAL",
    "MA": "FINANCIAL",
    "MS": "FINANCIAL",
    "V": "FINANCIAL",
    # Healthcare: ABBV, JNJ, MRK, PFE, UNH
    "ABBV": "HEALTHCARE",
    "JNJ": "HEALTHCARE",
    "MRK": "HEALTHCARE",
    "PFE": "HEALTHCARE",
    "UNH": "HEALTHCARE",
    # Consumer: DIS, NKE, SBUX, WMT, AMZN
    "DIS": "CONSUMER",
    "NKE": "CONSUMER",
    "SBUX": "CONSUMER",
    "WMT": "CONSUMER",
    "AMZN": "CONSUMER",
    # Energy: CVX, XOM
    "CVX": "ENERGY",
    "XOM": "ENERGY",
    # Industrial: BA, CAT, GE
    "BA": "INDUSTRIAL",
    "CAT": "INDUSTRIAL",
    "GE": "INDUSTRIAL",
    # Crypto/Fintech (High IV): COIN, HOOD, MSTR
    "COIN": "CRYPTO",
    "HOOD": "FINTECH",
    "MSTR": "CRYPTO",
    # Auto (High IV): TSLA
    "TSLA": "AUTO",
    # ETF Options Targets: QQQ, SPY
    "QQQ": "ETF_TECH",
    "SPY": "ETF_MARKET",
}

# ETF reference mapping
ETF_REFERENCES = {
    "GOLD_PROXY": "GLD",
    "OIL_PROXIES": ["CVX", "XOM"],
    "MARKET_DIRECTION": ["SPY", "QQQ"],
    "SECTOR_ETFS": {
        "XLK": "TECH",
        "XLF": "FINANCIAL",
        "XLV": "HEALTHCARE",
        "XLY": "CONSUMER",
        "XLE": "ENERGY",
        "XLI": "INDUSTRIAL",
    },
}


def parse_date_range(date_str: str) -> Tuple[date, date]:
    """Parse date string - single date or range (YYYY-MM-DD,YYYY-MM-DD)"""
    if "," in date_str:
        start_str, end_str = date_str.split(",")
        return (
            datetime.strptime(start_str.strip(), "%Y-%m-%d").date(),
            datetime.strptime(end_str.strip(), "%Y-%m-%d").date(),
        )
    else:
        single_date = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
        return single_date, single_date


def calculate_rolling_features(
    df: pl.DataFrame, window: int, use_gpu: bool = False
) -> pl.DataFrame:
    """Calculate rolling features for a single ticker group with optional GPU acceleration"""
    df_sorted = df.sort("date")

    if use_gpu and gpu_supported():
        print("🚀 Using GPU acceleration for rolling features")
        return df_sorted.with_columns(
            [
                # GPU-accelerated rolling features
                gpu_rolling_mean(df_sorted, "fred_fed_funds_rate", window).alias(
                    "fred_rate_mean"
                ),
                gpu_rolling_std(df_sorted, "vix_index", window).alias("vix_std"),
                pl.col("news_count_lag1")
                .rolling_sum(window)
                .alias("news_count_rolling"),
                pl.col("avg_sentiment_lag1")
                .rolling_mean(window)
                .alias("avg_sentiment_rolling"),
                gpu_rolling_mean(df_sorted, "optd_iv30", window).alias(
                    "optd_iv30_mean"
                ),
                gpu_rolling_std(df_sorted, "optd_volume", window).alias(
                    "optd_volume_std"
                ),
                gpu_rolling_std(df_sorted, "stockd_return_1d", window).alias(
                    "stockd_vol_rolling"
                ),
                gpu_rolling_mean(df_sorted, "stockd_volume", window).alias(
                    "stockd_volume_mean"
                ),
            ]
        )
    else:
        return df_sorted.with_columns(
            [
                # CPU rolling features
                pl.col("fred_fed_funds_rate")
                .rolling_mean(window)
                .alias("fred_rate_mean"),
                pl.col("vix_index").rolling_std(window).alias("vix_std"),
                pl.col("news_count_lag1")
                .rolling_sum(window)
                .alias("news_count_rolling"),
                pl.col("avg_sentiment_lag1")
                .rolling_mean(window)
                .alias("avg_sentiment_rolling"),
                pl.col("optd_iv30").rolling_mean(window).alias("optd_iv30_mean"),
                pl.col("optd_volume").rolling_std(window).alias("optd_volume_std"),
                pl.col("stockd_return_1d")
                .rolling_std(window)
                .alias("stockd_vol_rolling"),
                pl.col("stockd_volume")
                .rolling_mean(window)
                .alias("stockd_volume_mean"),
            ]
        )


def process_ticker_chunk(ticker_chunk, dm, window):
    """Process a chunk of tickers in parallel"""
    chunk_data = dm.filter(pl.col("ticker").is_in(ticker_chunk))
    return chunk_data.group_by("ticker").map_groups(
        lambda df: calculate_rolling_features(df, window)
    )


def calculate_intraday_features(im):
    """Calculate intraday features including lagged returns and spikes"""
    # Pivot intraday data to wide format for return calculations
    pivot = im.select(["ticker", "timestamp", "opt30_mid_price"]).pivot(
        values="opt30_mid_price", index="timestamp", columns="ticker"
    )

    # Calculate 1-hour returns (2 × 30min periods)
    ret_cols = [col for col in pivot.columns if col != "timestamp"]
    ret_exprs = [(pl.col(col).pct_change(2)).alias(f"{col}_ret_1h") for col in ret_cols]

    pivot_ret = pivot.with_columns(ret_exprs)

    # Melt back to long format
    ret_1h = (
        pivot_ret.select(["timestamp"] + [f"{col}_ret_1h" for col in ret_cols])
        .melt(id_vars="timestamp", variable_name="ticker_ret", value_name="ret_1h")
        .with_columns(pl.col("ticker_ret").str.replace("_ret_1h", "").alias("ticker"))
        .select(["timestamp", "ticker", "ret_1h"])
    )

    return ret_1h


def calculate_sector_features(features):
    """Add sector classification and sector-based features"""
    return features.with_columns(
        [
            # Sector classification
            pl.col("ticker")
            .map_elements(
                lambda x: SECTOR_MAPPING.get(x, "OTHER"), return_dtype=pl.Utf8
            )
            .alias("sector"),
            # Sector binary features
            pl.col("ticker")
            .map_elements(
                lambda x: SECTOR_MAPPING.get(x, "OTHER") == "TECH",
                return_dtype=pl.Boolean,
            )
            .alias("is_tech"),
            pl.col("ticker")
            .map_elements(
                lambda x: SECTOR_MAPPING.get(x, "OTHER") == "FINANCIAL",
                return_dtype=pl.Boolean,
            )
            .alias("is_financial"),
            pl.col("ticker")
            .map_elements(
                lambda x: SECTOR_MAPPING.get(x, "OTHER") == "ENERGY",
                return_dtype=pl.Boolean,
            )
            .alias("is_energy"),
            # ETF flags for options signals
            pl.col("ticker")
            .map_elements(lambda x: x == "QQQ", return_dtype=pl.Boolean)
            .alias("is_qqq_etf"),
            pl.col("ticker")
            .map_elements(lambda x: x == "SPY", return_dtype=pl.Boolean)
            .alias("is_spy_etf"),
            # High IV flags (high volatility stocks)
            pl.col("ticker")
            .map_elements(
                lambda x: x in ["TSLA", "NVDA", "PLTR", "AMD", "COIN", "HOOD", "MSTR"],
                return_dtype=pl.Boolean,
            )
            .alias("is_high_iv"),
        ]
    )


def calculate_market_features(features):
    """Calculate market direction features using SPY/QQQ"""
    market_cols = []

    # SPY market features
    if "SPY" in features["ticker"].unique():
        spy_data = features.filter(pl.col("ticker") == "SPY")
        market_cols.extend(
            [
                spy_data.select(pl.col("stockd_return_1d").alias("spy_return_1d")),
                spy_data.select(pl.col("stockd_volume").alias("spy_volume")),
            ]
        )

    # QQQ tech features
    if "QQQ" in features["ticker"].unique():
        qqq_data = features.filter(pl.col("ticker") == "QQQ")
        market_cols.extend(
            [
                qqq_data.select(pl.col("stockd_return_1d").alias("qqq_return_1d")),
                qqq_data.select(pl.col("stockd_volume").alias("qqq_volume")),
            ]
        )

    # Add market beta (correlation with SPY)
    if "SPY" in features["ticker"].unique():
        spy_returns = features.filter(pl.col("ticker") == "SPY")["stockd_return_1d"]
        features = features.with_columns(
            [
                pl.col("stockd_return_1d")
                .rolling_corr(spy_returns.first(), window_size=20)
                .alias("market_beta_20d")
            ]
        )

    return features


def calculate_commodity_features(features):
    """Calculate commodity reference features"""
    commodity_cols = []

    # Gold features (GLD proxy)
    if "GLD" in features["ticker"].unique():
        gld_data = features.filter(pl.col("ticker") == "GLD")
        commodity_cols.extend(
            [
                gld_data.select(pl.col("stockd_return_1d").alias("gold_return_1d")),
                gld_data.select(
                    pl.col("stockd_vol_rolling").alias("gold_volatility_10d")
                ),
            ]
        )

    # Oil proxy (average of energy stocks)
    oil_proxies = [t for t in ["CVX", "XOM"] if t in features["ticker"].unique()]
    if oil_proxies:
        oil_returns = []
        for ticker in oil_proxies:
            oil_data = features.filter(pl.col("ticker") == ticker)
            oil_returns.append(oil_data["stockd_return_1d"])

        # Average oil return
        if oil_returns:
            oil_avg = sum(oil_returns) / len(oil_returns)
            commodity_cols.append({"oil_return_1d": oil_avg})

    return features


def calculate_cross_sectional_features(features):
    """Calculate cross-sectional z-scores and relative features"""
    return features.with_columns(
        [
            # Volume z-score across tickers at each timestamp
            (
                (pl.col("optd_volume") - pl.col("optd_volume").mean().over("date"))
                / pl.col("optd_volume").std().over("date")
            ).alias("vol_zscore"),
            # IV percentile across tickers
            pl.col("optd_iv30").rank().over("date").alias("iv_rank"),
            # IV rank 30d - percentile of current IV30 vs past 30 days per ticker
            (
                pl.col("optd_iv30")
                .rolling_quantile(quantile=0.5, window_size=30)
                .over("ticker")
            ).alias("iv_rank_30d"),
            # Return relative to market
            (
                pl.col("stockd_return_1d")
                - pl.col("stockd_return_1d").mean().over("date")
            ).alias("ret_relative"),
        ]
    )


def filter_training_universe(features):
    """Filter to individual stocks + QQQ/SPY for options signals"""
    # Individual stocks for training
    individual_stocks = [
        k for k, v in SECTOR_MAPPING.items() if not v.startswith("ETF_")
    ]

    # Add QQQ/SPY for options signal generation
    options_targets = ["QQQ", "SPY"]

    training_universe = individual_stocks + options_targets
    return features.filter(pl.col("ticker").is_in(training_universe))


def calculate_all_features(daily_master, intraday_master, window=30, use_gpu=False):
    """Calculate all features from daily and intraday master datasets"""
    # Optimize for GPU if requested
    daily_master = optimize_for_gpu(daily_master, use_gpu)
    intraday_master = optimize_for_gpu(intraday_master, use_gpu)

    # Calculate rolling features with GPU support
    roll_features = daily_master.group_by("ticker").map_groups(
        lambda df: calculate_rolling_features(df.sort("date"), window, use_gpu)
    )

    # Calculate intraday features
    intraday_features = calculate_intraday_features(intraday_master)

    # Join daily and intraday features
    features = roll_features.join(intraday_features, on=["ticker"], how="left")

    # Calculate enhanced features
    features = calculate_sector_features(features)
    features = calculate_market_features(features)
    features = calculate_commodity_features(features)
    features = calculate_cross_sectional_features(features)

    # Filter to training universe (individual stocks + QQQ/SPY for options)
    features = filter_training_universe(features)

    return features


def main():
    parser = argparse.ArgumentParser(description="Calculate features using Polars")
    parser.add_argument(
        "--daily-master-path", required=True, help="Path to daily master parquet"
    )
    parser.add_argument(
        "--intraday-master-path", required=True, help="Path to intraday master parquet"
    )
    parser.add_argument("--output-path", required=True, help="Output path for features")
    parser.add_argument(
        "--date",
        required=True,
        help="Date or date range (YYYY-MM-DD or YYYY-MM-DD,YYYY-MM-DD)",
    )
    parser.add_argument(
        "--window-days", type=int, default=30, help="Rolling window size"
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Enable GPU acceleration"
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")

    args = parser.parse_args()

    # Configure Polars for GPU if requested
    # Configure GPU/CPU optimization for M2 Ultra
    if args.use_gpu:
        try:
            optimize_for_apple_silicon()
            pl.Config.set_streaming_chunk_size(50000)  # Larger chunks for 64GB RAM
            print("🚀 M2 Ultra GPU acceleration enabled")
        except Exception as e:
            print(f"⚠️  GPU acceleration setup failed, using optimized CPU: {e}")
            optimize_for_apple_silicon()  # Still apply CPU optimizations
    else:
        optimize_for_apple_silicon()  # Apply CPU optimizations
        print("🔧 M2 Ultra CPU optimization enabled (24 cores)")

    print(
        f"📊 Loading data from {args.daily_master_path} and {args.intraday_master_path}"
    )

    # Initialize lineage tracking
    lineage = LineageTracker()
    lineage.start_run(
        "calculate_features",
        inputs=[args.daily_master_path, args.intraday_master_path],
        outputs=[args.output_path],
    )

    try:
        # Load data with optimized settings
        print("📥 Loading master datasets...")
        dm = pl.read_parquet(args.daily_master_path)
        im = pl.read_parquet(args.intraday_master_path)

        print(f"Loaded {len(dm)} daily records and {len(im)} intraday records")

        # Parse date range
        start_date, end_date = parse_date_range(args.date)
        print(f"Processing date range: {start_date} to {end_date}")

        # Filter data by date range
        dm_filtered = dm.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )
        im_filtered = im.filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )

        # Calculate rolling features
        print("Calculating all features...")
        window = args.window_days

        # Use the unified feature calculation function with GPU support
        features = calculate_all_features(
            dm_filtered, im_filtered, window, args.use_gpu
        )

        # Ensure output directory exists
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

        print(f"Writing {len(features)} feature records to {args.output_path}")
        # Write output with date suffix for tracking
        date_suffix = start_date.strftime("%Y%m%d")
        parquet_path = args.output_path
        if not parquet_path.endswith(".parquet"):
            parquet_path = f"{parquet_path}/features_{date_suffix}.parquet"

        features.write_parquet(parquet_path)

        # Also write to standard data directory for downstream processing
        data_dir = Path("data/Parquet_data")
        data_dir.mkdir(parents=True, exist_ok=True)
        features.write_parquet(data_dir / f"features_{date_suffix}.parquet")

        print("Feature calculation completed successfully!")
        print(f"Output columns: {features.columns}")
        print(f"Features written to: {parquet_path}")
        print(f"Features also saved to: {data_dir / f'features_{date_suffix}.parquet'}")

        # Complete lineage tracking
        lineage.complete_run(success=True)

    except Exception as e:
        lineage.complete_run(success=False)
        raise e


def get_master_dataframes(date: str):
    """Load master dataframes for a given date"""
    import polars as pl

    daily_master = pl.read_parquet("staged/daily_master.parquet")
    intraday_master = pl.read_parquet("datasets/intraday_master.parquet")

    return daily_master, intraday_master


if __name__ == "__main__":
    import sys

    # Check if running standalone feature calculation
    if (
        len(sys.argv) > 1
        and "--date" in sys.argv
        and "--daily-master-path" not in sys.argv
    ):
        parser = argparse.ArgumentParser(description="Calculate features standalone")
        parser.add_argument(
            "--date", required=True, help="Processing date (YYYY-MM-DD)"
        )
        parser.add_argument(
            "--output-path",
            default="data/Parquet_data/features_{}.parquet",
            help="Output path template",
        )
        parser.add_argument(
            "--use-gpu", action="store_true", help="Enable GPU acceleration"
        )
        parser.add_argument(
            "--window-days", type=int, default=30, help="Rolling window size"
        )

        args = parser.parse_args()

        # Configure Polars for GPU if requested
        if args.use_gpu:
            try:
                import polars as pl

                pl.Config.set_streaming_chunk_size(10000)
                print("✅ GPU acceleration enabled")
            except:
                print("⚠️ GPU acceleration not available, using CPU")

        try:
            print(f"📊 Loading master dataframes for {args.date}...")
            dm, im = get_master_dataframes(args.date)
            print(f"✅ Loaded {len(dm)} daily records and {len(im)} intraday records")

            print(f"🔄 Calculating features with window={args.window_days} days...")
            feats = calculate_all_features(
                dm, im, window=args.window_days, use_gpu=args.use_gpu
            )

            out = args.output_path.format(args.date)

            # Ensure output directory exists
            Path(out).parent.mkdir(parents=True, exist_ok=True)

            feats.write_parquet(out)
            print(f"✅ Features written to {out}")
            print(
                f"📈 Generated {len(feats)} feature records with {len(feats.columns)} columns"
            )
        except FileNotFoundError as e:
            print(f"❌ Master dataset files not found: {e}")
            print("💡 Hint: Run the full pipeline first to generate master datasets:")
            print("   python src/run_full_pipeline.py --date <date>")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Feature calculation failed: {e}")
            sys.exit(1)
    else:
        main()
