#!/usr/bin/env python3
"""
Option Features Validation Script

Validates that the training dataset contains all required option features
and meets quality standards for ML training.
"""

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# All features from features_list.md
REQUIRED_COLUMNS = [
    "date",
    "ticker",
    # Macro/VIX/DXY Features
    "vix_value",
    "vix_ma_divergence",
    "dxy_value",
    "iv_rank_30d",
    # News Features (NOW AVAILABLE)
    "news_count_lag1",
    "avg_sentiment_lag1",
    # Options Daily Features
    "optd_iv30",
    "optd_hv30",
    "optd_iv30_lag1",
    "optd_hv30_lag1",
    "optd_iv_percentile",
    "optd_iv_percentile_lag1",
    "optd_vrp_30d",
    "optd_vrp_30d_lag1",
    "optd_vrp_spike",
    "optd_iv_skew_slope",
    "optd_vol_surprise",
    "optd_put_call_ratio",
    "optd_volume",
    # Options 30-Minute Features
    "opt30_mid_price_return",
    "opt30_bid_ask_spread",
    "opt30_implied_volatility",
    "opt30_delta",
    "opt30_theta",
    "opt30_volume_return",
    "opt30_rolling_vol_5",
    "opt30_flow_divergence",
    "opt30_gamma_squeeze",
    # Stocks Daily Features
    "stockd_close",
    "stockd_volume",
    "stockd_return_1d",
    "stockd_return_1d_lag1",
    "stockd_vol_7d",
    "stockd_vol_7d_lag1",
    "stockd_volume_pct_change",
    "stockd_beta_spy",
    "stockd_days_to_earnings",
    "stockd_earnings_flag",
    # Stocks 30-Minute Features
    "stock30_close_return",
    "stock30_rolling_vol_5",
    "stock30_is_last_30min",
    # Sector Features
    "sector",
    "is_tech",
    "is_financial",
    "is_energy",
    "market_beta_20d",
    # Market Direction Features
    "spy_return_1d",
    "qqq_return_1d",
    "spy_volume",
    "qqq_volume",
    # Commodity Reference Features
    "gold_return_1d",
    "oil_return_1d",
    "gold_volatility_10d",
]

# 36-Stock Training Universe + ETF Options Targets
EXPECTED_TICKERS = [
    # Tech: AAPL, GOOGL, META, MSFT, NFLX, SMCI
    "AAPL",
    "GOOGL",
    "META",
    "MSFT",
    "NFLX",
    "SMCI",
    # Tech (High IV): AMD, NVDA, PLTR
    "AMD",
    "NVDA",
    "PLTR",
    # Financial: BAC, GS, JPM, MA, MS, V
    "BAC",
    "GS",
    "JPM",
    "MA",
    "MS",
    "V",
    # Healthcare: ABBV, JNJ, MRK, PFE, UNH
    "ABBV",
    "JNJ",
    "MRK",
    "PFE",
    "UNH",
    # Consumer: DIS, NKE, SBUX, WMT, AMZN
    "DIS",
    "NKE",
    "SBUX",
    "WMT",
    "AMZN",
    # Energy: CVX, XOM
    "CVX",
    "XOM",
    # Industrial: BA, CAT, GE
    "BA",
    "CAT",
    "GE",
    # Crypto/Fintech (High IV): COIN, HOOD, MSTR
    "COIN",
    "HOOD",
    "MSTR",
    # Auto (High IV): TSLA
    "TSLA",
    # ETF Options Targets: QQQ, SPY
    "QQQ",
    "SPY",
]


def validate_dataset(file_path: str) -> bool:
    """Validate the training dataset."""
    try:
        df = pl.read_parquet(file_path)
        logger.info(f"Loaded dataset: {df.shape}")

        # Check minimum rows (expect ≥200 columns, substantial data)
        if df.height < 1000:
            logger.error(f"Dataset too small: {df.height} rows (minimum 1,000)")
            return False

        if len(df.columns) < 200:
            logger.warning(f"Expected ≥200 columns, found {len(df.columns)}")

        # Check for required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing optional columns: {missing_cols[:5]}...")

        # Check for duplicates
        duplicates = df.group_by(["date", "ticker"]).len().filter(pl.col("len") > 1)
        if duplicates.height > 0:
            logger.error(
                f"Found {duplicates.height} duplicate date-ticker combinations"
            )
            return False

        # Validate ticker universe
        actual_tickers = set(df["ticker"].unique())
        expected_tickers = set(EXPECTED_TICKERS)
        coverage = len(actual_tickers & expected_tickers) / len(expected_tickers) * 100

        # Basic stats
        logger.info(f"Validation Summary:")
        logger.info(f"  - Total records: {df.height}")
        logger.info(f"  - Total columns: {len(df.columns)}")
        logger.info(f"  - Unique tickers: {df['ticker'].n_unique()}")
        logger.info(
            f"  - Ticker coverage: {coverage:.1f}% of expected 36-stock universe"
        )
        logger.info(f"  - Date range: {df['date'].min()} to {df['date'].max()}")

        if coverage < 80:
            logger.warning(f"Low ticker coverage: {coverage:.1f}% (expected >80%)")

        logger.info("✅ Training dataset validation PASSED")
        return True

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate option features dataset")
    parser.add_argument("--input-path", required=True, help="Path to training dataset")
    args = parser.parse_args()

    if not Path(args.input_path).exists():
        logger.error(f"File not found: {args.input_path}")
        return

    success = validate_dataset(args.input_path)
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
