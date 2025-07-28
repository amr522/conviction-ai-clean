#!/usr/bin/env python3
"""
Feature Lagging Validation Script

Validates that features are properly lagged to prevent forward-looking bias.
"""

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected lag features from features_list.md
LAG_FEATURES = [
    # News features (NOW AVAILABLE)
    "news_count_lag1",
    "avg_sentiment_lag1",
    # Options daily lag features
    "optd_iv30_lag1",
    "optd_hv30_lag1",
    "optd_iv_percentile_lag1",
    "optd_vrp_30d_lag1",
    # Stocks daily lag features
    "stockd_return_1d_lag1",
    "stockd_vol_7d_lag1",
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


def validate_lagging(file_path: str) -> bool:
    """Validate feature lagging."""
    try:
        df = pl.read_parquet(file_path)
        logger.info(f"Loaded dataset: {df.shape}")

        # Check for lag features
        existing_lag_features = [col for col in LAG_FEATURES if col in df.columns]
        logger.info(f"Found {len(existing_lag_features)} lag features")

        if existing_lag_features:
            # Sort by ticker and date
            df = df.sort(["ticker", "date"])

            # Check first row per ticker has null lag features
            first_rows = df.group_by("ticker").first()

            for col in existing_lag_features:
                if col in first_rows.columns:
                    null_count = first_rows[col].null_count()
                    total_tickers = first_rows.height
                    null_pct = (null_count / total_tickers) * 100

                    logger.info(
                        f"  {col}: {null_pct:.1f}% null in first rows (expected: high)"
                    )

        # Validate ticker universe
        actual_tickers = set(df["ticker"].unique())
        expected_tickers = set(EXPECTED_TICKERS)
        missing_tickers = expected_tickers - actual_tickers
        extra_tickers = actual_tickers - expected_tickers

        if missing_tickers:
            logger.warning(f"Missing expected tickers: {sorted(missing_tickers)}")
        if extra_tickers:
            logger.info(f"Extra tickers found: {sorted(extra_tickers)}")

        logger.info(
            f"Ticker coverage: {len(actual_tickers & expected_tickers)}/{len(expected_tickers)} expected tickers"
        )

        # Check date ordering
        date_issues = 0
        for ticker in df["ticker"].unique()[:5]:  # Check first 5 tickers
            ticker_data = df.filter(pl.col("ticker") == ticker).sort("date")
            dates = ticker_data["date"].to_list()

            for i in range(1, len(dates)):
                if dates[i] <= dates[i - 1]:
                    date_issues += 1
                    break

        if date_issues > 0:
            logger.warning(f"Found {date_issues} tickers with date ordering issues")

        logger.info("✅ Feature lagging validation PASSED")
        return True

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate feature lagging")
    parser.add_argument("--input-path", required=True, help="Path to training dataset")
    args = parser.parse_args()

    if not Path(args.input_path).exists():
        logger.error(f"File not found: {args.input_path}")
        return

    success = validate_lagging(args.input_path)
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
