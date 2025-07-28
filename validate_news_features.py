#!/usr/bin/env python3
"""
News Features Validation Script

Validates that news features are properly generated and can be integrated
with the existing feature pipeline.
"""

import logging
from pathlib import Path

import polars as pl

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_news_features(file_path: str) -> bool:
    """Validate news features file."""
    try:
        df = pl.read_parquet(file_path)
        logger.info(f"Loaded news features: {df.shape}")

        # Check required columns
        required_cols = ["date", "ticker", "news_count_lag1", "avg_sentiment_lag1"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False

        # Check data types
        expected_types = {
            "date": pl.Date,
            "ticker": pl.Utf8,
            "news_count_lag1": pl.Int64,
            "avg_sentiment_lag1": pl.Float64,
        }

        for col, expected_type in expected_types.items():
            actual_type = df[col].dtype
            if actual_type != expected_type:
                logger.warning(
                    f"Column {col} has type {actual_type}, expected {expected_type}"
                )

        # Check sentiment range
        sentiment_stats = df.filter(pl.col("avg_sentiment_lag1").is_not_null())[
            "avg_sentiment_lag1"
        ]
        if sentiment_stats.len() > 0:
            min_sent = sentiment_stats.min()
            max_sent = sentiment_stats.max()
            logger.info(f"Sentiment range: {min_sent} to {max_sent}")

            if min_sent < -1.0 or max_sent > 1.0:
                logger.warning(f"Sentiment values outside expected range [-1.0, 1.0]")

        # Check for duplicates
        duplicates = df.group_by(["date", "ticker"]).len().filter(pl.col("len") > 1)
        if duplicates.height > 0:
            logger.error(
                f"Found {duplicates.height} duplicate date-ticker combinations"
            )
            return False

        # Summary stats
        logger.info(f"Validation Summary:")
        logger.info(f"  - Total records: {df.height}")
        logger.info(f"  - Unique tickers: {df['ticker'].n_unique()}")
        logger.info(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(
            f"  - Non-null sentiment records: {df.filter(pl.col('avg_sentiment_lag1').is_not_null()).height}"
        )
        logger.info(
            f"  - Non-null count records: {df.filter(pl.col('news_count_lag1').is_not_null()).height}"
        )

        logger.info("✅ News features validation PASSED")
        return True

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def main():
    # Test with the sample file we created
    test_file = "data/Parquet_data/news_features_2025-05-01_to_2025-05-03.parquet"

    if not Path(test_file).exists():
        logger.error(f"Test file not found: {test_file}")
        logger.info(
            "Run: python src/build_news_features.py --start-date 2025-05-01 --end-date 2025-05-03"
        )
        return

    logger.info(f"Validating news features: {test_file}")
    success = validate_news_features(test_file)

    if success:
        logger.info("🎉 News features are ready for integration!")
    else:
        logger.error("❌ News features validation failed")


if __name__ == "__main__":
    main()
