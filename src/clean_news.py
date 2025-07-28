#!/usr/bin/env python3
"""
News Data Processing Script for Conviction AI Pipeline

Processes news data from JSON files and generates sentiment features:
- news_count_lag1: Count of news articles per ticker (lagged)
- avg_sentiment_lag1: Average sentiment score per ticker (lagged)
"""

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_news_data(news_path: Path) -> List[Dict]:
    """Load news data from JSON file."""
    try:
        with open(news_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load news data from {news_path}: {e}")
        return []


def extract_sentiment_score(sentiment: str) -> float:
    """Convert sentiment string to numeric score."""
    sentiment_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    return sentiment_map.get(sentiment.lower(), 0.0)


def process_news_articles(articles: List[Dict], date: str) -> pl.DataFrame:
    """Process news articles and extract sentiment features."""
    processed_data = []

    for article in articles:
        # Extract basic info
        published_date = article.get("published_utc", "").split("T")[0]
        tickers = article.get("tickers", [])
        insights = article.get("insights", [])

        # Process each ticker mentioned in the article
        for insight in insights:
            ticker = insight.get("ticker", "").upper()
            sentiment = insight.get("sentiment", "neutral")

            if ticker and ticker in tickers:
                processed_data.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "published_date": published_date,
                        "sentiment_score": extract_sentiment_score(sentiment),
                        "article_count": 1,
                    }
                )

    if not processed_data:
        # Return empty DataFrame with correct schema
        return pl.DataFrame(
            {"date": [], "ticker": [], "news_count": [], "avg_sentiment": []},
            schema={
                "date": pl.Date,
                "ticker": pl.Utf8,
                "news_count": pl.Int64,
                "avg_sentiment": pl.Float64,
            },
        )

    # Convert to DataFrame and aggregate
    df = pl.DataFrame(processed_data)

    # Aggregate by ticker
    aggregated = df.group_by(["date", "ticker"]).agg(
        [
            pl.col("article_count").sum().alias("news_count"),
            pl.col("sentiment_score").mean().alias("avg_sentiment"),
        ]
    )

    # Convert date column to proper date type
    aggregated = aggregated.with_columns([pl.col("date").str.to_date().alias("date")])

    return aggregated


def apply_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """Apply lag to news features to prevent forward-looking bias."""
    if df.height == 0:
        return df.with_columns(
            [
                pl.lit(None, dtype=pl.Int64).alias("news_count_lag1"),
                pl.lit(None, dtype=pl.Float64).alias("avg_sentiment_lag1"),
            ]
        )

    # Sort by ticker and date
    df = df.sort(["ticker", "date"])

    # Apply lag by ticker
    df = df.with_columns(
        [
            pl.col("news_count").shift(1).over("ticker").alias("news_count_lag1"),
            pl.col("avg_sentiment").shift(1).over("ticker").alias("avg_sentiment_lag1"),
        ]
    )

    return df.select(["date", "ticker", "news_count_lag1", "avg_sentiment_lag1"])


def main():
    parser = argparse.ArgumentParser(
        description="Process news data for sentiment features"
    )
    parser.add_argument("--date", required=True, help="Date to process (YYYY-MM-DD)")
    parser.add_argument(
        "--news-dir",
        default="data/Parquet_data/Raw/news",
        help="Directory containing news data",
    )
    parser.add_argument("--output-path", help="Output path for processed data")

    args = parser.parse_args()

    # Parse date
    try:
        date_obj = datetime.strptime(args.date, "%Y-%m-%d")
        date_str = args.date
    except ValueError:
        logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
        return

    # Construct news file path
    year = date_obj.strftime("%Y")
    month = date_obj.strftime("%m")
    day = date_obj.strftime("%d")

    news_path = Path(args.news_dir) / year / month / day / "news_data.json"

    if not news_path.exists():
        logger.warning(f"News data not found for {date_str} at {news_path}")
        # Create empty DataFrame with correct schema
        df = pl.DataFrame(
            {
                "date": [date_str],
                "ticker": ["DUMMY"],
                "news_count_lag1": [None],
                "avg_sentiment_lag1": [None],
            },
            schema={
                "date": pl.Date,
                "ticker": pl.Utf8,
                "news_count_lag1": pl.Int64,
                "avg_sentiment_lag1": pl.Float64,
            },
        ).filter(
            pl.col("ticker") != "DUMMY"
        )  # Remove dummy row
    else:
        logger.info(f"Processing news data for {date_str}")

        # Load and process news data
        articles = load_news_data(news_path)
        logger.info(f"Loaded {len(articles)} articles")

        # Process articles
        df = process_news_articles(articles, date_str)
        logger.info(f"Processed {df.height} ticker-date combinations")

        # Apply lag features
        df = apply_lag_features(df)
        logger.info(f"Applied lag features, final shape: {df.shape}")

    # Set output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(f"data/Parquet_data/news_{date_str}.parquet")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save processed data
    df.write_parquet(output_path)
    logger.info(f"Saved news features to {output_path}")

    # Print summary
    if df.height > 0:
        logger.info(f"Summary:")
        logger.info(f"  - Unique tickers: {df['ticker'].n_unique()}")
        logger.info(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(
            f"  - Non-null sentiment records: {df.filter(pl.col('avg_sentiment_lag1').is_not_null()).height}"
        )


if __name__ == "__main__":
    main()
