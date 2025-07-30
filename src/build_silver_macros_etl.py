#!/usr/bin/env python3
"""
Silver Macros ETL Pipeline

Goal: Produce `staged/silver/macro_series.parquet` combining VIX, DXY, FRED and News signals
on a daily date index covering 2021-07-07 → 2024-07-07.

Author: Copilot Assistant
"""

import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import polars as pl

# GPU acceleration imports
try:
    import cudf
    import dask_cudf
    from dask.distributed import Client, LocalCUDACluster
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Sentiment analysis imports
try:
    import nltk
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True

    # Download required NLTK data (run once)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Warning: VADER sentiment analysis not available. Install with: pip install nltk vaderSentiment")

# Import config from bronze_etl
from bronze_etl.config import DATE_END, DATE_START, RAW_PATHS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SilverMacrosETL:
    """Silver Macros ETL Pipeline for VIX, DXY, FRED, and News data with M2 Ultra optimization"""

    def __init__(self, out_path: str = "staged/silver/macro_series.parquet", test_mode: bool = False):
        self.out_path = out_path
        self.test_mode = test_mode
        self.date_start = pd.to_datetime(DATE_START)
        self.date_end = pd.to_datetime(DATE_END)

        # M2 Ultra CPU optimization
        self.cpu_count = mp.cpu_count()
        self.workers = min(24, self.cpu_count)  # Optimal for M2 Ultra

        # Raw data paths from config
        self.raw_vix = RAW_PATHS["vix"]
        self.raw_dxy = RAW_PATHS["dxy"]
        self.raw_fred = RAW_PATHS["fred"]
        self.raw_news = RAW_PATHS["news"]

        # Initialize GPU cluster if available
        self.gpu_client = None
        if GPU_AVAILABLE and not test_mode:
            try:
                cluster = LocalCUDACluster(n_workers=2, threads_per_worker=2)
                self.gpu_client = Client(cluster)
                logger.info(f"🚀 GPU cluster initialized with {cluster.workers}")
            except Exception as e:
                logger.warning(f"GPU cluster failed, using CPU: {e}")
                self.gpu_client = None

        # Initialize sentiment analyzer
        if SENTIMENT_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("✅ VADER sentiment analyzer initialized")
        else:
            self.sentiment_analyzer = None
            logger.warning("❌ VADER sentiment analyzer not available - using default neutral sentiment")

        logger.info(f"🖥️ M2 Ultra optimization initialized")
        logger.info(f"CPU cores: {self.cpu_count}, Workers: {self.workers}")
        logger.info(f"GPU available: {GPU_AVAILABLE}")
        logger.info(f"Date range: {self.date_start.date()} → {self.date_end.date()}")
        logger.info(f"Output path: {self.out_path}")

    def __del__(self):
        """Clean up GPU resources"""
        if self.gpu_client:
            self.gpu_client.close()

    def compute_sentiment(self, text: str) -> float:
        """Compute sentiment score for given text using VADER"""
        if not self.sentiment_analyzer or not text or not text.strip():
            return 0.0

        try:
            # VADER returns dict with neg, neu, pos, compound scores
            # compound score ranges from -1 (most negative) to +1 (most positive)
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores['compound']
        except Exception as e:
            return 0.0

def process_news_file_batch(file_batch: List[str]) -> List[Dict]:
    """Process a batch of news files in parallel worker"""
    # Initialize sentiment analyzer for this worker
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
    except ImportError:
        analyzer = None

    records = []

    for file_path in file_batch:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Handle both list and dict formats
            articles = data if isinstance(data, list) else [data]

            # Process each article
            for article in articles:
                # Get timestamp from available fields
                timestamp_field = (
                    article.get("timestamp") or
                    article.get("published_utc") or
                    article.get("date") or
                    article.get("published_at")
                )

                if timestamp_field:
                    try:
                        ts = pd.to_datetime(timestamp_field)
                        # Convert to timezone-naive and get date only
                        if ts.tz is not None:
                            ts = ts.tz_convert('UTC').tz_localize(None)
                        date = ts.normalize()

                        # Build text for sentiment analysis
                        text_parts = []
                        for text_field in ["title", "headline", "description", "summary"]:
                            if text_field in article and article[text_field]:
                                text_parts.append(str(article[text_field]))

                        full_text = " ".join(text_parts).strip()

                        # Compute sentiment
                        if analyzer and full_text:
                            scores = analyzer.polarity_scores(full_text)
                            sentiment = scores['compound']
                        else:
                            sentiment = 0.0

                        records.append({
                            "date": date,
                            "sentiment": sentiment
                        })

                    except (ValueError, TypeError):
                        continue

        except Exception:
            continue

    return records

    def read_vix_data(self, logger):
        """Read VIX volatility index data"""
        logger.info(f"Reading VIX data from {self.raw_vix}")

        try:
            # Read with pandas first to handle the parquet format
            vix = pd.read_parquet(self.raw_vix)

            # Select and rename columns
            vix = vix[["date", "value"]].copy()
            vix = vix.rename(columns={"value": "vix_close"})

            # Ensure date column is datetime and timezone-naive
            vix["date"] = pd.to_datetime(vix["date"])
            if vix["date"].dt.tz is not None:
                vix["date"] = vix["date"].dt.tz_localize(None)

            # Remove any null values in vix_close
            vix = vix.dropna(subset=["vix_close"])

            logger.info(f"Loaded {len(vix)} VIX records from {vix['date'].min()} to {vix['date'].max()}")
            return vix

        except Exception as e:
            logger.error(f"Error reading VIX data: {e}")
            raise

    def read_dxy_data(self) -> pd.DataFrame:
        """Read US Dollar Index (DXY) data"""
        logger.info(f"Reading DXY data from {self.raw_dxy}")

        try:
            dxy = pd.read_csv(self.raw_dxy, parse_dates=["date"])
            dxy = dxy.rename(columns={"close": "dxy_close"})
            dxy = dxy[["date", "dxy_close"]].copy()

            # Ensure timezone-naive
            if dxy["date"].dt.tz is not None:
                dxy["date"] = dxy["date"].dt.tz_localize(None)

            # Remove any null values
            dxy = dxy.dropna(subset=["dxy_close"])

            logger.info(f"Loaded {len(dxy)} DXY records from {dxy['date'].min()} to {dxy['date'].max()}")
            return dxy

        except Exception as e:
            logger.error(f"Error reading DXY data: {e}")
            raise

    def read_fred_data(self) -> pd.DataFrame:
        """Read FRED economic indicators and interpolate to daily business frequency"""
        logger.info(f"Reading FRED data from {self.raw_fred}")

        try:
            fred = pd.read_csv(self.raw_fred, parse_dates=["date"])

            # Ensure timezone-naive
            if fred["date"].dt.tz is not None:
                fred["date"] = fred["date"].dt.tz_localize(None)

            # Keep key economic indicators
            key_columns = ["date", "FEDFUNDS", "UNRATE", "CPIAUCSL"]
            available_columns = [col for col in key_columns if col in fred.columns]

            if len(available_columns) < 2:  # At least date + 1 indicator
                logger.warning(f"Limited FRED columns available: {available_columns}")

            fred = fred[available_columns].copy()

            logger.info(f"Loaded {len(fred)} raw FRED records with columns: {available_columns}")

            # Convert monthly FRED data to daily business frequency
            logger.info("Interpolating FRED data to daily business frequency...")
            fred = fred.set_index("date").resample("B").ffill().reset_index()

            # Filter to date range
            fred = fred[
                (fred["date"] >= self.date_start) &
                (fred["date"] <= self.date_end)
            ].copy()

            logger.info(f"Interpolated to {len(fred)} daily FRED records from {fred['date'].min().date()} to {fred['date'].max().date()}")
            return fred

        except Exception as e:
            logger.error(f"Error reading FRED data: {e}")
            raise

    def read_news_data(self) -> pd.DataFrame:
        """Read and aggregate news data from JSON files with parallel sentiment analysis (M2 Ultra optimized)"""
        logger.info(f"📰 Reading news data from {self.raw_news} with {self.workers} parallel workers")

        try:
            # Use os.walk to find all JSON files recursively
            all_files = []
            for root, dirs, files in os.walk(self.raw_news):
                for file in sorted(files):
                    if file.endswith(".json"):
                        all_files.append(os.path.join(root, file))

            if not all_files:
                logger.warning(f"No JSON files found in {self.raw_news}")
                return pd.DataFrame(columns=["date", "news_count", "news_sentiment_avg"])

            logger.info(f"Found {len(all_files)} news JSON files")

            # Limit files for test mode
            if self.test_mode:
                all_files = all_files[:200]  # More files for better parallel testing
                logger.info(f"Test mode: limiting to {len(all_files)} files")

            # Divide files into batches for parallel processing
            batch_size = max(1, len(all_files) // self.workers)
            file_batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]

            logger.info(f"🚀 Processing {len(file_batches)} batches with {self.workers} workers (batch size: {batch_size})")

            # Process batches in parallel using all CPU cores
            all_records = []
            start_time = time.time()

            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                # Submit all batches for parallel processing
                future_to_batch = {executor.submit(process_news_file_batch, batch): i
                                 for i, batch in enumerate(file_batches)}

                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_records = future.result()
                        all_records.extend(batch_records)

                        if batch_idx % 5 == 0:  # Log every 5th batch
                            logger.info(f"Completed batch {batch_idx + 1}/{len(file_batches)}, "
                                      f"records so far: {len(all_records)}")
                    except Exception as e:
                        logger.warning(f"Batch {batch_idx} failed: {e}")

            processing_time = time.time() - start_time

            if not all_records:
                logger.warning("No valid news records found")
                return pd.DataFrame(columns=["date", "news_count", "news_sentiment_avg"])

            # Create DataFrame and aggregate by date
            logger.info(f"⚡ Parallel processing completed in {processing_time:.2f}s")
            logger.info(f"📊 Processing {len(all_records)} records from {len(all_files)} files")

            news_df = pd.DataFrame(all_records)

            # Daily aggregation: count and average sentiment
            news_daily = (
                news_df.groupby("date")
                       .agg(news_count=pd.NamedAgg("sentiment", "count"),
                            news_sentiment_avg=pd.NamedAgg("sentiment", "mean"))
                       .reset_index()
            )

            # Calculate processing metrics
            total_articles = len(all_records)
            sentiment_computed = (news_df["sentiment"] != 0.0).sum()
            articles_per_second = total_articles / processing_time if processing_time > 0 else 0

            logger.info(f"✅ Aggregated {total_articles} news articles into {len(news_daily)} daily records")
            logger.info(f"🎯 Computed sentiment for {sentiment_computed}/{total_articles} articles ({sentiment_computed/total_articles*100:.1f}%)")
            logger.info(f"⚡ Performance: {articles_per_second:,.0f} articles/second with {self.workers} workers")

            # Validate sentiment distribution
            sentiment_stats = news_daily["news_sentiment_avg"].describe()
            logger.info(f"📊 Sentiment stats - Mean: {sentiment_stats['mean']:.4f}, Std: {sentiment_stats['std']:.4f}")
            logger.info(f"📊 Sentiment range - Min: {sentiment_stats['min']:.4f}, Max: {sentiment_stats['max']:.4f}")

            non_zero_sentiment = (news_daily["news_sentiment_avg"] != 0.0).sum()
            logger.info(f"🔍 Non-zero sentiment days: {non_zero_sentiment}/{len(news_daily)} ({non_zero_sentiment/len(news_daily)*100:.1f}%)")

            return news_daily

        except Exception as e:
            logger.error(f"Error reading news data: {e}")
            return pd.DataFrame(columns=["date", "news_count", "news_sentiment_avg"])

    def merge_macro_data(self, vix: pd.DataFrame, dxy: pd.DataFrame,
                        fred: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
        """Merge all macro data sources using business day index"""
        logger.info("Merging macro data sources with business day index")

        # Create full business day index for the training period
        business_days = pd.date_range(self.date_start, self.date_end, freq="B")
        logger.info(f"Created business day index: {len(business_days)} days from {business_days[0].date()} to {business_days[-1].date()}")

        # Start with business day index as base
        macro = pd.DataFrame({"date": business_days})

        # Merge VIX with left join
        macro = macro.merge(vix, on="date", how="left")

        # Merge DXY with left join
        macro = macro.merge(dxy, on="date", how="left")

        # Merge FRED (already daily from interpolation) with left join
        macro = macro.merge(fred, on="date", how="left")

        # Merge News with left join
        macro = macro.merge(news, on="date", how="left")

        logger.info(f"Merged data: {len(macro)} records covering complete business day range")
        logger.info(f"Date range: {macro['date'].min().date()} → {macro['date'].max().date()}")

        return macro

    def compute_derived_features(self, macro: pd.DataFrame) -> pd.DataFrame:
        """Compute derived features like returns with improved filling strategy"""
        logger.info("Computing derived features")

        # Columns to compute returns for
        return_columns = ["vix_close", "dxy_close"]

        # Add FRED columns that exist
        fred_columns = ["FEDFUNDS", "UNRATE", "CPIAUCSL"]
        for col in fred_columns:
            if col in macro.columns:
                return_columns.append(col)

        logger.info(f"Computing returns for: {return_columns}")

        # Compute daily returns
        for col in return_columns:
            if col in macro.columns:
                macro[f"{col}_ret"] = macro[col].pct_change(fill_method=None)

        # Fill any remaining NaNs using forward fill then backward fill
        logger.info("Filling missing values with forward-fill then backward-fill strategy")
        macro = macro.ffill().bfill()

        # Handle news columns specifically
        if "news_count" in macro.columns:
            macro["news_count"] = macro["news_count"].fillna(0)
        if "news_sentiment_avg" in macro.columns:
            macro["news_sentiment_avg"] = macro["news_sentiment_avg"].fillna(0.0)

        logger.info("Derived features computed and missing values filled")
        return macro

    def validate_schema(self, macro: pd.DataFrame) -> Dict[str, Any]:
        """Validate the final schema and data quality with enhanced assertions"""
        logger.info("Validating schema and data quality")

        # Create expected business day index
        expected_dates = pd.date_range(self.date_start, self.date_end, freq="B")

        validation_results = {
            "total_records": len(macro),
            "expected_records": len(expected_dates),
            "date_range": {
                "min": str(macro["date"].min().date()),
                "max": str(macro["date"].max().date())
            },
            "columns": list(macro.columns),
            "missing_dates": [],
            "duplicate_dates": False,
            "null_counts": {},
            "data_types": {},
            "assertions_passed": []
        }

        # ASSERTION 1: No missing dates (complete business day coverage)
        actual_dates = set(macro["date"].dt.date)
        expected_dates_set = set(expected_dates.date)
        missing_dates = expected_dates_set - actual_dates

        try:
            assert len(macro) == len(expected_dates), f"Expected {len(expected_dates)} records, got {len(macro)}"
            assert not missing_dates, f"Missing business dates: {sorted(list(missing_dates))[:5]}"
            validation_results["assertions_passed"].append("Complete business day coverage")
            logger.info("✅ ASSERTION PASSED: Complete business day coverage")
        except AssertionError as e:
            logger.error(f"❌ ASSERTION FAILED: {e}")
            validation_results["missing_dates"] = sorted(list(missing_dates))[:10]

        # ASSERTION 2: No duplicate dates
        try:
            assert macro["date"].is_unique, "Duplicate dates found"
            validation_results["assertions_passed"].append("Unique dates")
            logger.info("✅ ASSERTION PASSED: Unique dates")
        except AssertionError as e:
            logger.error(f"❌ ASSERTION FAILED: {e}")
            validation_results["duplicate_dates"] = True
            duplicates = macro[macro["date"].duplicated()]["date"].tolist()
            logger.error(f"Duplicate dates: {duplicates}")

        # ASSERTION 3: News data availability (either sentiment variation or count > 0)
        if "news_sentiment_avg" in macro.columns:
            sentiment_unique = macro["news_sentiment_avg"].nunique()
            news_articles_total = macro["news_count"].sum() if "news_count" in macro.columns else 0

            if sentiment_unique > 1:
                validation_results["assertions_passed"].append("News sentiment variation")
                logger.info(f"✅ ASSERTION PASSED: News sentiment has {sentiment_unique} unique values")
            elif news_articles_total > 0:
                validation_results["assertions_passed"].append("News data availability (counts only)")
                logger.info(f"✅ ASSERTION PASSED: News data available ({news_articles_total:,} total articles, sentiment not pre-computed)")
            else:
                logger.error(f"❌ ASSERTION FAILED: No news data available")

        # ASSERTION 4: News counts are non-negative
        if "news_count" in macro.columns:
            try:
                assert (macro["news_count"] >= 0).all(), "Found negative news counts"
                validation_results["assertions_passed"].append("Non-negative news counts")
                logger.info("✅ ASSERTION PASSED: Non-negative news counts")
            except AssertionError as e:
                logger.error(f"❌ ASSERTION FAILED: {e}")

        # Check null counts and data types
        for col in macro.columns:
            null_count = macro[col].isnull().sum()
            validation_results["null_counts"][col] = int(null_count)
            validation_results["data_types"][col] = str(macro[col].dtype)

        # Summary statistics for key columns
        if "news_sentiment_avg" in macro.columns:
            sentiment_stats = macro["news_sentiment_avg"].describe()
            validation_results["news_sentiment_stats"] = {
                "mean": float(sentiment_stats["mean"]),
                "std": float(sentiment_stats["std"]),
                "min": float(sentiment_stats["min"]),
                "max": float(sentiment_stats["max"]),
                "unique_values": int(macro["news_sentiment_avg"].nunique())
            }

        logger.info(f"Validation complete: {len(validation_results['assertions_passed'])}/4 assertions passed")
        return validation_results

    def save_output(self, macro: pd.DataFrame) -> Dict[str, Any]:
        """Save the final macro series to parquet with enhanced specifications"""
        logger.info(f"Saving output to {self.out_path}")

        # Ensure output directory exists
        output_dir = Path(self.out_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Save to parquet with enhanced settings
        macro.to_parquet(
            self.out_path,
            index=False,
            engine="pyarrow",
            compression="snappy"
        )

        save_time = time.time() - start_time
        file_size = os.path.getsize(self.out_path) / (1024 * 1024)  # MB

        logger.info(f"Saved {len(macro)} records to {self.out_path}")
        logger.info(f"File size: {file_size:.2f} MB, Save time: {save_time:.2f}s")
        logger.info(f"Used pyarrow engine with snappy compression")

        # Generate summary stats as requested
        summary_stats = {
            "rows": len(macro),
            "columns": macro.columns.tolist(),
            "null_counts": macro.isnull().sum().to_dict()
        }

        logger.info("Dataset summary:")
        logger.info(f"  Rows: {summary_stats['rows']}")
        logger.info(f"  Columns: {len(summary_stats['columns'])}")
        logger.info(f"  Null counts: {summary_stats['null_counts']}")

        return {
            "output_path": self.out_path,
            "records_saved": len(macro),
            "file_size_mb": file_size,
            "save_time_seconds": save_time,
            "summary_stats": summary_stats
        }

    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete Silver Macros ETL pipeline"""
        start_time = time.time()
        logger.info("🚀 Starting Silver Macros ETL Pipeline")

        results = {
            "pipeline": "silver_macros_etl",
            "start_time": datetime.now().isoformat(),
            "test_mode": self.test_mode,
            "success": False
        }

        try:
            # Step 1: Read macro data sources
            logger.info("📊 Step 1: Reading macro data sources")
            vix = self.read_vix_data()
            dxy = self.read_dxy_data()
            fred = self.read_fred_data()
            news = self.read_news_data()

            results["data_sources"] = {
                "vix_records": len(vix),
                "dxy_records": len(dxy),
                "fred_records": len(fred),
                "news_records": len(news)
            }

            # Step 2: Merge data sources
            logger.info("🔄 Step 2: Merging macro data sources")
            macro = self.merge_macro_data(vix, dxy, fred, news)

            # Step 3: Compute derived features
            logger.info("⚙️ Step 3: Computing derived features")
            macro = self.compute_derived_features(macro)

            # Step 4: Schema validation
            logger.info("✅ Step 4: Schema validation")
            validation = self.validate_schema(macro)
            results["validation"] = validation

            # Step 5: Save output
            logger.info("💾 Step 5: Saving output")
            save_results = self.save_output(macro)
            results["save"] = save_results

            # Final summary
            total_time = time.time() - start_time
            results["total_time_seconds"] = total_time
            results["end_time"] = datetime.now().isoformat()
            results["success"] = True

            # Print summary
            print("\n" + "="*60)
            print("🎉 SILVER MACROS ETL PIPELINE COMPLETED")
            print("="*60)
            print(f"📈 Records: {validation['total_records']}")
            print(f"📅 Date Range: {validation['date_range']['min']} → {validation['date_range']['max']}")
            print(f"📊 Columns: {len(validation['columns'])}")
            print(f"📁 Output: {self.out_path}")
            print(f"⏱️  Total Time: {total_time:.2f}s")
            print("="*60)

            logger.info(f"✅ Silver Macros ETL completed successfully in {total_time:.2f}s")

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            raise

        return results


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Silver Macros ETL Pipeline")
    parser.add_argument(
        "--out-path",
        default="staged/silver/macro_series.parquet",
        help="Output path for macro series parquet file"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with limited data"
    )

    args = parser.parse_args()

    # Create and run pipeline
    etl = SilverMacrosETL(out_path=args.out_path, test_mode=args.test_mode)
    results = etl.run_pipeline()

    return results


if __name__ == "__main__":
    main()
