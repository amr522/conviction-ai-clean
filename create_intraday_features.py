#!/usr/bin/env python3
"""
Intraday Feature Engineering Pipeline with Twitter Sentiment Integration
Extends existing feature pipeline to include multi-timeframe Twitter sentiment aggregations
"""

import asyncio
import json
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import boto3
from botocore.exceptions import ClientError

from utils.sentiment_aggregation import SentimentAggregator

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/intraday_features.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntradayFeatureEngineer:
    """Intraday feature engineering with Twitter sentiment integration"""
    
    def __init__(self,
                 s3_bucket: str = "conviction-ai-data",
                 s3_sentiment_prefix: str = "twitter-sentiment/scored-tweets/",
                 s3_features_prefix: str = "processed-features/",
                 region_name: str = "us-east-1",
                 symbols: Optional[List[str]] = None):
        """
        Initialize intraday feature engineer
        
        Args:
            s3_bucket: S3 bucket for data storage
            s3_sentiment_prefix: S3 prefix for scored tweets
            s3_features_prefix: S3 prefix for feature output
            region_name: AWS region
            symbols: List of symbols to process (default: load from config)
        """
        self.s3_bucket = s3_bucket
        self.s3_sentiment_prefix = s3_sentiment_prefix
        self.s3_features_prefix = s3_features_prefix
        self.region_name = region_name
        
        self.s3_client = boto3.client('s3', region_name=region_name)
        
        self.symbols = symbols or self._load_symbols_from_config()
        
        self.sentiment_aggregator = SentimentAggregator()
        
        self.base_features = self._load_base_feature_schema()
        self.sentiment_features = ['sent_5m', 'sent_10m', 'sent_60m', 'sent_daily']
        
        logger.info(f"Initialized intraday feature engineer for {len(self.symbols)} symbols")
        logger.info(f"Base features: {len(self.base_features)}")
        logger.info(f"New sentiment features: {self.sentiment_features}")
    
    def _load_symbols_from_config(self) -> List[str]:
        """Load symbols from configuration file"""
        try:
            config_path = 'config/models_to_train_46.txt'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    symbols = [line.strip() for line in f.readlines() if line.strip()]
                logger.info(f"Loaded {len(symbols)} symbols from config")
                return symbols
            else:
                default_symbols = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
                    'JPM', 'JNJ', 'V', 'MA', 'UNH', 'HD', 'PG', 'DIS'
                ]
                logger.warning(f"Config file not found, using default symbols: {len(default_symbols)}")
                return default_symbols
                
        except Exception as e:
            logger.error(f"Error loading symbols from config: {e}")
            return ['AAPL', 'MSFT', 'GOOGL']  # Minimal fallback
    
    def _load_base_feature_schema(self) -> List[str]:
        """Load base feature schema from metadata"""
        try:
            metadata_path = 'data/sagemaker/feature_metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    features = metadata.get('feature_columns', [])
                    logger.info(f"Loaded {len(features)} base features from metadata")
                    return features
            else:
                logger.warning("Feature metadata not found, using default schema")
                return self._get_default_feature_schema()
                
        except Exception as e:
            logger.error(f"Error loading feature metadata: {e}")
            return self._get_default_feature_schema()
    
    def _get_default_feature_schema(self) -> List[str]:
        """Get default feature schema based on existing data"""
        return [
            "open", "high", "low", "close", "volume", "returns_1d", "returns_5d", 
            "returns_20d", "volatility_20d", "rsi_14", "macd", "macd_signal", 
            "bb_upper", "bb_lower", "bb_middle", "atr_14", "adx_14", "cci_20", 
            "williams_r", "stoch_k", "stoch_d", "obv", "mfi_14", "trix", "vwap", 
            "ema_12", "ema_26", "sma_50", "sma_200", "price_to_sma50", 
            "price_to_sma200", "volume_sma_20", "volume_ratio", "news_sentiment", 
            "news_volume", "sector_momentum", "market_cap", "pe_ratio", "pb_ratio", 
            "debt_to_equity", "roe", "roa", "current_ratio", "quick_ratio", 
            "gross_margin", "operating_margin", "net_margin", "asset_turnover", 
            "inventory_turnover", "receivables_turnover", "cash_ratio", 
            "interest_coverage", "dividend_yield", "payout_ratio", "earnings_growth", 
            "revenue_growth", "book_value_growth", "fcf_yield", "ev_ebitda", 
            "price_to_sales", "price_to_book", "price_to_fcf", "enterprise_value", 
            "shares_outstanding", "float_shares", "insider_ownership", 
            "institutional_ownership", "short_interest", "days_to_cover"
        ]
    
    async def load_existing_features(self, symbol: str, date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Load existing feature data for a symbol
        
        Args:
            symbol: Stock symbol
            date_range: Optional tuple of (start_date, end_date) in YYYY-MM-DD format
            
        Returns:
            DataFrame with existing features
        """
        try:
            local_path = f'data/processed_with_news_20250628/{symbol}_features.csv'
            if os.path.exists(local_path):
                logger.info(f"Loading existing features for {symbol} from local file")
                df = pd.read_csv(local_path)
                
                if date_range:
                    start_date, end_date = date_range
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                return df
            
            s3_key = f"{self.s3_features_prefix}{symbol}_features.csv"
            logger.info(f"Loading existing features for {symbol} from S3: {s3_key}")
            
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            df = pd.read_csv(response['Body'])
            
            if date_range:
                start_date, end_date = date_range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            return df
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"No existing features found for {symbol}")
                return pd.DataFrame()
            else:
                logger.error(f"S3 error loading features for {symbol}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error loading existing features for {symbol}: {e}")
            raise
    
    async def load_sentiment_data(self, symbol: str, date_range: Tuple[str, str]) -> pd.DataFrame:
        """
        Load Twitter sentiment data for a symbol and date range
        
        Args:
            symbol: Stock symbol
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            
        Returns:
            DataFrame with sentiment data
        """
        try:
            start_date, end_date = date_range
            logger.info(f"Loading sentiment data for {symbol} from {start_date} to {end_date}")
            
            prefix = f"{self.s3_sentiment_prefix}{symbol}/"
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix)
            
            sentiment_data = []
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    try:
                        filename = key.split('/')[-1]
                        file_date = filename.split('_')[0]  # Extract YYYY-MM-DD
                        
                        if start_date <= file_date <= end_date:
                            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
                            content = response['Body'].read().decode('utf-8')
                            
                            for line in content.strip().split('\n'):
                                if line.strip():
                                    tweet_data = json.loads(line)
                                    sentiment_data.append(tweet_data)
                    
                    except Exception as e:
                        logger.warning(f"Error processing sentiment file {key}: {e}")
                        continue
            
            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                logger.info(f"Loaded {len(df)} sentiment records for {symbol}")
                return df
            else:
                logger.warning(f"No sentiment data found for {symbol} in date range")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading sentiment data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def create_sentiment_features(self, symbol: str, sentiment_df: pd.DataFrame, 
                                      feature_dates: List[str]) -> pd.DataFrame:
        """
        Create sentiment features for specific dates
        
        Args:
            symbol: Stock symbol
            sentiment_df: DataFrame with sentiment data
            feature_dates: List of dates to create features for
            
        Returns:
            DataFrame with sentiment features
        """
        try:
            if sentiment_df.empty:
                logger.warning(f"No sentiment data available for {symbol}")
                return pd.DataFrame({
                    'date': feature_dates,
                    'sent_5m': [0.0] * len(feature_dates),
                    'sent_10m': [0.0] * len(feature_dates),
                    'sent_60m': [0.0] * len(feature_dates),
                    'sent_daily': [0.0] * len(feature_dates)
                })
            
            logger.info(f"Creating sentiment features for {symbol} on {len(feature_dates)} dates")
            
            if 'created_at' in sentiment_df.columns:
                sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['created_at'])
            elif 'timestamp' in sentiment_df.columns:
                sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
            else:
                logger.error(f"No timestamp column found in sentiment data for {symbol}")
                return pd.DataFrame()
            
            sentiment_features = []
            
            for date_str in feature_dates:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                date_sentiment = sentiment_df[
                    sentiment_df['timestamp'].dt.date == date_obj
                ].copy()
                
                if date_sentiment.empty:
                    features = {
                        'date': date_str,
                        'sent_5m': 0.0,
                        'sent_10m': 0.0,
                        'sent_60m': 0.0,
                        'sent_daily': 0.0
                    }
                else:
                    features = {
                        'date': date_str,
                        'sent_5m': self.sentiment_aggregator.aggregate_5min(date_sentiment),
                        'sent_10m': self.sentiment_aggregator.aggregate_10min(date_sentiment),
                        'sent_60m': self.sentiment_aggregator.aggregate_60min(date_sentiment),
                        'sent_daily': self.sentiment_aggregator.aggregate_daily(date_sentiment)
                    }
                
                sentiment_features.append(features)
            
            result_df = pd.DataFrame(sentiment_features)
            logger.info(f"Created sentiment features for {len(result_df)} dates")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error creating sentiment features for {symbol}: {e}")
            raise
    
    async def merge_features(self, existing_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge existing features with new sentiment features
        
        Args:
            existing_df: DataFrame with existing features
            sentiment_df: DataFrame with sentiment features
            
        Returns:
            DataFrame with merged features
        """
        try:
            if existing_df.empty:
                logger.warning("No existing features to merge with")
                return sentiment_df
            
            if sentiment_df.empty:
                logger.warning("No sentiment features to merge")
                for col in self.sentiment_features:
                    existing_df[col] = 0.0
                return existing_df
            
            merged_df = existing_df.merge(sentiment_df, on='date', how='left')
            
            for col in self.sentiment_features:
                if col in merged_df.columns:
                    merged_df[col] = merged_df[col].fillna(0.0)
                else:
                    merged_df[col] = 0.0
            
            logger.info(f"Merged features: {len(merged_df)} rows, {len(merged_df.columns)} columns")
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging features: {e}")
            raise
    
    async def save_enhanced_features(self, symbol: str, features_df: pd.DataFrame):
        """
        Save enhanced features with sentiment data
        
        Args:
            symbol: Stock symbol
            features_df: DataFrame with enhanced features
        """
        try:
            if features_df.empty:
                logger.warning(f"No features to save for {symbol}")
                return
            
            local_dir = 'data/processed_with_sentiment'
            os.makedirs(local_dir, exist_ok=True)
            local_path = f'{local_dir}/{symbol}_features_enhanced.csv'
            
            features_df.to_csv(local_path, index=False)
            logger.info(f"Saved enhanced features locally: {local_path}")
            
            s3_key = f"{self.s3_features_prefix}enhanced/{symbol}_features_enhanced.csv"
            
            csv_buffer = features_df.to_csv(index=False)
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=csv_buffer.encode('utf-8'),
                ContentType='text/csv'
            )
            
            logger.info(f"Saved enhanced features to S3: s3://{self.s3_bucket}/{s3_key}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced features for {symbol}: {e}")
            raise
    
    async def process_symbol(self, symbol: str, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Process a single symbol to create enhanced features
        
        Args:
            symbol: Stock symbol to process
            date_range: Optional date range (start_date, end_date)
            
        Returns:
            Processing summary
        """
        try:
            logger.info(f"🔄 Processing symbol: {symbol}")
            
            if not date_range:
                end_date = datetime.now().date().strftime('%Y-%m-%d')
                start_date = (datetime.now().date() - timedelta(days=365)).strftime('%Y-%m-%d')
                date_range = (start_date, end_date)
            
            start_date, end_date = date_range
            logger.info(f"   Date range: {start_date} to {end_date}")
            
            existing_df = await self.load_existing_features(symbol, date_range)
            
            if existing_df.empty:
                logger.warning(f"No existing features found for {symbol}")
                return {
                    'symbol': symbol,
                    'status': 'skipped',
                    'reason': 'no_existing_features',
                    'features_created': 0
                }
            
            feature_dates = existing_df['date'].unique().tolist()
            logger.info(f"   Processing {len(feature_dates)} dates")
            
            sentiment_df = await self.load_sentiment_data(symbol, date_range)
            
            sentiment_features_df = await self.create_sentiment_features(
                symbol, sentiment_df, feature_dates
            )
            
            enhanced_df = await self.merge_features(existing_df, sentiment_features_df)
            
            await self.save_enhanced_features(symbol, enhanced_df)
            
            summary = {
                'symbol': symbol,
                'status': 'completed',
                'features_created': len(enhanced_df),
                'sentiment_records': len(sentiment_df),
                'date_range': date_range,
                'new_columns': self.sentiment_features
            }
            
            logger.info(f"✅ Completed processing {symbol}: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error processing symbol {symbol}: {e}")
            return {
                'symbol': symbol,
                'status': 'failed',
                'error': str(e),
                'features_created': 0
            }
    
    async def process_all_symbols(self, 
                                date_range: Optional[Tuple[str, str]] = None,
                                max_concurrent: int = 5) -> Dict[str, Any]:
        """
        Process all symbols to create enhanced features
        
        Args:
            date_range: Optional date range (start_date, end_date)
            max_concurrent: Maximum concurrent symbol processing
            
        Returns:
            Processing summary
        """
        try:
            logger.info(f"🚀 Starting intraday feature engineering for {len(self.symbols)} symbols")
            
            semaphore = asyncio.Semaphore(max_concurrent)
            results = []
            
            async def process_symbol_with_semaphore(symbol: str):
                async with semaphore:
                    return await self.process_symbol(symbol, date_range)
            
            tasks = [process_symbol_with_semaphore(symbol) for symbol in self.symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            completed = [r for r in results if isinstance(r, dict) and r.get('status') == 'completed']
            failed = [r for r in results if isinstance(r, dict) and r.get('status') == 'failed']
            skipped = [r for r in results if isinstance(r, dict) and r.get('status') == 'skipped']
            
            total_features = sum(r.get('features_created', 0) for r in completed)
            total_sentiment_records = sum(r.get('sentiment_records', 0) for r in completed)
            
            summary = {
                'status': 'completed',
                'symbols_processed': len(completed),
                'symbols_failed': len(failed),
                'symbols_skipped': len(skipped),
                'total_features_created': total_features,
                'total_sentiment_records': total_sentiment_records,
                'date_range': date_range,
                'new_feature_columns': self.sentiment_features,
                'completed_symbols': [r['symbol'] for r in completed],
                'failed_symbols': [r['symbol'] for r in failed],
                'skipped_symbols': [r['symbol'] for r in skipped]
            }
            
            logger.info("✅ Intraday feature engineering completed")
            logger.info(f"   Symbols processed: {len(completed)}")
            logger.info(f"   Symbols failed: {len(failed)}")
            logger.info(f"   Symbols skipped: {len(skipped)}")
            logger.info(f"   Total features created: {total_features}")
            logger.info(f"   Total sentiment records: {total_sentiment_records}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    async def update_feature_metadata(self):
        """Update feature metadata to include new sentiment columns"""
        try:
            metadata_path = 'data/sagemaker/feature_metadata.json'
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    'feature_columns': self.base_features,
                    'target_column': 'target_binary'
                }
            
            current_features = metadata.get('feature_columns', [])
            updated_features = current_features.copy()
            
            for feature in self.sentiment_features:
                if feature not in updated_features:
                    updated_features.append(feature)
            
            metadata['feature_columns'] = updated_features
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Updated feature metadata: {len(updated_features)} total features")
            logger.info(f"Added sentiment features: {self.sentiment_features}")
            
            s3_key = "sagemaker/feature_metadata_enhanced.json"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=json.dumps(metadata, indent=2).encode('utf-8'),
                ContentType='application/json'
            )
            
            logger.info(f"Saved enhanced metadata to S3: s3://{self.s3_bucket}/{s3_key}")
            
        except Exception as e:
            logger.error(f"Error updating feature metadata: {e}")
            raise


async def main():
    """Main function for intraday feature engineering"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Intraday Feature Engineering with Twitter Sentiment')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to process')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--s3-bucket', default='conviction-ai-data',
                        help='S3 bucket for data storage')
    parser.add_argument('--max-concurrent', type=int, default=5,
                        help='Maximum concurrent symbol processing')
    parser.add_argument('--region', default='us-east-1',
                        help='AWS region')
    parser.add_argument('--update-metadata', action='store_true',
                        help='Update feature metadata with new columns')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate setup without processing')
    
    args = parser.parse_args()
    
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🚀 Starting Intraday Feature Engineering with Twitter Sentiment")
    logger.info(f"   S3 Bucket: {args.s3_bucket}")
    logger.info(f"   Max Concurrent: {args.max_concurrent}")
    logger.info(f"   Region: {args.region}")
    
    if args.start_date and args.end_date:
        date_range = (args.start_date, args.end_date)
        logger.info(f"   Date Range: {args.start_date} to {args.end_date}")
    else:
        date_range = None
        logger.info("   Date Range: Default (last 365 days)")
    
    try:
        engineer = IntradayFeatureEngineer(
            s3_bucket=args.s3_bucket,
            region_name=args.region,
            symbols=args.symbols
        )
        
        if args.dry_run:
            logger.info("🧪 DRY RUN MODE - Validating setup")
            logger.info(f"   Symbols to process: {len(engineer.symbols)}")
            logger.info(f"   Base features: {len(engineer.base_features)}")
            logger.info(f"   New sentiment features: {engineer.sentiment_features}")
            return
        
        if args.update_metadata:
            logger.info("📝 Updating feature metadata...")
            await engineer.update_feature_metadata()
        
        summary = await engineer.process_all_symbols(
            date_range=date_range,
            max_concurrent=args.max_concurrent
        )
        
        logger.info("✅ Intraday feature engineering completed successfully")
        logger.info(f"   Summary: {json.dumps(summary, indent=2)}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
