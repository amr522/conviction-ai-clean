#!/usr/bin/env python3
"""
Create Intraday Features with Sentiment Integration
Enhanced feature engineering for 5/10/60 minute windows with Twitter sentiment
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import boto3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sentiment_data(symbol: str, date: str, s3_bucket: str = 'hpo-bucket-773934887314') -> pd.DataFrame:
    """Load sentiment data for a symbol and date"""
    try:
        s3_client = boto3.client('s3')
        s3_key = f"sentiment/finbert/{symbol}/{date}.parquet"
        
        local_file = f"/tmp/{symbol}_{date}_sentiment.parquet"
        s3_client.download_file(s3_bucket, s3_key, local_file)
        
        df = pd.read_parquet(local_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        os.remove(local_file)
        return df
        
    except Exception as e:
        logger.warning(f"Failed to load sentiment data for {symbol} on {date}: {e}")
        return pd.DataFrame()

def create_sentiment_features(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Create sentiment features with 5/10/60 minute rolling windows"""
    if sentiment_df.empty:
        logger.warning("No sentiment data available, creating zero-filled sentiment features")
        price_df['sent_mean_5m'] = 0.0
        price_df['sent_sum_10m'] = 0.0
        price_df['sent_pos_ratio_60m'] = 0.0
        price_df['sent_volume_5m'] = 0
        price_df['sent_volume_10m'] = 0
        price_df['sent_volume_60m'] = 0
        return price_df
    
    try:
        sentiment_df = sentiment_df.set_index('timestamp').sort_index()
        price_df = price_df.set_index('timestamp').sort_index()
        
        sentiment_5m = sentiment_df['sent_compound'].rolling('5min').agg(['mean', 'count'])
        sentiment_10m = sentiment_df['sent_compound'].rolling('10min').agg(['sum', 'count'])
        sentiment_60m = sentiment_df['sent_positive'].rolling('60min').agg(['mean', 'count'])
        
        price_df = price_df.join(sentiment_5m['mean'].rename('sent_mean_5m'), how='left')
        price_df = price_df.join(sentiment_5m['count'].rename('sent_volume_5m'), how='left')
        price_df = price_df.join(sentiment_10m['sum'].rename('sent_sum_10m'), how='left')
        price_df = price_df.join(sentiment_10m['count'].rename('sent_volume_10m'), how='left')
        price_df = price_df.join(sentiment_60m['mean'].rename('sent_pos_ratio_60m'), how='left')
        price_df = price_df.join(sentiment_60m['count'].rename('sent_volume_60m'), how='left')
        
        sentiment_cols = ['sent_mean_5m', 'sent_sum_10m', 'sent_pos_ratio_60m', 
                         'sent_volume_5m', 'sent_volume_10m', 'sent_volume_60m']
        price_df[sentiment_cols] = price_df[sentiment_cols].fillna(0)
        
        price_df = price_df.reset_index()
        
        logger.info(f"Created sentiment features: {sentiment_cols}")
        return price_df
        
    except Exception as e:
        logger.error(f"Failed to create sentiment features: {e}")
        return price_df

def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical analysis features for intraday data"""
    try:
        df['returns_5m'] = df['close'].pct_change(periods=1)
        df['returns_10m'] = df['close'].pct_change(periods=2)
        df['returns_60m'] = df['close'].pct_change(periods=12)
        
        df['volume_ma_5m'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_10m'] = df['volume'].rolling(window=10).mean()
        
        df['price_ma_5m'] = df['close'].rolling(window=5).mean()
        df['price_ma_10m'] = df['close'].rolling(window=10).mean()
        df['price_ma_60m'] = df['close'].rolling(window=12).mean()
        
        df['volatility_5m'] = df['returns_5m'].rolling(window=5).std()
        df['volatility_10m'] = df['returns_5m'].rolling(window=10).std()
        
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        
        df = df.ffill().fillna(0)
        
        logger.info("Created technical analysis features")
        return df
        
    except Exception as e:
        logger.error(f"Failed to create technical features: {e}")
        return df

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta).where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def process_symbol_features(symbol: str, date: str, include_sentiment: bool = False, test_mode: bool = False) -> pd.DataFrame:
    """Process features for a single symbol"""
    try:
        if test_mode:
            logger.info(f"TEST MODE: Creating mock features for {symbol}")
            timestamps = pd.date_range(start=f"{date} 09:30:00", end=f"{date} 16:00:00", freq='5min')
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': 100 + np.random.randn(len(timestamps)) * 2,
                'high': 102 + np.random.randn(len(timestamps)) * 2,
                'low': 98 + np.random.randn(len(timestamps)) * 2,
                'close': 100 + np.random.randn(len(timestamps)) * 2,
                'volume': np.random.randint(1000, 10000, len(timestamps))
            })
        else:
            s3_client = boto3.client('s3')
            s3_key = f"intraday/{symbol}/5min/{date}.parquet"
            local_file = f"/tmp/{symbol}_{date}_5min.parquet"
            
            s3_client.download_file('hpo-bucket-773934887314', s3_key, local_file)
            df = pd.read_parquet(local_file)
            os.remove(local_file)
        
        df = create_technical_features(df)
        
        if include_sentiment:
            sentiment_df = load_sentiment_data(symbol, date) if not test_mode else pd.DataFrame()
            df = create_sentiment_features(df, sentiment_df)
        
        logger.info(f"Processed features for {symbol}: {len(df)} rows, {len(df.columns)} features")
        return df
        
    except Exception as e:
        logger.error(f"Failed to process features for {symbol}: {e}")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description='Create intraday features with sentiment integration')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to process')
    parser.add_argument('--date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='Date to process (YYYY-MM-DD)')
    parser.add_argument('--include-sentiment', action='store_true', help='Include sentiment features')
    parser.add_argument('--test-sentiment', action='store_true', help='Test sentiment feature creation')
    parser.add_argument('--output-dir', type=str, default='data/features/', help='Output directory')
    
    args = parser.parse_args()
    
    logger.info(f"Creating intraday features for {args.symbol} on {args.date}")
    
    df = process_symbol_features(
        args.symbol, 
        args.date, 
        include_sentiment=args.include_sentiment or args.test_sentiment,
        test_mode=args.test_sentiment
    )
    
    if not df.empty:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{args.symbol}_{args.date}_features.parquet"
        df.to_parquet(output_file, index=False)
        
        logger.info(f"Saved features to {output_file}")
        
        if args.include_sentiment or args.test_sentiment:
            sentiment_cols = [col for col in df.columns if col.startswith('sent_')]
            logger.info(f"Sentiment features created: {sentiment_cols}")
            logger.info(f"Sample sentiment values: {df[sentiment_cols].head()}")
    else:
        logger.error("Failed to create features")
        sys.exit(1)

if __name__ == "__main__":
    main()
