#!/usr/bin/env python3
"""
Multi-timeframe technical analysis feature engineering
"""
import pandas as pd
import numpy as np
import boto3
import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntradayFeatureEngineer:
    def __init__(self, s3_bucket='hpo-bucket-773934887314'):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        self.intervals = [5, 10, 60]
        
    def calculate_technical_indicators(self, df: pd.DataFrame, interval: int) -> pd.DataFrame:
        """Calculate technical indicators for specific timeframe"""
        if df.empty:
            return df
        
        features = df.copy()
        features = features.sort_values('timestamp')
        
        try:
            features[f'vwap_{interval}m'] = self._calculate_vwap(features)
            features[f'intraday_vol_{interval}m'] = features['close'].rolling(20, min_periods=1).std()
            features[f'atr_{interval}m'] = self._calculate_atr(features, period=14)
            features[f'rsi_{interval}m'] = self._calculate_rsi(features['close'], period=14)
            features[f'stoch_rsi_{interval}m'] = self._calculate_stoch_rsi(features['close'], period=14)
            features[f'bb_upper_{interval}m'], features[f'bb_lower_{interval}m'] = self._calculate_bollinger_bands(features['close'])
            features[f'macd_{interval}m'], features[f'macd_signal_{interval}m'] = self._calculate_macd(features['close'])
            features[f'volume_sma_{interval}m'] = features['volume'].rolling(20, min_periods=1).mean()
            features[f'price_momentum_{interval}m'] = features['close'].pct_change(periods=10)
            features[f'volume_ratio_{interval}m'] = features['volume'] / features[f'volume_sma_{interval}m']
            
            logger.info(f"Calculated {interval}min technical indicators for {len(features)} bars")
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {interval}min: {e}")
        
        return features
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.Series(np.maximum(high_low, np.maximum(high_close, low_close)))
        return true_range.rolling(period, min_periods=1).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=1).mean()
        
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def _calculate_stoch_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Stochastic RSI"""
        rsi = self._calculate_rsi(prices, period)
        stoch_rsi = (rsi - rsi.rolling(period, min_periods=1).min()) / (
            rsi.rolling(period, min_periods=1).max() - rsi.rolling(period, min_periods=1).min()
        )
        return stoch_rsi * 100
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period, min_periods=1).mean()
        std = prices.rolling(period, min_periods=1).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, lower_band
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        
        return macd, signal_line
    
    def aggregate_to_daily(self, intraday_df: pd.DataFrame, interval: int) -> pd.DataFrame:
        """Aggregate intraday features to daily resolution"""
        if intraday_df.empty:
            return pd.DataFrame()
        
        try:
            intraday_df['date'] = pd.to_datetime(intraday_df['timestamp']).dt.date
            
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                f'vwap_{interval}m': 'last',
                f'intraday_vol_{interval}m': ['mean', 'max'],
                f'atr_{interval}m': 'mean',
                f'rsi_{interval}m': 'last',
                f'stoch_rsi_{interval}m': 'last',
                f'bb_upper_{interval}m': 'last',
                f'bb_lower_{interval}m': 'last',
                f'macd_{interval}m': 'last',
                f'macd_signal_{interval}m': 'last',
                f'volume_sma_{interval}m': 'last',
                f'price_momentum_{interval}m': 'last',
                f'volume_ratio_{interval}m': 'mean'
            }
            
            daily_agg = intraday_df.groupby(['symbol', 'date']).agg(agg_dict)
            
            daily_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in daily_agg.columns.values]
            daily_agg = daily_agg.round(6)
            
            logger.info(f"Aggregated {len(intraday_df)} {interval}min bars to {len(daily_agg)} daily bars")
            
            return daily_agg.reset_index()
            
        except Exception as e:
            logger.error(f"Error aggregating {interval}min data to daily: {e}")
            return pd.DataFrame()
    
    def load_intraday_data_from_s3(self, symbol: str, interval: int, start_date: str, end_date: str, dry_run: bool = False) -> pd.DataFrame:
        """Load intraday data from S3"""
        if dry_run:
            logger.info(f"🧪 DRY RUN: Would load {interval}min data for {symbol} from S3")
            return self._generate_mock_intraday_data(symbol, interval, start_date, end_date)
        
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            all_data = []
            current_date = start
            
            while current_date <= end:
                s3_key = f"intraday/{symbol}/{interval}min/{current_date.strftime('%Y-%m-%d')}.csv"
                
                try:
                    response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
                    df = pd.read_csv(response['Body'])
                    all_data.append(df)
                except self.s3_client.exceptions.NoSuchKey:
                    pass
                except Exception as e:
                    logger.warning(f"Error loading {s3_key}: {e}")
                
                current_date += timedelta(days=1)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
                logger.info(f"Loaded {len(combined_df)} {interval}min bars for {symbol} from S3")
                return combined_df
            else:
                logger.warning(f"No {interval}min data found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _generate_mock_intraday_data(self, symbol: str, interval: int, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate mock intraday data for testing"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        timestamps = pd.date_range(start=start, end=end, freq=f'{interval}min')
        timestamps = timestamps[timestamps.indexer_between_time('09:30', '16:00')]
        
        n_points = min(len(timestamps), 500)
        timestamps = timestamps[:n_points]
        
        base_price = 100.0
        data = []
        
        for ts in timestamps:
            price = base_price + (hash(f"{symbol}{ts}") % 1000) / 100
            data.append({
                'timestamp': ts,
                'symbol': symbol,
                'interval': interval,
                'open': price,
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price * 1.01,
                'volume': 10000 + (hash(f"{symbol}{ts}") % 50000)
            })
        
        return pd.DataFrame(data)
    
    def process_symbol_intraday_features(self, symbol: str, intervals: List[int], 
                                       start_date: str, end_date: str, dry_run: bool = False) -> pd.DataFrame:
        """Process intraday features for a single symbol across multiple timeframes"""
        all_daily_features = []
        
        for interval in intervals:
            logger.info(f"Processing {interval}min features for {symbol}")
            
            intraday_df = self.load_intraday_data_from_s3(symbol, interval, start_date, end_date, dry_run)
            
            if not intraday_df.empty:
                intraday_with_features = self.calculate_technical_indicators(intraday_df, interval)
                daily_features = self.aggregate_to_daily(intraday_with_features, interval)
                
                if not daily_features.empty:
                    all_daily_features.append(daily_features)
        
        if all_daily_features:
            combined_features = all_daily_features[0]
            
            for df in all_daily_features[1:]:
                combined_features = pd.merge(
                    combined_features, df, 
                    on=['symbol', 'date'], 
                    how='outer',
                    suffixes=('', '_dup')
                )
                
                dup_cols = [col for col in combined_features.columns if col.endswith('_dup')]
                combined_features = combined_features.drop(columns=dup_cols)
            
            logger.info(f"Combined features for {symbol}: {len(combined_features)} days, {len(combined_features.columns)} features")
            return combined_features
        
        return pd.DataFrame()
    
    def save_features_to_s3(self, df: pd.DataFrame, symbol: str, dry_run: bool = False) -> bool:
        """Save processed features to S3"""
        if df.empty:
            return False
        
        s3_key = f"features/intraday/{symbol}_intraday_features.csv"
        
        if dry_run:
            logger.info(f"🧪 DRY RUN: Would save {len(df)} feature rows to s3://{self.s3_bucket}/{s3_key}")
            return True
        
        try:
            csv_buffer = df.to_csv(index=False)
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=csv_buffer,
                ContentType='text/csv'
            )
            logger.info(f"Saved {len(df)} feature rows to s3://{self.s3_bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save features to S3: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Generate intraday technical analysis features")
    parser.add_argument('--symbols-file', default='config/models_to_train_46.txt',
                       help='File containing symbols to process')
    parser.add_argument('--intervals', nargs='+', type=int, default=[5, 10, 60],
                       help='Intervals in minutes (default: 5 10 60)')
    parser.add_argument('--start-date', default='2022-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry-run mode (no S3 access)')
    parser.add_argument('--test', action='store_true',
                       help='Run test with single symbol')
    
    args = parser.parse_args()
    
    try:
        engineer = IntradayFeatureEngineer()
        
        if args.test:
            symbols = ['AAPL']
            logger.info("Running test mode with AAPL only")
        else:
            try:
                with open(args.symbols_file, 'r') as f:
                    symbols = [line.strip() for line in f.readlines() if line.strip()]
            except FileNotFoundError:
                logger.warning(f"Symbols file {args.symbols_file} not found, using default symbols")
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        logger.info(f"Processing intraday features:")
        logger.info(f"  Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
        logger.info(f"  Intervals: {args.intervals} minutes")
        logger.info(f"  Date range: {args.start_date} to {args.end_date}")
        logger.info(f"  Dry run: {args.dry_run}")
        
        success_count = 0
        failed_count = 0
        
        for symbol in symbols:
            try:
                features_df = engineer.process_symbol_intraday_features(
                    symbol=symbol,
                    intervals=args.intervals,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    dry_run=args.dry_run
                )
                
                if not features_df.empty:
                    success = engineer.save_features_to_s3(features_df, symbol, args.dry_run)
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                else:
                    logger.warning(f"No features generated for {symbol}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                failed_count += 1
        
        logger.info(f"Feature engineering completed:")
        logger.info(f"  Successful: {success_count}")
        logger.info(f"  Failed: {failed_count}")
        
        return 0 if failed_count == 0 else 1
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
