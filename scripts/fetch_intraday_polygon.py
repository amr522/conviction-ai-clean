#!/usr/bin/env python3
"""
Fetch intraday data from Polygon API for multi-timeframe technical analysis
"""
import os
import boto3
import pandas as pd
import requests
from datetime import datetime, timedelta
import argparse
import logging
import json
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntradayDataFetcher:
    def __init__(self, api_key=None, s3_bucket='hpo-bucket-773934887314'):
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"
        
        if not self.api_key:
            raise ValueError("Polygon API key not found. Set POLYGON_API_KEY environment variable.")
    
    def get_symbols_list(self, symbols_file='config/models_to_train_46.txt') -> List[str]:
        """Get list of symbols to fetch data for"""
        try:
            with open(symbols_file, 'r') as f:
                symbols = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
            return symbols
        except FileNotFoundError:
            logger.warning(f"Symbols file {symbols_file} not found, using default symbols")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
    
    def fetch_intraday_bars(self, symbol: str, interval: int, start_date: str, end_date: str, dry_run: bool = False) -> pd.DataFrame:
        """Fetch intraday bars from Polygon API"""
        if dry_run:
            logger.info(f"🧪 DRY RUN: Would fetch {interval}min bars for {symbol} from {start_date} to {end_date}")
            return self._generate_mock_data(symbol, interval, start_date, end_date)
        
        url = f"{self.base_url}/{symbol}/range/{interval}/minute/{start_date}/{end_date}"
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            logger.info(f"Fetching {interval}min bars for {symbol}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'results' not in data or not data['results']:
                logger.warning(f"No data returned for {symbol} {interval}min bars")
                return pd.DataFrame()
            
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df['symbol'] = symbol
            df['interval'] = interval
            df['date'] = df['timestamp'].dt.date
            
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df = df[['timestamp', 'date', 'symbol', 'interval', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Fetched {len(df)} {interval}min bars for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def _generate_mock_data(self, symbol: str, interval: int, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate mock intraday data for dry run testing"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        timestamps = pd.date_range(start=start, end=end, freq=f'{interval}min')
        timestamps = timestamps[timestamps.indexer_between_time('09:30', '16:00')]
        
        n_points = min(len(timestamps), 1000)
        timestamps = timestamps[:n_points]
        
        base_price = 100.0
        data = []
        
        for ts in timestamps:
            price = base_price + (hash(f"{symbol}{ts}") % 1000) / 100
            data.append({
                'timestamp': ts,
                'date': ts.date(),
                'symbol': symbol,
                'interval': interval,
                'open': price,
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price * 1.01,
                'volume': 10000 + (hash(f"{symbol}{ts}") % 50000)
            })
        
        return pd.DataFrame(data)
    
    def save_to_s3(self, df: pd.DataFrame, symbol: str, interval: int, date: str, dry_run: bool = False) -> bool:
        """Save intraday data to S3"""
        if df.empty:
            return False
        
        s3_key = f"intraday/{symbol}/{interval}min/{date}.csv"
        
        if dry_run:
            logger.info(f"🧪 DRY RUN: Would save {len(df)} rows to s3://{self.s3_bucket}/{s3_key}")
            return True
        
        try:
            csv_buffer = df.to_csv(index=False)
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=csv_buffer,
                ContentType='text/csv'
            )
            logger.info(f"Saved {len(df)} rows to s3://{self.s3_bucket}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to S3: {e}")
            return False
    
    def fetch_and_store_intraday_data(self, symbols: List[str], intervals: List[int], 
                                    start_date: str, end_date: str, dry_run: bool = False) -> Dict:
        """Fetch and store intraday data for multiple symbols and intervals"""
        results = {
            'success': 0,
            'failed': 0,
            'symbols_processed': [],
            'errors': []
        }
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    df = self.fetch_intraday_bars(symbol, interval, start_date, end_date, dry_run)
                    
                    if not df.empty:
                        dates = df['date'].unique()
                        for date in dates:
                            date_df = df[df['date'] == date]
                            success = self.save_to_s3(date_df, symbol, interval, str(date), dry_run)
                            
                            if success:
                                results['success'] += 1
                            else:
                                results['failed'] += 1
                    
                    results['symbols_processed'].append(f"{symbol}_{interval}min")
                    
                except Exception as e:
                    error_msg = f"Failed to process {symbol} {interval}min: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    results['failed'] += 1
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Fetch intraday data from Polygon API")
    parser.add_argument('--intervals', nargs='+', type=int, default=[5, 10, 60],
                       help='Intervals in minutes (default: 5 10 60)')
    parser.add_argument('--symbols-file', default='config/models_to_train_46.txt',
                       help='File containing symbols to fetch')
    parser.add_argument('--start-date', default='2022-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry-run mode (no API calls or S3 uploads)')
    parser.add_argument('--api-key', help='Polygon API key (overrides environment variable)')
    
    args = parser.parse_args()
    
    try:
        fetcher = IntradayDataFetcher(api_key=args.api_key)
        symbols = fetcher.get_symbols_list(args.symbols_file)
        
        logger.info(f"Starting intraday data fetch:")
        logger.info(f"  Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
        logger.info(f"  Intervals: {args.intervals} minutes")
        logger.info(f"  Date range: {args.start_date} to {args.end_date}")
        logger.info(f"  Dry run: {args.dry_run}")
        
        results = fetcher.fetch_and_store_intraday_data(
            symbols=symbols,
            intervals=args.intervals,
            start_date=args.start_date,
            end_date=args.end_date,
            dry_run=args.dry_run
        )
        
        logger.info("Intraday data fetch completed:")
        logger.info(f"  Successful: {results['success']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Symbols processed: {len(results['symbols_processed'])}")
        
        if results['errors']:
            logger.warning(f"Errors encountered: {len(results['errors'])}")
            for error in results['errors'][:5]:
                logger.warning(f"  {error}")
        
        return 0 if results['failed'] == 0 else 1
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
