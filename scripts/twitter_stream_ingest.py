#!/usr/bin/env python3
"""
Twitter Stream Ingestion for 46-Stock Universe
Implements async Tweepy v2 API for filtered stream with cashtags
"""

import os
import sys
import json
import gzip
import asyncio
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
import boto3
import tweepy
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TwitterStreamProcessor(tweepy.StreamingClient):
    def __init__(self, bearer_token: str, symbols: List[str], s3_bucket: str, dry_run: bool = False):
        super().__init__(bearer_token)
        self.symbols = symbols
        self.s3_bucket = s3_bucket
        self.dry_run = dry_run
        self.s3_client = boto3.client('s3')
        self.tweet_buffer = {}
        
    def on_tweet(self, tweet):
        try:
            tweet_data = {
                'id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at.isoformat(),
                'author_id': tweet.author_id,
                'public_metrics': tweet.public_metrics,
                'context_annotations': tweet.context_annotations,
                'entities': tweet.entities
            }
            
            for symbol in self.symbols:
                cashtag = f"${symbol}"
                if cashtag.upper() in tweet.text.upper():
                    self._store_tweet(symbol, tweet_data)
                    
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
    
    def _store_tweet(self, symbol: str, tweet_data: Dict[str, Any]):
        try:
            date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            
            if self.dry_run:
                logger.info(f"DRY RUN: Would store tweet for {symbol}: {tweet_data['text'][:100]}...")
                return
            
            if symbol not in self.tweet_buffer:
                self.tweet_buffer[symbol] = []
            
            self.tweet_buffer[symbol].append(tweet_data)
            
            if len(self.tweet_buffer[symbol]) >= 100:
                self._flush_buffer(symbol, date_str)
                
        except Exception as e:
            logger.error(f"Error storing tweet for {symbol}: {e}")
    
    def _flush_buffer(self, symbol: str, date_str: str):
        try:
            if not self.tweet_buffer.get(symbol):
                return
                
            s3_key = f"raw/twitter/{symbol}/{date_str}.json.gz"
            
            if not self.dry_run:
                json_data = '\n'.join([json.dumps(tweet) for tweet in self.tweet_buffer[symbol]])
                compressed_data = gzip.compress(json_data.encode('utf-8'))
                
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=compressed_data,
                    ContentType='application/gzip'
                )
            else:
                logger.info(f"DRY RUN: Would store {len(self.tweet_buffer[symbol])} tweets to s3://{self.s3_bucket}/{s3_key}")
            
            logger.info(f"Stored {len(self.tweet_buffer[symbol])} tweets for {symbol} to s3://{self.s3_bucket}/{s3_key}")
            self.tweet_buffer[symbol] = []
            
        except Exception as e:
            logger.error(f"Error flushing buffer for {symbol}: {e}")

def get_twitter_bearer_token(dry_run=False):
    """Get Twitter Bearer Token from AWS Secrets Manager"""
    try:
        secrets_client = boto3.client('secretsmanager')
        response = secrets_client.get_secret_value(SecretId='twitter-api-keys')
        secrets = json.loads(response['SecretString'])
        return secrets['bearer_token']
    except Exception as e:
        logger.warning(f"Failed to get Twitter token from Secrets Manager: {e}")
        token = os.environ.get('TWITTER_BEARER_TOKEN')
        if not token and dry_run:
            logger.info("🧪 DRY RUN: Using mock Twitter Bearer Token")
            return "mock_token_for_dry_run_testing"
        return token

def get_stock_symbols():
    """Get 46 stock symbols from config"""
    symbols_file = Path(__file__).parent.parent / 'config' / 'models_to_train_46.txt'
    if symbols_file.exists():
        with open(symbols_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']

def main():
    parser = argparse.ArgumentParser(description='Twitter Stream Ingestion for Stock Sentiment')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    parser.add_argument('--duration', type=int, default=3600, help='Duration to run in seconds')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to track')
    
    args = parser.parse_args()
    
    bearer_token = get_twitter_bearer_token(args.dry_run)
    if not bearer_token:
        logger.error("Twitter Bearer Token not found. Set TWITTER_BEARER_TOKEN or configure AWS Secrets Manager")
        sys.exit(1)
    
    symbols = args.symbols or get_stock_symbols()
    s3_bucket = 'hpo-bucket-773934887314'
    
    logger.info(f"Starting Twitter stream for {len(symbols)} symbols: {symbols[:5]}...")
    
    stream = TwitterStreamProcessor(bearer_token, symbols, s3_bucket, args.dry_run)
    
    cashtags = [f"${symbol}" for symbol in symbols]
    rules = [tweepy.StreamRule(f"({' OR '.join(cashtags)}) lang:en")]
    
    try:
        stream.add_rules(rules)
        logger.info(f"Added stream rules for cashtags: {cashtags[:5]}...")
        
        if args.dry_run:
            logger.info("DRY RUN: Would start Twitter stream")
            import time
            time.sleep(5)
        else:
            stream.filter(tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations', 'entities'])
            
    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")
    except Exception as e:
        logger.error(f"Stream error: {e}")
    finally:
        stream.disconnect()

if __name__ == "__main__":
    main()
