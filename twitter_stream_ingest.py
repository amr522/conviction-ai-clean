#!/usr/bin/env python3
"""
Twitter Stream Ingestion with Async Tweepy v2
Real-time Twitter sentiment data collection for financial symbols
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Set, Optional, Dict, Any
import boto3
import tweepy
from botocore.exceptions import ClientError

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.twitter_secrets_manager import get_twitter_credentials

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/twitter_stream_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TwitterStreamIngester:
    """Async Twitter stream ingester for financial sentiment analysis"""
    
    def __init__(self, 
                 target_symbols: Optional[List[str]] = None,
                 s3_bucket: str = "conviction-ai-data",
                 s3_prefix: str = "twitter-sentiment/raw-tweets/",
                 region_name: str = "us-east-1"):
        """
        Initialize Twitter Stream Ingester
        
        Args:
            target_symbols: List of stock symbols to track (e.g., ['AAPL', 'GOOGL'])
            s3_bucket: S3 bucket for storing tweets
            s3_prefix: S3 prefix for organizing tweet data
            region_name: AWS region
        """
        self.target_symbols = target_symbols or [
            'AAPL', 'AMZN', 'GOOGL', 'TSLA', 'META', 'MSFT', 
            'NVDA', 'JPM', 'JNJ', 'V', 'MA'
        ]
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.region_name = region_name
        
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.twitter_client = None
        self.stream_client = None
        
        self.is_streaming = False
        self.tweet_buffer = []
        self.buffer_size = 100  # Buffer tweets before S3 upload
        self.last_upload = datetime.now(timezone.utc)
        self.upload_interval = 300  # Upload every 5 minutes
        
        self.financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'forecast',
            'bullish', 'bearish', 'buy', 'sell', 'hold', 'upgrade', 'downgrade',
            'analyst', 'target', 'price', 'stock', 'shares', 'market',
            'trading', 'volume', 'breakout', 'support', 'resistance',
            'dividend', 'split', 'merger', 'acquisition', 'ipo'
        ]
        
        logger.info(f"Initialized Twitter Stream Ingester for symbols: {self.target_symbols}")
    
    async def initialize_twitter_clients(self):
        """Initialize Twitter API clients with credentials"""
        try:
            credentials = get_twitter_credentials(region_name=self.region_name)
            
            self.twitter_client = tweepy.Client(
                bearer_token=credentials['bearer_token'],
                consumer_key=credentials['api_key'],
                consumer_secret=credentials['api_secret'],
                access_token=credentials['access_token'],
                access_token_secret=credentials['access_token_secret'],
                wait_on_rate_limit=True
            )
            
            try:
                me = self.twitter_client.get_me()
                logger.info(f"Twitter authentication successful. User: {me.data.username}")
            except Exception as e:
                logger.error(f"Twitter authentication test failed: {e}")
                raise
                
            logger.info("Twitter clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Twitter clients: {e}")
            raise
    
    def create_stream_rules(self) -> List[Dict[str, str]]:
        """
        Create Twitter stream rules for financial symbols and keywords
        
        Returns:
            List of rule dictionaries for Twitter API
        """
        rules = []
        
        for symbol in self.target_symbols:
            rules.append({
                'value': f'${symbol}',
                'tag': f'cashtag_{symbol}'
            })
        
        cashtags = ' OR '.join([f'${symbol}' for symbol in self.target_symbols])
        financial_keywords_str = ' OR '.join(self.financial_keywords)
        
        rules.append({
            'value': f'({cashtags}) ({financial_keywords_str})',
            'tag': 'financial_keywords'
        })
        
        company_names = {
            'AAPL': 'Apple',
            'GOOGL': 'Google OR Alphabet',
            'TSLA': 'Tesla',
            'META': 'Meta OR Facebook',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'NVDA': 'NVIDIA'
        }
        
        for symbol, company_name in company_names.items():
            if symbol in self.target_symbols:
                rules.append({
                    'value': f'({company_name}) (stock OR shares OR trading)',
                    'tag': f'company_{symbol}'
                })
        
        logger.info(f"Created {len(rules)} stream rules")
        return rules
    
    async def setup_stream_rules(self):
        """Setup Twitter stream rules"""
        try:
            if not self.twitter_client:
                raise Exception("Twitter client not initialized. Call initialize_twitter_clients() first.")
                
            existing_rules = self.twitter_client.get_stream_rules()
            if existing_rules.data:
                rule_ids = [rule.id for rule in existing_rules.data]
                self.twitter_client.delete_stream_rules(rule_ids)
                logger.info(f"Deleted {len(rule_ids)} existing stream rules")
            
            rules = self.create_stream_rules()
            rule_values = [rule['value'] for rule in rules]
            
            response = self.twitter_client.add_stream_rules(
                tweepy.StreamRule(value=rule['value'], tag=rule['tag']) 
                for rule in rules
            )
            
            if response.data:
                logger.info(f"Successfully added {len(response.data)} stream rules")
                for rule in response.data:
                    logger.info(f"  Rule: {rule.value} (tag: {rule.tag})")
            else:
                logger.error("Failed to add stream rules")
                raise Exception("No stream rules were added")
                
        except Exception as e:
            logger.error(f"Failed to setup stream rules: {e}")
            raise
    
    def extract_symbol_from_tweet(self, tweet_text: str, matching_rules: List[str]) -> Optional[str]:
        """
        Extract the primary stock symbol from tweet text and matching rules
        
        Args:
            tweet_text: The tweet text content
            matching_rules: List of rule tags that matched this tweet
            
        Returns:
            Primary stock symbol or None
        """
        for rule_tag in matching_rules:
            if rule_tag.startswith('cashtag_'):
                return rule_tag.replace('cashtag_', '')
            elif rule_tag.startswith('company_'):
                return rule_tag.replace('company_', '')
        
        tweet_upper = tweet_text.upper()
        for symbol in self.target_symbols:
            if f'${symbol}' in tweet_upper:
                return symbol
        
        return self.target_symbols[0] if self.target_symbols else None
    
    def process_tweet(self, tweet_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process and enrich tweet data for storage
        
        Args:
            tweet_data: Raw tweet data from Twitter API
            
        Returns:
            Processed tweet data
        """
        try:
            tweet_id = tweet_data.get('id')
            tweet_text = tweet_data.get('text', '')
            created_at = tweet_data.get('created_at')
            author_id = tweet_data.get('author_id')
            
            matching_rules = []
            if 'matching_rules' in tweet_data:
                matching_rules = [rule.get('tag', '') for rule in tweet_data['matching_rules']]
            
            primary_symbol = self.extract_symbol_from_tweet(tweet_text, matching_rules)
            
            public_metrics = tweet_data.get('public_metrics', {})
            
            processed_tweet = {
                'tweet_id': tweet_id,
                'text': tweet_text,
                'created_at': created_at,
                'author_id': author_id,
                'primary_symbol': primary_symbol,
                'matching_rules': matching_rules,
                'retweet_count': public_metrics.get('retweet_count', 0),
                'like_count': public_metrics.get('like_count', 0),
                'reply_count': public_metrics.get('reply_count', 0),
                'quote_count': public_metrics.get('quote_count', 0),
                'processed_at': datetime.now(timezone.utc).isoformat(),
                'ingestion_source': 'twitter_stream_v2'
            }
            
            return processed_tweet
            
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            return None
    
    async def upload_tweets_to_s3(self, tweets: List[Dict[str, Any]]):
        """
        Upload tweet batch to S3 with date/symbol partitioning
        
        Args:
            tweets: List of processed tweet dictionaries
        """
        if not tweets:
            return
            
        try:
            grouped_tweets = {}
            for tweet in tweets:
                symbol = tweet.get('primary_symbol', 'UNKNOWN')
                created_at = datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00'))
                date_str = created_at.strftime('%Y/%m/%d')
                hour_str = created_at.strftime('%H')
                
                key = f"{symbol}/{date_str}/{hour_str}"
                if key not in grouped_tweets:
                    grouped_tweets[key] = []
                grouped_tweets[key].append(tweet)
            
            for partition_key, partition_tweets in grouped_tweets.items():
                timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                s3_key = f"{self.s3_prefix}{partition_key}/tweets_{timestamp}.json"
                
                json_lines = '\n'.join([json.dumps(tweet) for tweet in partition_tweets])
                
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=json_lines.encode('utf-8'),
                    ContentType='application/json'
                )
                
                logger.info(f"Uploaded {len(partition_tweets)} tweets to s3://{self.s3_bucket}/{s3_key}")
            
            logger.info(f"Successfully uploaded {len(tweets)} tweets in {len(grouped_tweets)} partitions")
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload: {e}")
            raise
    
    async def process_tweet_buffer(self):
        """Process and upload buffered tweets"""
        if not self.tweet_buffer:
            return
            
        try:
            tweets_to_upload = self.tweet_buffer.copy()
            self.tweet_buffer.clear()
            
            await self.upload_tweets_to_s3(tweets_to_upload)
            self.last_upload = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error processing tweet buffer: {e}")
            self.tweet_buffer.extend(tweets_to_upload)
    
    async def start_streaming(self):
        """Start the Twitter stream"""
        try:
            await self.initialize_twitter_clients()
            await self.setup_stream_rules()
            
            logger.info("Starting Twitter stream...")
            self.is_streaming = True
            
            stream = tweepy.StreamingClient(
                bearer_token=get_twitter_credentials(self.region_name)['bearer_token'],
                wait_on_rate_limit=True
            )
            
            class TwitterStreamHandler(tweepy.StreamingClient):
                def __init__(self, ingester, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.ingester = ingester
                
                def on_tweet(self, tweet):
                    try:
                        tweet_data = {
                            'id': tweet.id,
                            'text': tweet.text,
                            'created_at': tweet.created_at.isoformat(),
                            'author_id': tweet.author_id,
                            'public_metrics': {
                                'retweet_count': getattr(tweet, 'retweet_count', 0),
                                'like_count': getattr(tweet, 'like_count', 0),
                                'reply_count': getattr(tweet, 'reply_count', 0),
                                'quote_count': getattr(tweet, 'quote_count', 0)
                            }
                        }
                        
                        processed_tweet = self.ingester.process_tweet(tweet_data)
                        if processed_tweet:
                            self.ingester.tweet_buffer.append(processed_tweet)
                            
                            if (len(self.ingester.tweet_buffer) >= self.ingester.buffer_size or
                                (datetime.now(timezone.utc) - self.ingester.last_upload).seconds >= self.ingester.upload_interval):
                                asyncio.create_task(self.ingester.process_tweet_buffer())
                        
                    except Exception as e:
                        logger.error(f"Error handling tweet: {e}")
                
                def on_error(self, status_code):
                    logger.error(f"Twitter stream error: {status_code}")
                    if status_code == 420:
                        logger.error("Rate limit exceeded. Stopping stream.")
                        return False
                    return True
                
                def on_disconnect(self):
                    logger.warning("Twitter stream disconnected")
            
            handler = TwitterStreamHandler(
                self,
                bearer_token=get_twitter_credentials(self.region_name)['bearer_token'],
                wait_on_rate_limit=True
            )
            
            handler.filter(
                tweet_fields=['created_at', 'author_id', 'public_metrics'],
                threaded=True
            )
            
            logger.info("Twitter stream started successfully")
            
            while self.is_streaming:
                await asyncio.sleep(60)  # Check every minute
                
                if self.tweet_buffer and (datetime.now(timezone.utc) - self.last_upload).seconds >= self.upload_interval:
                    await self.process_tweet_buffer()
            
        except Exception as e:
            logger.error(f"Error starting Twitter stream: {e}")
            raise
        finally:
            if self.tweet_buffer:
                await self.process_tweet_buffer()
    
    async def stop_streaming(self):
        """Stop the Twitter stream"""
        logger.info("Stopping Twitter stream...")
        self.is_streaming = False
        
        if self.tweet_buffer:
            await self.process_tweet_buffer()
        
        logger.info("Twitter stream stopped")


async def main():
    """Main function for running Twitter stream ingestion"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Twitter Stream Ingestion for Financial Sentiment')
    parser.add_argument('--symbols', nargs='+', 
                        default=['AAPL', 'AMZN', 'GOOGL', 'TSLA', 'META', 'MSFT', 'NVDA', 'JPM', 'JNJ', 'V', 'MA'],
                        help='Stock symbols to track')
    parser.add_argument('--s3-bucket', default='conviction-ai-data',
                        help='S3 bucket for storing tweets')
    parser.add_argument('--s3-prefix', default='twitter-sentiment/raw-tweets/',
                        help='S3 prefix for organizing tweet data')
    parser.add_argument('--region', default='us-east-1',
                        help='AWS region')
    parser.add_argument('--buffer-size', type=int, default=100,
                        help='Tweet buffer size before S3 upload')
    parser.add_argument('--upload-interval', type=int, default=300,
                        help='Upload interval in seconds')
    parser.add_argument('--dry-run', action='store_true',
                        help='Test configuration without streaming')
    
    args = parser.parse_args()
    
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🚀 Starting Twitter Stream Ingestion")
    logger.info(f"   Symbols: {args.symbols}")
    logger.info(f"   S3 Bucket: {args.s3_bucket}")
    logger.info(f"   S3 Prefix: {args.s3_prefix}")
    logger.info(f"   Buffer Size: {args.buffer_size}")
    logger.info(f"   Upload Interval: {args.upload_interval}s")
    
    try:
        ingester = TwitterStreamIngester(
            target_symbols=args.symbols,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            region_name=args.region
        )
        
        ingester.buffer_size = args.buffer_size
        ingester.upload_interval = args.upload_interval
        
        if args.dry_run:
            logger.info("🧪 DRY RUN MODE - Testing configuration")
            await ingester.initialize_twitter_clients()
            await ingester.setup_stream_rules()
            logger.info("✅ Configuration test successful")
        else:
            await ingester.start_streaming()
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping...")
        if 'ingester' in locals():
            await ingester.stop_streaming()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
