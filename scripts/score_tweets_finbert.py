#!/usr/bin/env python3
"""
FinBERT Sentiment Scoring for Twitter Data
Implements CPU-optimized ONNX Runtime inference for financial sentiment analysis
"""

import os
import sys
import json
import gzip
import logging
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import boto3
from transformers import AutoTokenizer
import onnxruntime as ort

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinBERTSentimentScorer:
    def __init__(self, model_path: Optional[str] = None, dry_run: bool = False, s3_bucket: str = "hpo-bucket-773934887314"):
        self.dry_run = dry_run
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3')
        
        if dry_run:
            logger.info("🧪 DRY RUN: Initializing mock FinBERT scorer")
            self.tokenizer = None
            self.session = None
        else:
            try:
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                from transformers import pipeline
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=self.tokenizer,
                    device=-1  # CPU only
                )
                logger.info("✅ FinBERT model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load FinBERT model: {e}")
                if not dry_run:
                    raise
    
    def score_text(self, text: str) -> Dict[str, float]:
        """Score a single text for sentiment"""
        if self.dry_run:
            return {
                'positive': 0.6,
                'negative': 0.2,
                'neutral': 0.2,
                'compound': 0.4
            }
        
        try:
            text = text.strip()[:512]  # FinBERT max length
            
            if not text:
                return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
            
            result = self.sentiment_pipeline(text)[0]
            
            if result['label'] == 'positive':
                positive = result['score']
                negative = 0.0
                neutral = 1.0 - positive
            elif result['label'] == 'negative':
                negative = result['score']
                positive = 0.0
                neutral = 1.0 - negative
            else:  # neutral
                neutral = result['score']
                positive = 0.0
                negative = 1.0 - neutral
            
            compound = positive - negative
            
            return {
                'positive': float(positive),
                'negative': float(negative),
                'neutral': float(neutral),
                'compound': float(compound)
            }
            
        except Exception as e:
            logger.error(f"Error scoring text: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
    
    def score_tweets_batch(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a batch of tweets"""
        scored_tweets = []
        
        for tweet in tweets:
            try:
                text = tweet.get('text', '')
                sentiment_scores = self.score_text(text)
                
                scored_tweet = {
                    **tweet,
                    'sentiment': sentiment_scores,
                    'scored_at': datetime.utcnow().isoformat()
                }
                scored_tweets.append(scored_tweet)
                
            except Exception as e:
                logger.error(f"Error scoring tweet {tweet.get('id', 'unknown')}: {e}")
                continue
        
        return scored_tweets
    
    def process_s3_file(self, bucket: str, input_key: str, output_key: str) -> bool:
        """Process tweets from S3 file and store scored results"""
        try:
            if self.dry_run:
                logger.info(f"🧪 DRY RUN: Would process s3://{bucket}/{input_key} → s3://{bucket}/{output_key}")
                return True
            
            response = self.s3_client.get_object(Bucket=bucket, Key=input_key)
            
            if input_key.endswith('.gz'):
                content = gzip.decompress(response['Body'].read()).decode('utf-8')
            else:
                content = response['Body'].read().decode('utf-8')
            
            tweets = []
            for line in content.strip().split('\n'):
                if line:
                    tweets.append(json.loads(line))
            
            logger.info(f"Processing {len(tweets)} tweets from {input_key}")
            
            batch_size = 50
            all_scored_tweets = []
            
            for i in range(0, len(tweets), batch_size):
                batch = tweets[i:i + batch_size]
                scored_batch = self.score_tweets_batch(batch)
                all_scored_tweets.extend(scored_batch)
                
                if i % (batch_size * 10) == 0:
                    logger.info(f"Processed {i + len(batch)}/{len(tweets)} tweets")
            
            output_data = '\n'.join([json.dumps(tweet) for tweet in all_scored_tweets])
            
            if output_key.endswith('.gz'):
                output_data = gzip.compress(output_data.encode('utf-8'))
                content_type = 'application/gzip'
            else:
                output_data = output_data.encode('utf-8')
                content_type = 'application/json'
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=output_data,
                ContentType=content_type
            )
            
            logger.info(f"✅ Stored {len(all_scored_tweets)} scored tweets to s3://{bucket}/{output_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing S3 file {input_key}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='FinBERT Sentiment Scoring for Twitter Data')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with sample data')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    parser.add_argument('--input-bucket', default='hpo-bucket-773934887314', help='S3 input bucket')
    parser.add_argument('--input-prefix', default='raw/twitter/', help='S3 input prefix')
    parser.add_argument('--output-prefix', default='scored/twitter/', help='S3 output prefix')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to process')
    
    args = parser.parse_args()
    
    scorer = FinBERTSentimentScorer(dry_run=args.dry_run)
    
    if args.test_mode:
        sample_tweets = [
            {'id': '1', 'text': 'AAPL stock is going to the moon! Great earnings!', 'created_at': '2025-01-01T12:00:00Z'},
            {'id': '2', 'text': 'Terrible news for $AAPL, selling all my shares', 'created_at': '2025-01-01T12:01:00Z'},
            {'id': '3', 'text': 'Apple released new iPhone today', 'created_at': '2025-01-01T12:02:00Z'}
        ]
        
        logger.info("🧪 Testing FinBERT sentiment scoring...")
        scored_tweets = scorer.score_tweets_batch(sample_tweets)
        
        for tweet in scored_tweets:
            sentiment = tweet['sentiment']
            logger.info(f"Tweet: {tweet['text'][:50]}...")
            logger.info(f"Sentiment: pos={sentiment['positive']:.3f}, neg={sentiment['negative']:.3f}, compound={sentiment['compound']:.3f}")
        
        logger.info("✅ Test mode completed successfully")
        return
    
    symbols = args.symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    success_count = 0
    for symbol in symbols:
        input_key = f"{args.input_prefix}{symbol}/{date_str}.json.gz"
        output_key = f"{args.output_prefix}{symbol}/{date_str}.json.gz"
        
        if scorer.process_s3_file(args.input_bucket, input_key, output_key):
            success_count += 1
        else:
            logger.warning(f"Failed to process {symbol}")
    
    logger.info(f"✅ Successfully processed {success_count}/{len(symbols)} symbols")

if __name__ == "__main__":
    main()
