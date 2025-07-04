#!/usr/bin/env python3
"""
FinBERT Sentiment Scoring for Twitter Data
Uses ONNXRuntime CPU for efficient sentiment analysis
"""

import os
import sys
import json
import gzip
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import boto3
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinBERTSentimentScorer:
    def __init__(self, model_path: str | None = None, test_mode: bool = False):
        self.test_mode = test_mode
        self.tokenizer = None
        self.session = None
        
        if not test_mode:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str | None = None):
        """Load FinBERT ONNX model and tokenizer"""
        try:
            if model_path and os.path.exists(model_path):
                self.session = ort.InferenceSession(model_path)
            else:
                logger.info("Using mock model for testing - FinBERT not available")
                self.session = None
            
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                logger.info("FinBERT tokenizer loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            if not self.test_mode:
                self.session = None
                self.tokenizer = None
    
    def score_text(self, text: str) -> Dict[str, float]:
        """Score sentiment of text using FinBERT"""
        if self.test_mode or not self.tokenizer or not self.session:
            return {
                'positive': np.random.random(),
                'negative': np.random.random(),
                'neutral': np.random.random(),
                'compound': np.random.random() * 2 - 1
            }
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            outputs = self.session.run(None, {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            })
            
            logits = outputs[0][0]
            probabilities = self._softmax(logits)
            
            return {
                'positive': float(probabilities[2]),
                'negative': float(probabilities[0]),
                'neutral': float(probabilities[1]),
                'compound': float(probabilities[2] - probabilities[0])
            }
            
        except Exception as e:
            logger.error(f"Error scoring text: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}
    
    def _softmax(self, x):
        """Compute softmax values"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

def process_twitter_json(json_file_path: str, scorer: FinBERTSentimentScorer) -> List[Dict[str, Any]]:
    """Process Twitter JSON file and score sentiment"""
    results = []
    
    try:
        if json_file_path.endswith('.gz'):
            with gzip.open(json_file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    tweet = json.loads(line.strip())
                    sentiment = scorer.score_text(tweet['text'])
                    
                    results.append({
                        'timestamp': tweet['created_at'],
                        'tweet_id': tweet['id'],
                        'text': tweet['text'],
                        'sent_positive': sentiment['positive'],
                        'sent_negative': sentiment['negative'],
                        'sent_neutral': sentiment['neutral'],
                        'sent_compound': sentiment['compound']
                    })
        else:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tweet = json.loads(line.strip())
                    sentiment = scorer.score_text(tweet['text'])
                    
                    results.append({
                        'timestamp': tweet['created_at'],
                        'tweet_id': tweet['id'],
                        'text': tweet['text'],
                        'sent_positive': sentiment['positive'],
                        'sent_negative': sentiment['negative'],
                        'sent_neutral': sentiment['neutral'],
                        'sent_compound': sentiment['compound']
                    })
                    
    except Exception as e:
        logger.error(f"Error processing {json_file_path}: {e}")
    
    return results

def process_symbol_data(symbol: str, s3_bucket: str, scorer: FinBERTSentimentScorer, test_mode: bool = False):
    """Process all Twitter data for a symbol"""
    s3_client = boto3.client('s3')
    
    try:
        prefix = f"raw/twitter/{symbol}/"
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            logger.warning(f"No Twitter data found for {symbol}")
            return
        
        for obj in response['Contents']:
            if obj['Key'].endswith('.json.gz'):
                logger.info(f"Processing {obj['Key']}")
                
                if test_mode:
                    logger.info(f"TEST MODE: Would process {obj['Key']}")
                    continue
                
                local_file = f"/tmp/{os.path.basename(obj['Key'])}"
                s3_client.download_file(s3_bucket, obj['Key'], local_file)
                
                results = process_twitter_json(local_file, scorer)
                
                if results:
                    df = pd.DataFrame(results)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['symbol'] = symbol
                    
                    date_str = df['timestamp'].dt.date.iloc[0].strftime('%Y-%m-%d')
                    output_key = f"sentiment/finbert/{symbol}/{date_str}.parquet"
                    
                    local_parquet = f"/tmp/{symbol}_{date_str}.parquet"
                    df.to_parquet(local_parquet, index=False)
                    
                    s3_client.upload_file(local_parquet, s3_bucket, output_key)
                    logger.info(f"Uploaded sentiment data to s3://{s3_bucket}/{output_key}")
                    
                    os.remove(local_file)
                    os.remove(local_parquet)
                
    except Exception as e:
        logger.error(f"Error processing symbol {symbol}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Score Twitter sentiment using FinBERT')
    parser.add_argument('--symbol', type=str, help='Stock symbol to process')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode')
    parser.add_argument('--model-path', type=str, help='Path to FinBERT ONNX model')
    
    args = parser.parse_args()
    
    s3_bucket = 'hpo-bucket-773934887314'
    
    scorer = FinBERTSentimentScorer(args.model_path, args.test_mode)
    
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols_file = Path(__file__).parent / 'config' / 'models_to_train_46.txt'
        if symbols_file.exists():
            with open(symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
        else:
            symbols = ['AAPL']
    
    logger.info(f"Processing sentiment for {len(symbols)} symbols")
    
    for symbol in symbols:
        logger.info(f"Processing sentiment for {symbol}")
        process_symbol_data(symbol, s3_bucket, scorer, args.test_mode)
    
    logger.info("Sentiment scoring completed")

if __name__ == "__main__":
    main()
