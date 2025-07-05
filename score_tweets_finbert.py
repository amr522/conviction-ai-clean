#!/usr/bin/env python3
"""
FinBERT Sentiment Scoring for Twitter Data
CPU-optimized sentiment analysis using ONNX Runtime and FinBERT model
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import boto3
from botocore.exceptions import ClientError
import onnxruntime as ort
from transformers import AutoTokenizer
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.twitter_secrets_manager import get_twitter_credentials

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/finbert_scoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinBERTSentimentScorer:
    """FinBERT sentiment scorer with ONNX Runtime optimization"""
    
    def __init__(self, 
                 model_path: str = "models/finbert_onnx/finbert.onnx",
                 tokenizer_name: str = "ProsusAI/finbert",
                 s3_bucket: str = "hpo-bucket-773934887314",
                 s3_input_prefix: str = "twitter-sentiment/raw-tweets/",
                 s3_output_prefix: str = "twitter-sentiment/scored-tweets/",
                 region_name: str = "us-east-1",
                 batch_size: int = 32,
                 max_length: int = 512):
        """
        Initialize FinBERT sentiment scorer
        
        Args:
            model_path: Path to ONNX model file
            tokenizer_name: HuggingFace tokenizer name
            s3_bucket: S3 bucket for input/output
            s3_input_prefix: S3 prefix for raw tweets
            s3_output_prefix: S3 prefix for scored tweets
            region_name: AWS region
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.s3_bucket = s3_bucket
        self.s3_input_prefix = s3_input_prefix
        self.s3_output_prefix = s3_output_prefix
        self.region_name = region_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.ort_session = None
        self.tokenizer = None
        
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        logger.info(f"Initialized FinBERT scorer with batch size {batch_size}")
    
    async def initialize_model(self):
        """Initialize ONNX model and tokenizer"""
        try:
            if not os.path.exists(self.model_path):
                await self.download_finbert_model()
            
            logger.info(f"Loading ONNX model from {self.model_path}")
            
            providers = ['CPUExecutionProvider']
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = os.cpu_count()
            session_options.inter_op_num_threads = 1
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.ort_session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            logger.info(f"Loading tokenizer: {self.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            
            logger.info("FinBERT model and tokenizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FinBERT model: {e}")
            raise
    
    async def download_finbert_model(self):
        """Download and convert FinBERT model to ONNX format"""
        try:
            logger.info("Downloading FinBERT model and converting to ONNX...")
            
            from transformers import AutoModelForSequenceClassification
            import torch.onnx
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model = AutoModelForSequenceClassification.from_pretrained(self.tokenizer_name)
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            
            model.eval()
            
            dummy_text = "The stock market is performing well today."
            inputs = tokenizer(
                dummy_text,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            logger.info("Converting model to ONNX format...")
            torch.onnx.export(
                model,
                (inputs['input_ids'], inputs['attention_mask']),
                self.model_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size'}
                }
            )
            
            logger.info(f"FinBERT model converted and saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to download/convert FinBERT model: {e}")
            raise
    
    def preprocess_tweets(self, tweets: List[str]) -> Dict[str, np.ndarray]:
        """
        Preprocess tweets for FinBERT inference
        
        Args:
            tweets: List of tweet texts
            
        Returns:
            Dict with input_ids and attention_mask arrays
        """
        try:
            if not self.tokenizer:
                raise Exception("Tokenizer not initialized. Call initialize_model() first.")
                
            encoded = self.tokenizer(
                tweets,
                return_tensors="np",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            return {
                'input_ids': encoded['input_ids'].astype(np.int64),
                'attention_mask': encoded['attention_mask'].astype(np.int64)
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing tweets: {e}")
            raise
    
    def run_inference(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run FinBERT inference on preprocessed inputs
        
        Args:
            inputs: Dict with input_ids and attention_mask
            
        Returns:
            Logits array from model
        """
        try:
            if not self.ort_session:
                raise Exception("ONNX session not initialized. Call initialize_model() first.")
                
            outputs = self.ort_session.run(
                None,
                {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask']
                }
            )
            
            return outputs[0]  # logits
            
        except Exception as e:
            logger.error(f"Error running FinBERT inference: {e}")
            raise
    
    def postprocess_logits(self, logits: np.ndarray) -> List[Dict[str, Any]]:
        """
        Convert logits to sentiment probabilities
        
        Args:
            logits: Raw logits from model
            
        Returns:
            List of sentiment probability dictionaries
        """
        try:
            probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            
            results = []
            for prob_row in probabilities:
                sentiment_scores = {
                    label: float(prob_row[i])
                    for i, label in enumerate(self.sentiment_labels)
                }
                
                predicted_idx = np.argmax(prob_row)
                sentiment_scores['predicted_sentiment'] = self.sentiment_labels[predicted_idx]
                sentiment_scores['confidence'] = float(prob_row[predicted_idx])
                
                results.append(sentiment_scores)
            
            return results
            
        except Exception as e:
            logger.error(f"Error postprocessing logits: {e}")
            raise
    
    async def score_tweet_batch(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score a batch of tweets with sentiment analysis
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            List of tweets with sentiment scores added
        """
        try:
            if not tweets:
                return []
            
            tweet_texts = [tweet.get('text', '') for tweet in tweets]
            
            scored_tweets = []
            for i in range(0, len(tweet_texts), self.batch_size):
                batch_texts = tweet_texts[i:i + self.batch_size]
                batch_tweets = tweets[i:i + self.batch_size]
                
                inputs = self.preprocess_tweets(batch_texts)
                
                logits = self.run_inference(inputs)
                
                sentiment_scores = self.postprocess_logits(logits)
                
                for tweet, scores in zip(batch_tweets, sentiment_scores):
                    scored_tweet = tweet.copy()
                    scored_tweet.update({
                        'sentiment_negative': scores['negative'],
                        'sentiment_neutral': scores['neutral'],
                        'sentiment_positive': scores['positive'],
                        'sentiment_predicted': scores['predicted_sentiment'],
                        'sentiment_confidence': scores['confidence'],
                        'sentiment_model': 'finbert',
                        'sentiment_scored_at': datetime.now(timezone.utc).isoformat()
                    })
                    scored_tweets.append(scored_tweet)
                
                logger.info(f"Scored batch of {len(batch_tweets)} tweets")
            
            return scored_tweets
            
        except Exception as e:
            logger.error(f"Error scoring tweet batch: {e}")
            raise
    
    async def load_tweets_from_s3(self, s3_key: str) -> List[Dict[str, Any]]:
        """
        Load tweets from S3 JSON Lines file
        
        Args:
            s3_key: S3 key for tweet file
            
        Returns:
            List of tweet dictionaries
        """
        try:
            logger.info(f"Loading tweets from s3://{self.s3_bucket}/{s3_key}")
            
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            
            tweets = []
            for line in content.strip().split('\n'):
                if line.strip():
                    tweet = json.loads(line)
                    tweets.append(tweet)
            
            logger.info(f"Loaded {len(tweets)} tweets from S3")
            return tweets
            
        except ClientError as e:
            logger.error(f"S3 error loading tweets: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading tweets from S3: {e}")
            raise
    
    async def save_scored_tweets_to_s3(self, scored_tweets: List[Dict[str, Any]], output_key: str):
        """
        Save scored tweets to S3 in JSON Lines format
        
        Args:
            scored_tweets: List of scored tweet dictionaries
            output_key: S3 key for output file
        """
        try:
            if not scored_tweets:
                return
            
            json_lines = '\n'.join([json.dumps(tweet) for tweet in scored_tweets])
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=output_key,
                Body=json_lines.encode('utf-8'),
                ContentType='application/json'
            )
            
            logger.info(f"Saved {len(scored_tweets)} scored tweets to s3://{self.s3_bucket}/{output_key}")
            
        except ClientError as e:
            logger.error(f"S3 error saving scored tweets: {e}")
            raise
        except Exception as e:
            logger.error(f"Error saving scored tweets to S3: {e}")
            raise
    
    async def process_s3_file(self, input_key: str) -> Optional[str]:
        """
        Process a single S3 file with tweet sentiment scoring
        
        Args:
            input_key: S3 key for input tweet file
            
        Returns:
            S3 key for output scored tweet file
        """
        try:
            tweets = await self.load_tweets_from_s3(input_key)
            
            if not tweets:
                logger.warning(f"No tweets found in {input_key}")
                return None
            
            scored_tweets = await self.score_tweet_batch(tweets)
            
            output_key = input_key.replace(self.s3_input_prefix, self.s3_output_prefix)
            output_key = output_key.replace('.json', '_scored.json')
            
            await self.save_scored_tweets_to_s3(scored_tweets, output_key)
            
            return output_key
            
        except Exception as e:
            logger.error(f"Error processing S3 file {input_key}: {e}")
            raise
    
    async def list_unprocessed_files(self, symbol: Optional[str] = None, 
                                   date_prefix: Optional[str] = None) -> List[str]:
        """
        List unprocessed tweet files in S3
        
        Args:
            symbol: Filter by stock symbol
            date_prefix: Filter by date prefix (YYYY/MM/DD)
            
        Returns:
            List of S3 keys for unprocessed files
        """
        try:
            list_prefix = self.s3_input_prefix
            if symbol:
                list_prefix += f"{symbol}/"
            if date_prefix:
                list_prefix += date_prefix
            
            logger.info(f"Listing files with prefix: {list_prefix}")
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=list_prefix)
            
            input_files = []
            processed_files = set()
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('.json'):
                            if key.startswith(self.s3_input_prefix):
                                input_files.append(key)
                            elif key.startswith(self.s3_output_prefix) and '_scored.json' in key:
                                original_key = key.replace(self.s3_output_prefix, self.s3_input_prefix)
                                original_key = original_key.replace('_scored.json', '.json')
                                processed_files.add(original_key)
            
            unprocessed_files = [f for f in input_files if f not in processed_files]
            
            logger.info(f"Found {len(input_files)} input files, {len(processed_files)} processed, {len(unprocessed_files)} unprocessed")
            
            return unprocessed_files
            
        except Exception as e:
            logger.error(f"Error listing unprocessed files: {e}")
            raise
    
    async def process_all_unprocessed(self, symbol: Optional[str] = None,
                                    date_prefix: Optional[str] = None,
                                    max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all unprocessed tweet files
        
        Args:
            symbol: Filter by stock symbol
            date_prefix: Filter by date prefix
            max_files: Maximum number of files to process
            
        Returns:
            Processing summary
        """
        try:
            unprocessed_files = await self.list_unprocessed_files(symbol, date_prefix)
            
            if max_files:
                unprocessed_files = unprocessed_files[:max_files]
            
            if not unprocessed_files:
                logger.info("No unprocessed files found")
                return {
                    'status': 'completed',
                    'files_processed': 0,
                    'total_tweets_scored': 0
                }
            
            logger.info(f"Processing {len(unprocessed_files)} files...")
            
            processed_files = []
            total_tweets = 0
            
            for input_key in unprocessed_files:
                try:
                    output_key = await self.process_s3_file(input_key)
                    if output_key:
                        processed_files.append({
                            'input_key': input_key,
                            'output_key': output_key
                        })
                        
                        response = self.s3_client.head_object(Bucket=self.s3_bucket, Key=input_key)
                        file_size = response['ContentLength']
                        estimated_tweets = max(1, file_size // 500)  # Rough estimate
                        total_tweets += estimated_tweets
                        
                except Exception as e:
                    logger.error(f"Failed to process {input_key}: {e}")
                    continue
            
            summary = {
                'status': 'completed',
                'files_processed': len(processed_files),
                'files_failed': len(unprocessed_files) - len(processed_files),
                'total_tweets_scored': total_tweets,
                'processed_files': processed_files
            }
            
            logger.info(f"Processing complete: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error processing all unprocessed files: {e}")
            raise


async def main():
    """Main function for FinBERT sentiment scoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FinBERT Sentiment Scoring for Twitter Data')
    parser.add_argument('--model-path', default='models/finbert_onnx/finbert.onnx',
                        help='Path to ONNX model file')
    parser.add_argument('--s3-bucket', default='hpo-bucket-773934887314',
                        help='S3 bucket for input/output')
    parser.add_argument('--s3-input-prefix', default='twitter-sentiment/raw-tweets/',
                        help='S3 prefix for raw tweets')
    parser.add_argument('--s3-output-prefix', default='twitter-sentiment/scored-tweets/',
                        help='S3 prefix for scored tweets')
    parser.add_argument('--symbol', help='Filter by stock symbol')
    parser.add_argument('--date-prefix', help='Filter by date prefix (YYYY/MM/DD)')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--region', default='us-east-1',
                        help='AWS region')
    parser.add_argument('--dry-run', action='store_true',
                        help='List files without processing')
    
    args = parser.parse_args()
    
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🚀 Starting FinBERT Sentiment Scoring")
    logger.info(f"   Model Path: {args.model_path}")
    logger.info(f"   S3 Bucket: {args.s3_bucket}")
    logger.info(f"   Input Prefix: {args.s3_input_prefix}")
    logger.info(f"   Output Prefix: {args.s3_output_prefix}")
    logger.info(f"   Batch Size: {args.batch_size}")
    
    try:
        scorer = FinBERTSentimentScorer(
            model_path=args.model_path,
            s3_bucket=args.s3_bucket,
            s3_input_prefix=args.s3_input_prefix,
            s3_output_prefix=args.s3_output_prefix,
            region_name=args.region,
            batch_size=args.batch_size
        )
        
        await scorer.initialize_model()
        
        if args.dry_run:
            logger.info("🧪 DRY RUN MODE - Listing unprocessed files")
            unprocessed_files = await scorer.list_unprocessed_files(args.symbol, args.date_prefix)
            
            logger.info(f"Found {len(unprocessed_files)} unprocessed files:")
            for file_key in unprocessed_files[:10]:  # Show first 10
                logger.info(f"  {file_key}")
            
            if len(unprocessed_files) > 10:
                logger.info(f"  ... and {len(unprocessed_files) - 10} more files")
        else:
            summary = await scorer.process_all_unprocessed(
                symbol=args.symbol,
                date_prefix=args.date_prefix,
                max_files=args.max_files
            )
            
            logger.info("✅ FinBERT sentiment scoring completed")
            logger.info(f"   Files processed: {summary['files_processed']}")
            logger.info(f"   Tweets scored: {summary['total_tweets_scored']}")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
