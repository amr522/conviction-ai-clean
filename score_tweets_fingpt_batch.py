#!/usr/bin/env python3
"""
FinGPT GPU Sentiment Scoring with Spot Instance Management
Optional GPU-accelerated sentiment scoring using FinGPT on AWS Spot instances
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import boto3
from botocore.exceptions import ClientError
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from utils.spot_instance_manager import SpotInstanceManager
from score_tweets_finbert import FinBERTSentimentScorer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fingpt_scoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinGPTSentimentScorer:
    """FinGPT sentiment scorer with GPU acceleration and spot instance management"""
    
    def __init__(self,
                 model_name: str = "FinGPT/fingpt-sentiment_llama2-7b_lora",
                 s3_bucket: str = "hpo-bucket-773934887314",
                 s3_input_prefix: str = "twitter-sentiment/raw-tweets/",
                 s3_output_prefix: str = "twitter-sentiment/scored-tweets-gpu/",
                 region_name: str = "us-east-1",
                 batch_size: int = 8,
                 max_length: int = 512,
                 spot_instance_type: str = "g4dn.xlarge",
                 max_spot_price: float = 0.50,
                 fallback_to_finbert: bool = True):
        """
        Initialize FinGPT sentiment scorer
        
        Args:
            model_name: HuggingFace model name for FinGPT
            s3_bucket: S3 bucket for input/output
            s3_input_prefix: S3 prefix for raw tweets
            s3_output_prefix: S3 prefix for scored tweets
            region_name: AWS region
            batch_size: Batch size for GPU inference
            max_length: Maximum sequence length
            spot_instance_type: EC2 spot instance type for GPU
            max_spot_price: Maximum spot price per hour
            fallback_to_finbert: Whether to fallback to FinBERT if GPU unavailable
        """
        self.model_name = model_name
        self.s3_bucket = s3_bucket
        self.s3_input_prefix = s3_input_prefix
        self.s3_output_prefix = s3_output_prefix
        self.region_name = region_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.spot_instance_type = spot_instance_type
        self.max_spot_price = max_spot_price
        self.fallback_to_finbert = fallback_to_finbert
        
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.spot_manager = SpotInstanceManager(
            instance_type=spot_instance_type,
            max_price=max_spot_price,
            region_name=region_name
        )
        
        self.tokenizer = None
        self.model = None
        self.device = None
        self.gpu_available = False
        
        self.finbert_scorer = None
        
        self.sentiment_labels = ["negative", "neutral", "positive"]
        
        logger.info(f"Initialized FinGPT scorer with model: {model_name}")
        logger.info(f"GPU instance type: {spot_instance_type}, max price: ${max_spot_price}/hr")
        logger.info(f"Fallback to FinBERT: {fallback_to_finbert}")
    
    async def initialize_model(self) -> bool:
        """
        Initialize FinGPT model with GPU support and spot instance management
        
        Returns:
            True if GPU model loaded successfully, False if fallback needed
        """
        try:
            self.gpu_available = torch.cuda.is_available()
            
            if not self.gpu_available:
                logger.warning("CUDA not available on this instance")
                if self.fallback_to_finbert:
                    return await self._initialize_fallback()
                else:
                    raise Exception("GPU required but not available")
            
            if not await self._check_gpu_resources():
                logger.info("Insufficient GPU resources, launching spot instance...")
                success = await self.spot_manager.launch_spot_instance()
                if not success:
                    logger.warning("Failed to launch spot instance")
                    if self.fallback_to_finbert:
                        return await self._initialize_fallback()
                    else:
                        raise Exception("Failed to launch GPU spot instance")
            
            logger.info(f"Loading FinGPT model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True  # Use 8-bit quantization for memory efficiency
            )
            
            self.device = next(self.model.parameters()).device
            logger.info(f"FinGPT model loaded on device: {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing FinGPT model: {e}")
            if self.fallback_to_finbert:
                return await self._initialize_fallback()
            else:
                raise
    
    async def _check_gpu_resources(self) -> bool:
        """
        Check if current instance has sufficient GPU resources
        
        Returns:
            True if GPU resources are sufficient
        """
        try:
            if not torch.cuda.is_available():
                return False
            
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                return False
            
            for i in range(gpu_count):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                logger.info(f"GPU {i}: {gpu_memory_gb:.1f}GB memory")
                
                if gpu_memory_gb >= 8.0:
                    return True
            
            logger.warning("Insufficient GPU memory for FinGPT (requires 8GB+)")
            return False
            
        except Exception as e:
            logger.error(f"Error checking GPU resources: {e}")
            return False
    
    async def _initialize_fallback(self) -> bool:
        """
        Initialize FinBERT fallback scorer
        
        Returns:
            False to indicate fallback mode
        """
        try:
            logger.info("Initializing FinBERT fallback scorer...")
            
            self.finbert_scorer = FinBERTSentimentScorer(
                s3_bucket=self.s3_bucket,
                s3_input_prefix=self.s3_input_prefix,
                s3_output_prefix=self.s3_output_prefix.replace('-gpu', '-fallback'),
                region_name=self.region_name,
                batch_size=self.batch_size * 2  # CPU can handle larger batches
            )
            
            await self.finbert_scorer.initialize_model()
            logger.info("FinBERT fallback scorer initialized successfully")
            
            return False  # Indicate fallback mode
            
        except Exception as e:
            logger.error(f"Error initializing FinBERT fallback: {e}")
            raise
    
    def create_fingpt_prompt(self, tweet_text: str) -> str:
        """
        Create FinGPT prompt for sentiment analysis
        
        Args:
            tweet_text: Tweet text to analyze
            
        Returns:
            Formatted prompt for FinGPT
        """
        prompt = f"""Analyze the sentiment of this financial tweet and classify it as positive, negative, or neutral.

Tweet: {tweet_text}

Sentiment:"""
        return prompt
    
    async def score_tweet_batch_gpu(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score a batch of tweets using FinGPT on GPU
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            List of tweets with sentiment scores added
        """
        try:
            if not tweets:
                return []
            
            if self.finbert_scorer:
                logger.info("Using FinBERT fallback for scoring")
                return await self.finbert_scorer.score_tweet_batch(tweets)
            
            if not self.model or not self.tokenizer:
                raise Exception("FinGPT model not initialized")
            
            scored_tweets = []
            
            for i in range(0, len(tweets), self.batch_size):
                batch_tweets = tweets[i:i + self.batch_size]
                
                prompts = [self.create_fingpt_prompt(tweet.get('text', '')) for tweet in batch_tweets]
                
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                responses = self.tokenizer.batch_decode(
                    outputs[:, inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                for tweet, response in zip(batch_tweets, responses):
                    sentiment_scores = self._parse_fingpt_response(response)
                    
                    scored_tweet = tweet.copy()
                    scored_tweet.update({
                        'sentiment_negative': sentiment_scores['negative'],
                        'sentiment_neutral': sentiment_scores['neutral'],
                        'sentiment_positive': sentiment_scores['positive'],
                        'sentiment_predicted': sentiment_scores['predicted_sentiment'],
                        'sentiment_confidence': sentiment_scores['confidence'],
                        'sentiment_model': 'fingpt-gpu',
                        'sentiment_scored_at': datetime.now(timezone.utc).isoformat(),
                        'fingpt_response': response.strip()
                    })
                    scored_tweets.append(scored_tweet)
                
                logger.info(f"Scored batch of {len(batch_tweets)} tweets with FinGPT")
            
            return scored_tweets
            
        except Exception as e:
            logger.error(f"Error scoring tweet batch with FinGPT: {e}")
            
            if self.fallback_to_finbert and not self.finbert_scorer:
                logger.info("Attempting FinBERT fallback after FinGPT error")
                await self._initialize_fallback()
                if self.finbert_scorer:
                    return await self.finbert_scorer.score_tweet_batch(tweets)
            
            raise
    
    def _parse_fingpt_response(self, response: str) -> Dict[str, Any]:
        """
        Parse FinGPT response to extract sentiment scores
        
        Args:
            response: Raw response from FinGPT
            
        Returns:
            Dict with sentiment scores and prediction
        """
        try:
            response_lower = response.lower().strip()
            
            if 'positive' in response_lower:
                predicted = 'positive'
                scores = {'positive': 0.8, 'neutral': 0.15, 'negative': 0.05}
            elif 'negative' in response_lower:
                predicted = 'negative'
                scores = {'positive': 0.05, 'neutral': 0.15, 'negative': 0.8}
            else:
                predicted = 'neutral'
                scores = {'positive': 0.25, 'neutral': 0.5, 'negative': 0.25}
            
            return {
                'negative': scores['negative'],
                'neutral': scores['neutral'],
                'positive': scores['positive'],
                'predicted_sentiment': predicted,
                'confidence': scores[predicted]
            }
            
        except Exception as e:
            logger.warning(f"Error parsing FinGPT response '{response}': {e}")
            return {
                'negative': 0.33,
                'neutral': 0.34,
                'positive': 0.33,
                'predicted_sentiment': 'neutral',
                'confidence': 0.34
            }
    
    async def load_tweets_from_s3(self, s3_key: str) -> List[Dict[str, Any]]:
        """Load tweets from S3 JSON Lines file"""
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
        """Save scored tweets to S3 in JSON Lines format"""
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
        Process a single S3 file with FinGPT sentiment scoring
        
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
            
            scored_tweets = await self.score_tweet_batch_gpu(tweets)
            
            output_key = input_key.replace(self.s3_input_prefix, self.s3_output_prefix)
            output_key = output_key.replace('.json', '_scored_gpu.json')
            
            await self.save_scored_tweets_to_s3(scored_tweets, output_key)
            
            return output_key
            
        except Exception as e:
            logger.error(f"Error processing S3 file {input_key}: {e}")
            raise
    
    async def process_batch_with_queue(self, 
                                     file_keys: List[str],
                                     max_concurrent: int = 2) -> Dict[str, Any]:
        """
        Process multiple files with queue management for GPU efficiency
        
        Args:
            file_keys: List of S3 keys to process
            max_concurrent: Maximum concurrent file processing
            
        Returns:
            Processing summary
        """
        try:
            logger.info(f"Processing {len(file_keys)} files with max {max_concurrent} concurrent")
            
            semaphore = asyncio.Semaphore(max_concurrent)
            processed_files = []
            failed_files = []
            
            async def process_file_with_semaphore(file_key: str):
                async with semaphore:
                    try:
                        output_key = await self.process_s3_file(file_key)
                        if output_key:
                            processed_files.append({
                                'input_key': file_key,
                                'output_key': output_key
                            })
                        return True
                    except Exception as e:
                        logger.error(f"Failed to process {file_key}: {e}")
                        failed_files.append(file_key)
                        return False
            
            tasks = [process_file_with_semaphore(key) for key in file_keys]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            summary = {
                'status': 'completed',
                'files_processed': len(processed_files),
                'files_failed': len(failed_files),
                'processed_files': processed_files,
                'failed_files': failed_files,
                'model_used': 'fingpt-gpu' if not self.finbert_scorer else 'finbert-fallback'
            }
            
            logger.info(f"Batch processing complete: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    async def cleanup_resources(self):
        """Clean up GPU resources and spot instances"""
        try:
            if self.model:
                del self.model
                self.model = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            await self.spot_manager.cleanup_spot_instance()
            
            logger.info("GPU resources and spot instances cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")


async def main():
    """Main function for FinGPT GPU sentiment scoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FinGPT GPU Sentiment Scoring for Twitter Data')
    parser.add_argument('--model-name', default='FinGPT/fingpt-sentiment_llama2-7b_lora',
                        help='HuggingFace model name for FinGPT')
    parser.add_argument('--s3-bucket', default='hpo-bucket-773934887314',
                        help='S3 bucket for input/output')
    parser.add_argument('--s3-input-prefix', default='twitter-sentiment/raw-tweets/',
                        help='S3 prefix for raw tweets')
    parser.add_argument('--s3-output-prefix', default='twitter-sentiment/scored-tweets-gpu/',
                        help='S3 prefix for scored tweets')
    parser.add_argument('--file-keys', nargs='+', help='Specific S3 file keys to process')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for GPU inference')
    parser.add_argument('--max-concurrent', type=int, default=2,
                        help='Maximum concurrent file processing')
    parser.add_argument('--spot-instance-type', default='g4dn.xlarge',
                        help='EC2 spot instance type for GPU')
    parser.add_argument('--max-spot-price', type=float, default=0.50,
                        help='Maximum spot price per hour')
    parser.add_argument('--region', default='us-east-1',
                        help='AWS region')
    parser.add_argument('--no-fallback', action='store_true',
                        help='Disable FinBERT fallback')
    parser.add_argument('--dry-run', action='store_true',
                        help='Initialize model without processing')
    
    args = parser.parse_args()
    
    os.makedirs('logs', exist_ok=True)
    
    logger.info("🚀 Starting FinGPT GPU Sentiment Scoring")
    logger.info(f"   Model: {args.model_name}")
    logger.info(f"   S3 Bucket: {args.s3_bucket}")
    logger.info(f"   Spot Instance: {args.spot_instance_type} (max ${args.max_spot_price}/hr)")
    logger.info(f"   Batch Size: {args.batch_size}")
    logger.info(f"   Fallback Enabled: {not args.no_fallback}")
    
    scorer = None
    try:
        scorer = FinGPTSentimentScorer(
            model_name=args.model_name,
            s3_bucket=args.s3_bucket,
            s3_input_prefix=args.s3_input_prefix,
            s3_output_prefix=args.s3_output_prefix,
            region_name=args.region,
            batch_size=args.batch_size,
            spot_instance_type=args.spot_instance_type,
            max_spot_price=args.max_spot_price,
            fallback_to_finbert=not args.no_fallback
        )
        
        gpu_success = await scorer.initialize_model()
        
        if gpu_success:
            logger.info("✅ FinGPT GPU model initialized successfully")
        else:
            logger.info("⚠️  Using FinBERT fallback mode")
        
        if args.dry_run:
            logger.info("🧪 DRY RUN MODE - Model initialized, exiting")
            return
        
        if not args.file_keys:
            logger.error("No file keys specified for processing")
            return
        
        summary = await scorer.process_batch_with_queue(
            file_keys=args.file_keys,
            max_concurrent=args.max_concurrent
        )
        
        logger.info("✅ FinGPT sentiment scoring completed")
        logger.info(f"   Model used: {summary['model_used']}")
        logger.info(f"   Files processed: {summary['files_processed']}")
        logger.info(f"   Files failed: {summary['files_failed']}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if scorer:
            await scorer.cleanup_resources()


if __name__ == "__main__":
    asyncio.run(main())
