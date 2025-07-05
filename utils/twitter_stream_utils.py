#!/usr/bin/env python3
"""
Twitter Stream Utilities
Helper functions for Twitter streaming and data processing
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class TwitterTextProcessor:
    """Utilities for processing Twitter text content"""
    
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.cashtag_pattern = re.compile(r'\$[A-Z]{1,5}')
        self.whitespace_pattern = re.compile(r'\s+')
        
        self.bullish_indicators = {
            'buy', 'bull', 'bullish', 'long', 'calls', 'moon', 'rocket', 'pump',
            'breakout', 'rally', 'surge', 'spike', 'gap up', 'strong', 'beat',
            'upgrade', 'outperform', 'overweight', 'positive', 'good news'
        }
        
        self.bearish_indicators = {
            'sell', 'bear', 'bearish', 'short', 'puts', 'crash', 'dump', 'drop',
            'breakdown', 'decline', 'fall', 'gap down', 'weak', 'miss',
            'downgrade', 'underperform', 'underweight', 'negative', 'bad news'
        }
    
    def clean_tweet_text(self, text: str, remove_urls: bool = True, 
                        remove_mentions: bool = False, remove_hashtags: bool = False) -> str:
        """
        Clean tweet text for sentiment analysis
        
        Args:
            text: Raw tweet text
            remove_urls: Remove URLs from text
            remove_mentions: Remove @mentions from text
            remove_hashtags: Remove #hashtags from text
            
        Returns:
            Cleaned text
        """
        cleaned_text = text
        
        if remove_urls:
            cleaned_text = self.url_pattern.sub('', cleaned_text)
        
        if remove_mentions:
            cleaned_text = self.mention_pattern.sub('', cleaned_text)
        
        if remove_hashtags:
            cleaned_text = self.hashtag_pattern.sub('', cleaned_text)
        
        cleaned_text = self.whitespace_pattern.sub(' ', cleaned_text).strip()
        
        return cleaned_text
    
    def extract_cashtags(self, text: str) -> List[str]:
        """
        Extract cashtags ($SYMBOL) from tweet text
        
        Args:
            text: Tweet text
            
        Returns:
            List of cashtags without $ prefix
        """
        cashtags = self.cashtag_pattern.findall(text.upper())
        return [tag[1:] for tag in cashtags]  # Remove $ prefix
    
    def extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags from tweet text
        
        Args:
            text: Tweet text
            
        Returns:
            List of hashtags without # prefix
        """
        hashtags = self.hashtag_pattern.findall(text.lower())
        return [tag[1:] for tag in hashtags]  # Remove # prefix
    
    def get_sentiment_indicators(self, text: str) -> Dict[str, int]:
        """
        Count bullish and bearish sentiment indicators in text
        
        Args:
            text: Tweet text
            
        Returns:
            Dict with bullish_count and bearish_count
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for indicator in self.bullish_indicators 
                           if indicator in text_lower)
        bearish_count = sum(1 for indicator in self.bearish_indicators 
                           if indicator in text_lower)
        
        return {
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'net_sentiment': bullish_count - bearish_count
        }


class TwitterMetricsCalculator:
    """Calculate engagement and quality metrics for tweets"""
    
    @staticmethod
    def calculate_engagement_score(public_metrics: Dict[str, int]) -> float:
        """
        Calculate engagement score based on public metrics
        
        Args:
            public_metrics: Dict with retweet_count, like_count, reply_count, quote_count
            
        Returns:
            Engagement score (0.0 to 1.0)
        """
        retweets = public_metrics.get('retweet_count', 0)
        likes = public_metrics.get('like_count', 0)
        replies = public_metrics.get('reply_count', 0)
        quotes = public_metrics.get('quote_count', 0)
        
        weighted_score = (retweets * 3) + (quotes * 2) + (replies * 1.5) + (likes * 1)
        
        import math
        if weighted_score == 0:
            return 0.0
        
        normalized_score = min(1.0, math.log10(weighted_score + 1) / math.log10(1001))
        return normalized_score
    
    @staticmethod
    def calculate_virality_score(public_metrics: Dict[str, int]) -> float:
        """
        Calculate virality score based on retweet/like ratio
        
        Args:
            public_metrics: Dict with retweet_count, like_count
            
        Returns:
            Virality score (0.0 to 1.0)
        """
        retweets = public_metrics.get('retweet_count', 0)
        likes = public_metrics.get('like_count', 0)
        
        if likes == 0:
            return 0.0
        
        ratio = retweets / likes
        
        normalized_ratio = min(1.0, ratio / 0.3)
        return normalized_ratio
    
    @staticmethod
    def is_high_quality_tweet(text: str, public_metrics: Dict[str, int]) -> bool:
        """
        Determine if tweet is high quality for sentiment analysis
        
        Args:
            text: Tweet text
            public_metrics: Tweet engagement metrics
            
        Returns:
            True if tweet meets quality criteria
        """
        if len(text) < 20:  # Too short
            return False
        
        if len(text) > 280:  # Likely truncated or spam
            return False
        
        total_engagement = sum(public_metrics.values())
        if total_engagement < 2:  # Very low engagement
            return False
        
        text_lower = text.lower()
        
        spam_indicators = ['follow me', 'dm me', 'check my bio', 'link in bio', 
                          'subscribe', 'like and retweet', 'giveaway']
        if any(indicator in text_lower for indicator in spam_indicators):
            return False
        
        words = text_lower.split()
        if len(set(words)) < len(words) * 0.5:  # More than 50% repeated words
            return False
        
        return True


class TwitterSymbolMatcher:
    """Match tweets to stock symbols with confidence scoring"""
    
    def __init__(self, target_symbols: List[str]):
        self.target_symbols = [symbol.upper() for symbol in target_symbols]
        
        self.company_mappings = {
            'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'ios'],
            'GOOGL': ['google', 'alphabet', 'youtube', 'android', 'gmail'],
            'TSLA': ['tesla', 'elon musk', 'model 3', 'model y', 'cybertruck'],
            'META': ['meta', 'facebook', 'instagram', 'whatsapp', 'metaverse'],
            'MSFT': ['microsoft', 'windows', 'office', 'azure', 'xbox'],
            'AMZN': ['amazon', 'aws', 'prime', 'alexa', 'bezos'],
            'NVDA': ['nvidia', 'gpu', 'ai chip', 'graphics card'],
            'JPM': ['jpmorgan', 'chase', 'jamie dimon'],
            'JNJ': ['johnson', 'johnson & johnson', 'j&j'],
            'V': ['visa', 'payment'],
            'MA': ['mastercard', 'payment']
        }
    
    def match_symbol_to_tweet(self, text: str, matching_rules: Optional[List[str]] = None) -> Tuple[Optional[str], float]:
        """
        Match tweet to most relevant stock symbol with confidence score
        
        Args:
            text: Tweet text
            matching_rules: List of matching rule tags from Twitter API
            
        Returns:
            Tuple of (symbol, confidence_score) or (None, 0.0)
        """
        text_lower = text.lower()
        symbol_scores = {}
        
        cashtags = re.findall(r'\$([A-Z]{1,5})', text.upper())
        for cashtag in cashtags:
            if cashtag in self.target_symbols:
                symbol_scores[cashtag] = symbol_scores.get(cashtag, 0) + 1.0
        
        if matching_rules:
            for rule_tag in matching_rules:
                if rule_tag.startswith('cashtag_'):
                    symbol = rule_tag.replace('cashtag_', '').upper()
                    if symbol in self.target_symbols:
                        symbol_scores[symbol] = symbol_scores.get(symbol, 0) + 0.9
                elif rule_tag.startswith('company_'):
                    symbol = rule_tag.replace('company_', '').upper()
                    if symbol in self.target_symbols:
                        symbol_scores[symbol] = symbol_scores.get(symbol, 0) + 0.7
        
        for symbol, keywords in self.company_mappings.items():
            if symbol in self.target_symbols:
                for keyword in keywords:
                    if keyword in text_lower:
                        symbol_scores[symbol] = symbol_scores.get(symbol, 0) + 0.5
        
        if symbol_scores:
            best_symbol = max(symbol_scores.items(), key=lambda x: x[1])
            return best_symbol[0], min(1.0, best_symbol[1])  # Cap confidence at 1.0
        
        return None, 0.0


def create_s3_key(symbol: str, timestamp: datetime, prefix: str = "twitter-sentiment/raw-tweets/") -> str:
    """
    Create S3 key with proper partitioning
    
    Args:
        symbol: Stock symbol
        timestamp: Tweet timestamp
        prefix: S3 prefix
        
    Returns:
        S3 key string
    """
    date_str = timestamp.strftime('%Y/%m/%d')
    hour_str = timestamp.strftime('%H')
    minute_str = timestamp.strftime('%M')
    
    return f"{prefix}symbol={symbol}/year={timestamp.year}/month={timestamp.month:02d}/day={timestamp.day:02d}/hour={hour_str}/tweets_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"


def validate_tweet_data(tweet_data: Dict) -> bool:
    """
    Validate tweet data structure
    
    Args:
        tweet_data: Tweet data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['tweet_id', 'text', 'created_at', 'primary_symbol']
    
    for field in required_fields:
        if field not in tweet_data or not tweet_data[field]:
            logger.warning(f"Missing required field: {field}")
            return False
    
    try:
        datetime.fromisoformat(tweet_data['created_at'].replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        logger.warning(f"Invalid timestamp format: {tweet_data.get('created_at')}")
        return False
    
    return True


def batch_tweets_by_symbol_and_time(tweets: List[Dict], batch_size: int = 100) -> Dict[str, List[Dict]]:
    """
    Batch tweets by symbol and time for efficient processing
    
    Args:
        tweets: List of tweet dictionaries
        batch_size: Maximum tweets per batch
        
    Returns:
        Dict mapping batch keys to tweet lists
    """
    batches = {}
    
    for tweet in tweets:
        symbol = tweet.get('primary_symbol', 'UNKNOWN')
        created_at = datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00'))
        
        batch_key = f"{symbol}_{created_at.strftime('%Y%m%d_%H')}"
        
        if batch_key not in batches:
            batches[batch_key] = []
        
        batches[batch_key].append(tweet)
        
        if len(batches[batch_key]) >= batch_size:
            suffix = 1
            while f"{batch_key}_{suffix}" in batches:
                suffix += 1
            
            new_batch_key = f"{batch_key}_{suffix}"
            batches[new_batch_key] = batches[batch_key][batch_size:]
            batches[batch_key] = batches[batch_key][:batch_size]
    
    return batches
