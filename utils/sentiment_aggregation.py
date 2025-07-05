#!/usr/bin/env python3
"""
Sentiment Aggregation Utilities
Multi-timeframe sentiment aggregation for Twitter data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SentimentAggregator:
    """Aggregates Twitter sentiment data across multiple timeframes"""
    
    def __init__(self):
        """Initialize sentiment aggregator"""
        self.sentiment_columns = [
            'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
            'sentiment_confidence'
        ]
        
    def _validate_sentiment_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that sentiment data has required columns
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            True if valid, False otherwise
        """
        required_cols = ['timestamp'] + self.sentiment_columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing sentiment columns: {missing_cols}")
            return False
        
        return True
    
    def _calculate_weighted_sentiment(self, df: pd.DataFrame) -> float:
        """
        Calculate weighted sentiment score
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Weighted sentiment score (-1 to 1)
        """
        if df.empty or not self._validate_sentiment_data(df):
            return 0.0
        
        try:
            positive_scores = df['sentiment_positive'] * df['sentiment_confidence']
            negative_scores = df['sentiment_negative'] * df['sentiment_confidence']
            
            total_positive = positive_scores.sum()
            total_negative = negative_scores.sum()
            total_weight = df['sentiment_confidence'].sum()
            
            if total_weight == 0:
                return 0.0
            
            net_sentiment = (total_positive - total_negative) / total_weight
            
            return float(np.clip(net_sentiment, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating weighted sentiment: {e}")
            return 0.0
    
    def _calculate_volume_weighted_sentiment(self, df: pd.DataFrame) -> float:
        """
        Calculate volume-weighted sentiment (considering engagement metrics)
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Volume-weighted sentiment score
        """
        if df.empty or not self._validate_sentiment_data(df):
            return 0.0
        
        try:
            if 'engagement_score' in df.columns:
                weights = df['engagement_score']
            elif 'retweet_count' in df.columns and 'like_count' in df.columns:
                weights = df['retweet_count'] + df['like_count'] + 1  # +1 to avoid zero weights
            else:
                return self._calculate_weighted_sentiment(df)
            
            positive_weighted = (df['sentiment_positive'] * weights).sum()
            negative_weighted = (df['sentiment_negative'] * weights).sum()
            total_weight = weights.sum()
            
            if total_weight == 0:
                return 0.0
            
            net_sentiment = (positive_weighted - negative_weighted) / total_weight
            return float(np.clip(net_sentiment, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating volume-weighted sentiment: {e}")
            return self._calculate_weighted_sentiment(df)
    
    def aggregate_5min(self, df: pd.DataFrame) -> float:
        """
        Aggregate sentiment over 5-minute intervals
        
        Args:
            df: DataFrame with sentiment data for a single day
            
        Returns:
            5-minute aggregated sentiment score
        """
        try:
            if df.empty:
                return 0.0
            
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df_resampled = df.set_index('timestamp').resample('5T').agg({
                'sentiment_positive': 'mean',
                'sentiment_negative': 'mean',
                'sentiment_neutral': 'mean',
                'sentiment_confidence': 'mean'
            }).dropna()
            
            if df_resampled.empty:
                return 0.0
            
            interval_sentiments = []
            for _, interval_data in df_resampled.iterrows():
                pos = interval_data['sentiment_positive']
                neg = interval_data['sentiment_negative']
                conf = interval_data['sentiment_confidence']
                
                if conf > 0:
                    sentiment = (pos - neg) * conf
                    interval_sentiments.append(sentiment)
            
            if not interval_sentiments:
                return 0.0
            
            return float(np.clip(np.mean(interval_sentiments), -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error in 5-minute aggregation: {e}")
            return 0.0
    
    def aggregate_10min(self, df: pd.DataFrame) -> float:
        """
        Aggregate sentiment over 10-minute intervals
        
        Args:
            df: DataFrame with sentiment data for a single day
            
        Returns:
            10-minute aggregated sentiment score
        """
        try:
            if df.empty:
                return 0.0
            
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df_resampled = df.set_index('timestamp').resample('10T').agg({
                'sentiment_positive': 'mean',
                'sentiment_negative': 'mean',
                'sentiment_neutral': 'mean',
                'sentiment_confidence': 'mean'
            }).dropna()
            
            if df_resampled.empty:
                return 0.0
            
            interval_sentiments = []
            for _, interval_data in df_resampled.iterrows():
                pos = interval_data['sentiment_positive']
                neg = interval_data['sentiment_negative']
                conf = interval_data['sentiment_confidence']
                
                if conf > 0:
                    sentiment = (pos - neg) * conf
                    interval_sentiments.append(sentiment)
            
            if not interval_sentiments:
                return 0.0
            
            return float(np.clip(np.mean(interval_sentiments), -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error in 10-minute aggregation: {e}")
            return 0.0
    
    def aggregate_60min(self, df: pd.DataFrame) -> float:
        """
        Aggregate sentiment over 60-minute intervals
        
        Args:
            df: DataFrame with sentiment data for a single day
            
        Returns:
            60-minute aggregated sentiment score
        """
        try:
            if df.empty:
                return 0.0
            
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df_resampled = df.set_index('timestamp').resample('60T').agg({
                'sentiment_positive': 'mean',
                'sentiment_negative': 'mean',
                'sentiment_neutral': 'mean',
                'sentiment_confidence': 'mean'
            }).dropna()
            
            if df_resampled.empty:
                return 0.0
            
            interval_sentiments = []
            for _, interval_data in df_resampled.iterrows():
                pos = interval_data['sentiment_positive']
                neg = interval_data['sentiment_negative']
                conf = interval_data['sentiment_confidence']
                
                if conf > 0:
                    sentiment = (pos - neg) * conf
                    interval_sentiments.append(sentiment)
            
            if not interval_sentiments:
                return 0.0
            
            return float(np.clip(np.mean(interval_sentiments), -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error in 60-minute aggregation: {e}")
            return 0.0
    
    def aggregate_daily(self, df: pd.DataFrame) -> float:
        """
        Aggregate sentiment over the entire day
        
        Args:
            df: DataFrame with sentiment data for a single day
            
        Returns:
            Daily aggregated sentiment score
        """
        try:
            if df.empty:
                return 0.0
            
            return self._calculate_volume_weighted_sentiment(df)
            
        except Exception as e:
            logger.error(f"Error in daily aggregation: {e}")
            return 0.0
    
    def calculate_sentiment_momentum(self, df: pd.DataFrame, window_hours: int = 4) -> float:
        """
        Calculate sentiment momentum over a rolling window
        
        Args:
            df: DataFrame with sentiment data
            window_hours: Rolling window size in hours
            
        Returns:
            Sentiment momentum score
        """
        try:
            if df.empty or len(df) < 2:
                return 0.0
            
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df_sorted = df.sort_values('timestamp')
            
            df_sorted = df_sorted.set_index('timestamp')
            window_str = f'{window_hours}H'
            
            rolling_sentiment = df_sorted.rolling(window_str).apply(
                lambda x: self._calculate_weighted_sentiment(x.reset_index())
            )['sentiment_positive']  # Use positive as proxy
            
            if len(rolling_sentiment.dropna()) < 2:
                return 0.0
            
            recent_sentiment = rolling_sentiment.iloc[-1]
            previous_sentiment = rolling_sentiment.iloc[-2]
            
            momentum = recent_sentiment - previous_sentiment
            return float(np.clip(momentum, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating sentiment momentum: {e}")
            return 0.0
    
    def calculate_sentiment_volatility(self, df: pd.DataFrame) -> float:
        """
        Calculate sentiment volatility for the day
        
        Args:
            df: DataFrame with sentiment data for a single day
            
        Returns:
            Sentiment volatility score (0 to 1)
        """
        try:
            if df.empty or len(df) < 2:
                return 0.0
            
            sentiment_scores = []
            for _, row in df.iterrows():
                if self._validate_sentiment_data(pd.DataFrame([row])):
                    score = (row['sentiment_positive'] - row['sentiment_negative']) * row['sentiment_confidence']
                    sentiment_scores.append(score)
            
            if len(sentiment_scores) < 2:
                return 0.0
            
            volatility = np.std(sentiment_scores)
            
            normalized_volatility = min(float(volatility) / 2.0, 1.0)
            
            return float(normalized_volatility)
            
        except Exception as e:
            logger.error(f"Error calculating sentiment volatility: {e}")
            return 0.0
    
    def get_aggregation_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive sentiment aggregation summary
        
        Args:
            df: DataFrame with sentiment data for a single day
            
        Returns:
            Dict with all aggregation metrics
        """
        try:
            summary = {
                'sent_5m': self.aggregate_5min(df),
                'sent_10m': self.aggregate_10min(df),
                'sent_60m': self.aggregate_60min(df),
                'sent_daily': self.aggregate_daily(df),
                'sent_momentum': self.calculate_sentiment_momentum(df),
                'sent_volatility': self.calculate_sentiment_volatility(df),
                'tweet_count': len(df),
                'avg_confidence': df['sentiment_confidence'].mean() if not df.empty else 0.0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating aggregation summary: {e}")
            return {
                'sent_5m': 0.0,
                'sent_10m': 0.0,
                'sent_60m': 0.0,
                'sent_daily': 0.0,
                'sent_momentum': 0.0,
                'sent_volatility': 0.0,
                'tweet_count': 0,
                'avg_confidence': 0.0
            }


def test_sentiment_aggregator():
    """Test sentiment aggregator functionality"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    test_data = []
    base_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    
    for i in range(100):
        timestamp = base_time + timedelta(minutes=i*2)  # Every 2 minutes
        
        if i < 30:  # Morning: slightly positive
            pos, neg, neu = 0.6, 0.2, 0.2
        elif i < 60:  # Midday: neutral
            pos, neg, neu = 0.4, 0.3, 0.3
        else:  # Afternoon: slightly negative
            pos, neg, neu = 0.3, 0.5, 0.2
        
        test_data.append({
            'timestamp': timestamp,
            'sentiment_positive': pos + np.random.normal(0, 0.1),
            'sentiment_negative': neg + np.random.normal(0, 0.1),
            'sentiment_neutral': neu + np.random.normal(0, 0.1),
            'sentiment_confidence': 0.8 + np.random.normal(0, 0.1),
            'retweet_count': np.random.randint(0, 50),
            'like_count': np.random.randint(0, 100)
        })
    
    df = pd.DataFrame(test_data)
    
    aggregator = SentimentAggregator()
    summary = aggregator.get_aggregation_summary(df)
    
    print("🧪 Sentiment Aggregator Test Results:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Sentiment aggregator test completed")


if __name__ == "__main__":
    test_sentiment_aggregator()
