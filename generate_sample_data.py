#!/usr/bin/env python3
"""
Generate sample feature data for ticker symbols in the expected format.
This script creates CSV files with synthetic data for each ticker in the models_to_train.txt file.
"""

import os
import pandas as pd
import numpy as np
import datetime as dt

# Configuration
OUTPUT_DIR = 'data/processed_with_news_20250628'
TICKERS_FILE = 'config/models_to_train.txt'
START_DATE = '2021-01-01'
END_DATE = '2024-06-28'  # Based on the directory name

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read ticker list
with open(TICKERS_FILE, 'r') as f:
    tickers = [line.strip() for line in f if line.strip()]

print(f"Generating sample data for {len(tickers)} tickers...")

# Generate date range
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='B')  # Business days
dates = [d.strftime('%Y-%m-%d') for d in date_range]

# Features to generate
features = [
    'open', 'high', 'low', 'close', 'volume',
    'vwap', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower',
    'news_sentiment', 'news_volume',
    'options_put_call_ratio', 'options_implied_volatility',
    'target_1d', 'target_3d', 'target_5d', 'target_10d'
]

for ticker in tickers:
    # Create DataFrame with dates
    df = pd.DataFrame(index=dates)
    df.index.name = 'date'
    
    # Generate random data for each feature
    np.random.seed(hash(ticker) % 10000)  # Different seed for each ticker
    
    # Price series (random walk with drift)
    initial_price = np.random.uniform(50, 500)
    returns = np.random.normal(0.0005, 0.015, len(dates))  # Small positive drift
    price_series = initial_price * (1 + returns).cumprod()
    
    # Price-based features
    df['open'] = price_series * np.random.uniform(0.99, 1.01, len(dates))
    df['close'] = price_series
    df['high'] = np.maximum(df['open'], df['close']) * np.random.uniform(1.001, 1.03, len(dates))
    df['low'] = np.minimum(df['open'], df['close']) * np.random.uniform(0.97, 0.999, len(dates))
    df['volume'] = np.random.lognormal(15, 1, len(dates))
    df['vwap'] = df['close'] * np.random.uniform(0.995, 1.005, len(dates))
    
    # Technical indicators
    df['rsi_14'] = np.random.uniform(30, 70, len(dates))
    df['macd'] = np.random.normal(0, 1, len(dates))
    df['macd_signal'] = df['macd'].rolling(9).mean().fillna(0)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean().fillna(df['close'])
    df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(20).std().fillna(df['close'] * 0.02)
    df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(20).std().fillna(df['close'] * 0.02)
    
    # News and options data
    df['news_sentiment'] = np.random.normal(0, 1, len(dates))
    df['news_volume'] = np.random.poisson(5, len(dates))
    df['options_put_call_ratio'] = np.random.beta(2, 2, len(dates))
    df['options_implied_volatility'] = np.random.gamma(2, 0.05, len(dates))
    
    # Target variables (future returns)
    for shift, col in [(1, 'target_1d'), (3, 'target_3d'), (5, 'target_5d'), (10, 'target_10d')]:
        future_returns = np.random.normal(0.001 * shift, 0.01 * np.sqrt(shift), len(dates))
        df[col] = future_returns
    
    # Add symbol column
    df['symbol'] = ticker
    
    # Save to CSV
    output_file = os.path.join(OUTPUT_DIR, f"{ticker}_features.csv")
    df.reset_index().to_csv(output_file, index=False)
    print(f"Generated {output_file} with {len(df)} rows")

print("\nSample data generation complete. Files are saved in:", OUTPUT_DIR)
