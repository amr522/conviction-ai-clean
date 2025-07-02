#!/usr/bin/env python3
"""
Generate synthetic features data for 46 stocks to match the expected schema
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_features():
    """Generate synthetic features data for 46 stocks"""
    
    symbols = []
    with open('config/models_to_train_46.txt', 'r') as f:
        symbols = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Generating synthetic data for {len(symbols)} symbols")
    
    start_date = datetime(2021, 1, 5)
    end_date = datetime(2025, 6, 27)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    all_data = []
    
    for symbol in symbols:
        print(f"Generating data for {symbol}...")
        
        for date in date_range:
            if date.weekday() < 5:
                row = {
                    'date': date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'target_next_day': np.random.choice([0, 1], p=[0.52, 0.48]),
                    'open': np.random.uniform(50, 500),
                    'high': np.random.uniform(50, 500),
                    'low': np.random.uniform(50, 500),
                    'close': np.random.uniform(50, 500),
                    'volume': np.random.randint(1000000, 100000000),
                    'returns_1d': np.random.normal(0, 0.02),
                    'returns_5d': np.random.normal(0, 0.05),
                    'returns_20d': np.random.normal(0, 0.1),
                    'volatility_20d': np.random.uniform(0.1, 0.8),
                    'rsi_14': np.random.uniform(20, 80),
                    'macd': np.random.normal(0, 2),
                    'macd_signal': np.random.normal(0, 2),
                    'bb_upper': np.random.uniform(100, 600),
                    'bb_lower': np.random.uniform(50, 400),
                    'bb_middle': np.random.uniform(75, 500),
                    'atr_14': np.random.uniform(1, 20),
                    'adx_14': np.random.uniform(10, 60),
                    'cci_20': np.random.uniform(-200, 200),
                    'williams_r': np.random.uniform(-100, 0),
                    'stoch_k': np.random.uniform(0, 100),
                    'stoch_d': np.random.uniform(0, 100),
                    'obv': np.random.randint(-1000000, 1000000),
                    'mfi_14': np.random.uniform(0, 100),
                    'trix': np.random.normal(0, 0.01),
                    'vwap': np.random.uniform(50, 500),
                    'ema_12': np.random.uniform(50, 500),
                    'ema_26': np.random.uniform(50, 500),
                    'sma_50': np.random.uniform(50, 500),
                    'sma_200': np.random.uniform(50, 500),
                    'price_to_sma50': np.random.uniform(0.8, 1.2),
                    'price_to_sma200': np.random.uniform(0.7, 1.3),
                    'volume_sma_20': np.random.randint(1000000, 50000000),
                    'volume_ratio': np.random.uniform(0.5, 3.0),
                    'news_sentiment': np.random.uniform(-1, 1),
                    'news_volume': np.random.randint(0, 50),
                    'sector_momentum': np.random.normal(0, 0.03),
                    'market_cap': np.random.uniform(1e9, 3e12),
                    'pe_ratio': np.random.uniform(5, 50),
                    'pb_ratio': np.random.uniform(0.5, 10),
                    'debt_to_equity': np.random.uniform(0, 2),
                    'roe': np.random.uniform(-0.2, 0.4),
                    'roa': np.random.uniform(-0.1, 0.2),
                    'current_ratio': np.random.uniform(0.5, 5),
                    'quick_ratio': np.random.uniform(0.3, 3),
                    'gross_margin': np.random.uniform(0.1, 0.8),
                    'operating_margin': np.random.uniform(-0.2, 0.5),
                    'net_margin': np.random.uniform(-0.3, 0.4),
                    'asset_turnover': np.random.uniform(0.1, 3),
                    'inventory_turnover': np.random.uniform(1, 20),
                    'receivables_turnover': np.random.uniform(2, 50),
                    'cash_ratio': np.random.uniform(0, 2),
                    'interest_coverage': np.random.uniform(-5, 50),
                    'dividend_yield': np.random.uniform(0, 0.08),
                    'payout_ratio': np.random.uniform(0, 1.5),
                    'earnings_growth': np.random.uniform(-0.5, 1),
                    'revenue_growth': np.random.uniform(-0.3, 0.8),
                    'book_value_growth': np.random.uniform(-0.4, 0.6),
                    'fcf_yield': np.random.uniform(-0.1, 0.2),
                    'ev_ebitda': np.random.uniform(5, 100),
                    'price_to_sales': np.random.uniform(0.5, 20),
                    'price_to_book': np.random.uniform(0.5, 15),
                    'price_to_fcf': np.random.uniform(5, 100),
                    'enterprise_value': np.random.uniform(1e9, 4e12),
                    'shares_outstanding': np.random.uniform(1e6, 1e10),
                    'float_shares': np.random.uniform(1e6, 1e10),
                    'insider_ownership': np.random.uniform(0, 0.7),
                    'institutional_ownership': np.random.uniform(0.2, 0.95),
                    'short_interest': np.random.uniform(0, 0.3),
                    'days_to_cover': np.random.uniform(0.1, 20)
                }
                all_data.append(row)
    
    df = pd.DataFrame(all_data)
    
    os.makedirs('data/processed_features', exist_ok=True)
    output_file = 'data/processed_features/all_symbols_features.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Generated {len(df)} rows of synthetic data")
    print(f"Data saved to {output_file}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Symbols: {sorted(df['symbol'].unique())}")
    
    return output_file

if __name__ == "__main__":
    generate_synthetic_features()
