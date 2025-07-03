#!/usr/bin/env python3
"""
Create Combined Features dataset for 46-stock pipeline integration

This script:
1. Loads individual symbol feature files from data/processed_with_news_20250628/
2. Removes future return columns that cause data leakage
3. Adds Combined Features (lagged returns + cross-asset signals) for each symbol
4. Creates the missing data/processed_features/all_symbols_features.csv file
5. Prepares data for bundling with train_models_and_prepare_56_new.sh --bundle-only
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import os
from datetime import datetime

def load_symbol_data(symbol, data_dir):
    """Load individual symbol data from processed files"""
    file_path = data_dir / f"{symbol}_features.csv"
    
    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    future_columns = ['target_1d', 'target_3d', 'target_5d', 'target_10d']
    df = df.drop(columns=[col for col in future_columns if col in df.columns])
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    return df

def fetch_cross_asset_data(start_date, end_date):
    """Fetch cross-asset data (SPY, QQQ) for the given date range"""
    print(f"Fetching cross-asset data from {start_date} to {end_date}...")
    
    try:
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        qqq = yf.download('QQQ', start=start_date, end=end_date, progress=False)
        
        if spy.empty or qqq.empty:
            print("⚠️ Warning: Cross-asset data is empty")
            return None
        
        spy = spy.reset_index()
        qqq = qqq.reset_index()
        
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = ['date'] + [f'spy_{col[0].lower()}' for col in spy.columns[1:]]
        else:
            spy.columns = ['date'] + [f'spy_{col.lower()}' for col in spy.columns[1:]]
            
        if isinstance(qqq.columns, pd.MultiIndex):
            qqq.columns = ['date'] + [f'qqq_{col[0].lower()}' for col in qqq.columns[1:]]
        else:
            qqq.columns = ['date'] + [f'qqq_{col.lower()}' for col in qqq.columns[1:]]
        
        cross_asset_df = pd.merge(spy, qqq, on='date', how='inner')
        
        print(f"✅ Cross-asset data loaded: {len(cross_asset_df)} rows")
        return cross_asset_df
        
    except Exception as e:
        print(f"❌ Error fetching cross-asset data: {e}")
        return None

def create_combined_features(df, cross_asset_df=None):
    """Create Combined Features (lagged returns + cross-asset signals) for a symbol"""
    enhanced_df = df.copy()
    
    enhanced_df['ret_1d_lag1'] = enhanced_df['close'].pct_change(1).shift(1)
    enhanced_df['ret_3d_lag1'] = enhanced_df['close'].pct_change(3).shift(1)
    enhanced_df['ret_5d_lag1'] = enhanced_df['close'].pct_change(5).shift(1)
    
    enhanced_df['vol_5d_lag1'] = enhanced_df['close'].pct_change().rolling(5).std().shift(1)
    enhanced_df['vol_10d_lag1'] = enhanced_df['close'].pct_change().rolling(10).std().shift(1)
    enhanced_df['price_mom_5d_lag1'] = (enhanced_df['close'] / enhanced_df['close'].shift(5)).shift(1)
    enhanced_df['price_mom_10d_lag1'] = (enhanced_df['close'] / enhanced_df['close'].shift(10)).shift(1)
    
    if cross_asset_df is not None:
        enhanced_df = pd.merge(enhanced_df, cross_asset_df, on='date', how='left')
        
        enhanced_df['spy_ret_1d_lag1'] = enhanced_df['spy_close'].pct_change(1).shift(1)
        enhanced_df['spy_ret_3d_lag1'] = enhanced_df['spy_close'].pct_change(3).shift(1)
        enhanced_df['spy_ret_5d_lag1'] = enhanced_df['spy_close'].pct_change(5).shift(1)
        
        enhanced_df['qqq_ret_1d_lag1'] = enhanced_df['qqq_close'].pct_change(1).shift(1)
        enhanced_df['qqq_ret_3d_lag1'] = enhanced_df['qqq_close'].pct_change(3).shift(1)
        enhanced_df['qqq_ret_5d_lag1'] = enhanced_df['qqq_close'].pct_change(5).shift(1)
        
        enhanced_df['spy_qqq_ratio_lag1'] = (enhanced_df['spy_close'] / enhanced_df['qqq_close']).shift(1)
        enhanced_df['spy_qqq_ratio_change_lag1'] = enhanced_df['spy_qqq_ratio_lag1'].pct_change(1).shift(1)
        
        enhanced_df['spy_vol_5d_lag1'] = enhanced_df['spy_close'].pct_change().rolling(5).std().shift(1)
        enhanced_df['qqq_vol_5d_lag1'] = enhanced_df['qqq_close'].pct_change().rolling(5).std().shift(1)
        
        cross_asset_price_cols = [col for col in enhanced_df.columns if 'spy_' in col or 'qqq_' in col]
        cross_asset_price_cols = [col for col in cross_asset_price_cols if any(price_term in col for price_term in ['open', 'high', 'low', 'close', 'volume'])]
        enhanced_df = enhanced_df.drop(columns=cross_asset_price_cols)
    
    enhanced_df['target_next_day'] = (enhanced_df['close'].shift(-1) > enhanced_df['close']).astype(int)
    
    enhanced_df = enhanced_df.dropna()
    
    return enhanced_df

def create_46_stock_combined_features():
    """Create Combined Features dataset for all 46 stocks"""
    print("=== Creating 46-Stock Combined Features Dataset ===")
    
    symbols_file = Path("config/models_to_train_46.txt")
    if not symbols_file.exists():
        print(f"❌ Error: Symbol file not found: {symbols_file}")
        return None
    
    with open(symbols_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(symbols)} symbols: {symbols}")
    
    data_dir = Path("data/processed_with_news_20250628")
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        return None
    
    first_symbol_df = load_symbol_data(symbols[0], data_dir)
    if first_symbol_df is None:
        print(f"❌ Error: Could not load first symbol {symbols[0]}")
        return None
    
    start_date = first_symbol_df['date'].min()
    end_date = first_symbol_df['date'].max()
    
    cross_asset_df = fetch_cross_asset_data(start_date, end_date)
    
    all_data = []
    successful_symbols = []
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        df = load_symbol_data(symbol, data_dir)
        if df is None:
            print(f"⚠️ Skipping {symbol} - could not load data")
            continue
        
        try:
            enhanced_df = create_combined_features(df, cross_asset_df)
            
            if len(enhanced_df) == 0:
                print(f"⚠️ Skipping {symbol} - no data after feature creation")
                continue
            
            all_data.append(enhanced_df)
            successful_symbols.append(symbol)
            print(f"✅ {symbol}: {len(enhanced_df)} rows with Combined Features")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            continue
    
    if not all_data:
        print("❌ Error: No symbols processed successfully")
        return None
    
    print(f"\nCombining data from {len(successful_symbols)} symbols...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    output_dir = Path("data/processed_features")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "all_symbols_features.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Combined Features dataset created:")
    print(f"   File: {output_file}")
    print(f"   Symbols: {len(successful_symbols)}")
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Features: {len(combined_df.columns)}")
    print(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    feature_cols = [col for col in combined_df.columns if col not in ['date', 'symbol', 'target_next_day']]
    combined_feature_cols = [col for col in feature_cols if any(term in col for term in ['ret_', 'vol_', 'mom_', 'spy_', 'qqq_'])]
    
    print(f"\n📊 Feature Summary:")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Combined Features: {len(combined_feature_cols)}")
    print(f"   Original features: {len(feature_cols) - len(combined_feature_cols)}")
    
    symbols_file = output_dir / "processed_symbols.txt"
    with open(symbols_file, 'w') as f:
        for symbol in successful_symbols:
            f.write(f"{symbol}\n")
    
    print(f"   Processed symbols saved to: {symbols_file}")
    
    return output_file

def main():
    """Main function to create 46-stock Combined Features dataset"""
    print("🚀 Starting 46-Stock Combined Features Creation")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    try:
        output_file = create_46_stock_combined_features()
        
        if output_file:
            print(f"\n🎉 Success! Combined Features dataset ready for bundling:")
            print(f"   {output_file}")
            print(f"\nNext steps:")
            print(f"1. Run bundling: bash train_models_and_prepare_56_new.sh --bundle-only")
            print(f"2. Launch HPO: python aws_hpo_launch.py")
            return True
        else:
            print(f"\n❌ Failed to create Combined Features dataset")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
