#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import yfinance as yf
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_symbol_data(symbol):
    """Load individual symbol data from processed files"""
    data_dir = Path("data/processed_with_news_20250628")
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
    """Fetch cross-asset data (SPY, QQQ) for the same period"""
    print("Fetching cross-asset data (SPY, QQQ)...")
    
    try:
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        qqq = yf.download('QQQ', start=start_date, end=end_date, progress=False)
        
        cross_asset = pd.DataFrame()
        cross_asset['date'] = spy.index
        cross_asset['spy_close'] = spy['Close'].values
        cross_asset['qqq_close'] = qqq['Close'].values
        
        cross_asset['spy_ret_1d'] = cross_asset['spy_close'].pct_change(1)
        cross_asset['spy_ret_3d'] = cross_asset['spy_close'].pct_change(3)
        cross_asset['spy_ret_5d'] = cross_asset['spy_close'].pct_change(5)
        
        cross_asset['qqq_ret_1d'] = cross_asset['qqq_close'].pct_change(1)
        cross_asset['qqq_ret_3d'] = cross_asset['qqq_close'].pct_change(3)
        cross_asset['qqq_ret_5d'] = cross_asset['qqq_close'].pct_change(5)
        
        cross_asset['spy_qqq_ratio'] = cross_asset['spy_close'] / cross_asset['qqq_close']
        cross_asset['spy_qqq_ratio_change'] = cross_asset['spy_qqq_ratio'].pct_change(1)
        
        cross_asset = cross_asset.reset_index(drop=True)
        return cross_asset
        
    except Exception as e:
        print(f"Error fetching cross-asset data: {e}")
        return None

def create_enhanced_features(df, cross_asset_df=None):
    """Create enhanced features with strict past-only constraints"""
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return None
    
    enhanced_df = df.copy()
    
    print("  Creating lagged returns...")
    enhanced_df['ret_1d_lag1'] = enhanced_df['close'].pct_change(1).shift(1)  # Yesterday's 1-day return
    enhanced_df['ret_3d_lag1'] = enhanced_df['close'].pct_change(3).shift(1)  # 3-day return ending yesterday
    enhanced_df['ret_5d_lag1'] = enhanced_df['close'].pct_change(5).shift(1)  # 5-day return ending yesterday
    
    enhanced_df['vol_5d_lag1'] = enhanced_df['close'].pct_change().rolling(5).std().shift(1)
    enhanced_df['vol_10d_lag1'] = enhanced_df['close'].pct_change().rolling(10).std().shift(1)
    
    enhanced_df['volume_ratio_5d_lag1'] = (enhanced_df['volume'] / enhanced_df['volume'].rolling(5).mean()).shift(1)
    enhanced_df['volume_ratio_10d_lag1'] = (enhanced_df['volume'] / enhanced_df['volume'].rolling(10).mean()).shift(1)
    
    enhanced_df['price_mom_5d_lag1'] = (enhanced_df['close'] / enhanced_df['close'].shift(5)).shift(1)
    enhanced_df['price_mom_10d_lag1'] = (enhanced_df['close'] / enhanced_df['close'].shift(10)).shift(1)
    
    enhanced_df['hl_spread_lag1'] = ((enhanced_df['high'] - enhanced_df['low']) / enhanced_df['close']).shift(1)
    enhanced_df['hl_spread_5d_avg_lag1'] = enhanced_df['hl_spread_lag1'].rolling(5).mean().shift(1)
    
    if cross_asset_df is not None and 'date' in enhanced_df.columns:
        print("  Adding cross-asset features...")
        enhanced_df = enhanced_df.merge(cross_asset_df, on='date', how='left')
        
        cross_asset_features = ['spy_ret_1d', 'spy_ret_3d', 'spy_ret_5d', 
                               'qqq_ret_1d', 'qqq_ret_3d', 'qqq_ret_5d',
                               'spy_qqq_ratio_change']
        
        for feature in cross_asset_features:
            if feature in enhanced_df.columns:
                enhanced_df[f'{feature}_lag1'] = enhanced_df[feature].shift(1)
                enhanced_df = enhanced_df.drop(columns=[feature])  # Remove non-lagged version
    
    enhanced_df['target_next_day'] = (enhanced_df['close'].shift(-1) > enhanced_df['close']).astype(int)
    
    initial_rows = len(enhanced_df)
    enhanced_df = enhanced_df.dropna()
    final_rows = len(enhanced_df)
    
    print(f"  Rows after feature creation: {initial_rows} -> {final_rows} (removed {initial_rows - final_rows} NaN rows)")
    
    return enhanced_df

def generate_feature_report(symbols):
    """Generate feature validation report for selected symbols"""
    print("=== Feature Validation Report ===")
    
    feature_report = {
        'symbols_processed': [],
        'feature_definitions': {},
        'feature_statistics': {},
        'cross_asset_availability': False
    }
    
    first_symbol_df = load_symbol_data(symbols[0])
    if first_symbol_df is None or 'date' not in first_symbol_df.columns:
        print("Cannot determine date range for cross-asset data")
        cross_asset_df = None
    else:
        start_date = first_symbol_df['date'].min()
        end_date = first_symbol_df['date'].max()
        cross_asset_df = fetch_cross_asset_data(start_date, end_date)
        feature_report['cross_asset_availability'] = cross_asset_df is not None
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        df = load_symbol_data(symbol)
        if df is None:
            continue
            
        enhanced_df = create_enhanced_features(df, cross_asset_df)
        if enhanced_df is None:
            continue
            
        feature_report['symbols_processed'].append(symbol)
        
        basic_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'target_next_day', 'symbol']
        feature_cols = [col for col in enhanced_df.columns if col not in basic_cols]
        
        feature_stats = {}
        for feature in feature_cols:
            if feature in enhanced_df.columns:
                stats = {
                    'mean': float(enhanced_df[feature].mean()),
                    'std': float(enhanced_df[feature].std()),
                    'min': float(enhanced_df[feature].min()),
                    'max': float(enhanced_df[feature].max()),
                    'count': int(enhanced_df[feature].count())
                }
                feature_stats[feature] = stats
        
        feature_report['feature_statistics'][symbol] = feature_stats
    
    feature_report['feature_definitions'] = {
        'ret_1d_lag1': 'close.pct_change(1).shift(1) - Yesterday\'s 1-day return',
        'ret_3d_lag1': 'close.pct_change(3).shift(1) - 3-day return ending yesterday',
        'ret_5d_lag1': 'close.pct_change(5).shift(1) - 5-day return ending yesterday',
        'vol_5d_lag1': 'close.pct_change().rolling(5).std().shift(1) - 5-day volatility lag',
        'vol_10d_lag1': 'close.pct_change().rolling(10).std().shift(1) - 10-day volatility lag',
        'volume_ratio_5d_lag1': '(volume / volume.rolling(5).mean()).shift(1) - Volume vs 5-day avg',
        'volume_ratio_10d_lag1': '(volume / volume.rolling(10).mean()).shift(1) - Volume vs 10-day avg',
        'price_mom_5d_lag1': '(close / close.shift(5)).shift(1) - 5-day price momentum',
        'price_mom_10d_lag1': '(close / close.shift(10)).shift(1) - 10-day price momentum',
        'hl_spread_lag1': '((high - low) / close).shift(1) - High-low spread ratio',
        'hl_spread_5d_avg_lag1': 'hl_spread_lag1.rolling(5).mean().shift(1) - 5-day avg spread',
        'spy_ret_1d_lag1': 'SPY 1-day return lagged',
        'spy_ret_3d_lag1': 'SPY 3-day return lagged',
        'spy_ret_5d_lag1': 'SPY 5-day return lagged',
        'qqq_ret_1d_lag1': 'QQQ 1-day return lagged',
        'qqq_ret_3d_lag1': 'QQQ 3-day return lagged',
        'qqq_ret_5d_lag1': 'QQQ 5-day return lagged',
        'spy_qqq_ratio_change_lag1': 'SPY/QQQ ratio change lagged'
    }
    
    return feature_report

def run_lightgbm_cv(symbols, feature_report):
    """Run 5-fold CV LightGBM for each symbol with enhanced features"""
    print("\n=== LightGBM Cross-Validation ===")
    
    cv_results = {}
    
    if feature_report['cross_asset_availability']:
        first_symbol_df = load_symbol_data(symbols[0])
        if first_symbol_df is not None and 'date' in first_symbol_df.columns:
            start_date = first_symbol_df['date'].min()
            end_date = first_symbol_df['date'].max()
            cross_asset_df = fetch_cross_asset_data(start_date, end_date)
        else:
            cross_asset_df = None
    else:
        cross_asset_df = None
    
    for symbol in symbols:
        print(f"\nTraining {symbol}...")
        
        df = load_symbol_data(symbol)
        if df is None:
            cv_results[symbol] = {'auc': 'N/A', 'error': 'Data not found'}
            continue
            
        enhanced_df = create_enhanced_features(df, cross_asset_df)
        if enhanced_df is None:
            cv_results[symbol] = {'auc': 'N/A', 'error': 'Feature creation failed'}
            continue
        
        basic_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'target_next_day', 'symbol']
        feature_cols = [col for col in enhanced_df.columns if col not in basic_cols]
        
        if len(feature_cols) == 0:
            cv_results[symbol] = {'auc': 'N/A', 'error': 'No features available'}
            continue
        
        X = enhanced_df[feature_cols]
        y = enhanced_df['target_next_day']
        
        X = X.fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        try:
            lgb_model = lgb.LGBMClassifier(
                objective='binary',
                metric='auc',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42
            )
            
            cv_scores = cross_val_score(
                lgb_model, X_scaled, y, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            mean_auc = cv_scores.mean()
            std_auc = cv_scores.std()
            
            cv_results[symbol] = {
                'auc': f"{mean_auc:.4f}",
                'std': f"{std_auc:.4f}",
                'feature_count': len(feature_cols),
                'sample_count': len(X)
            }
            
            print(f"  {symbol}: AUC = {mean_auc:.4f} ± {std_auc:.4f} ({len(feature_cols)} features, {len(X)} samples)")
            
        except Exception as e:
            cv_results[symbol] = {'auc': 'N/A', 'error': str(e)}
            print(f"  {symbol}: Error - {e}")
    
    return cv_results

def generate_markdown_reports(feature_report, cv_results):
    """Generate markdown reports for feature validation and CV results"""
    
    feature_md = "# Enhanced Feature Validation Report\n\n"
    feature_md += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
    feature_md += f"**Symbols Processed**: {', '.join(feature_report['symbols_processed'])}\n\n"
    feature_md += f"**Cross-Asset Data Available**: {'✅ Yes' if feature_report['cross_asset_availability'] else '❌ No'}\n\n"
    
    feature_md += "## Feature Definitions\n\n"
    feature_md += "| Feature Name | Computation Method | Lookback/Window |\n"
    feature_md += "|--------------|-------------------|------------------|\n"
    
    for feature, definition in feature_report['feature_definitions'].items():
        parts = definition.split(' - ')
        computation = parts[0] if len(parts) > 0 else definition
        description = parts[1] if len(parts) > 1 else ""
        
        if 'lag1' in feature:
            lookback = "1 day lag"
        elif '5d' in feature:
            lookback = "5 days"
        elif '10d' in feature:
            lookback = "10 days"
        elif '3d' in feature:
            lookback = "3 days"
        else:
            lookback = "Variable"
            
        feature_md += f"| `{feature}` | {computation} | {lookback} |\n"
    
    feature_md += "\n## Feature Statistics by Symbol\n\n"
    
    if feature_report['symbols_processed']:
        all_features = set()
        for symbol_stats in feature_report['feature_statistics'].values():
            all_features.update(symbol_stats.keys())
        
        for feature in sorted(all_features):
            feature_md += f"### {feature}\n\n"
            feature_md += "| Symbol | Mean | Std | Min | Max | Count |\n"
            feature_md += "|--------|------|-----|-----|-----|-------|\n"
            
            for symbol in feature_report['symbols_processed']:
                stats = feature_report['feature_statistics'][symbol].get(feature, {})
                if stats:
                    feature_md += f"| {symbol} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |\n"
                else:
                    feature_md += f"| {symbol} | N/A | N/A | N/A | N/A | N/A |\n"
            feature_md += "\n"
    
    cv_md = "# LightGBM Cross-Validation Results\n\n"
    cv_md += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
    cv_md += "**Model**: LightGBM with 5-fold stratified cross-validation\n\n"
    
    cv_md += "## Results Summary\n\n"
    cv_md += "| Symbol | AUC | Std | Features | Samples | Status |\n"
    cv_md += "|--------|-----|-----|----------|---------|--------|\n"
    
    successful_symbols = []
    high_auc_symbols = []
    
    for symbol, results in cv_results.items():
        if 'error' in results:
            cv_md += f"| {symbol} | N/A | N/A | N/A | N/A | ❌ {results['error']} |\n"
        else:
            auc_val = float(results['auc'])
            status = "✅ Good" if auc_val >= 0.55 else "⚠️ Below threshold"
            cv_md += f"| {symbol} | {results['auc']} | {results['std']} | {results['feature_count']} | {results['sample_count']} | {status} |\n"
            
            successful_symbols.append(symbol)
            if auc_val >= 0.55:
                high_auc_symbols.append((symbol, auc_val))
    
    cv_md += "\n## Analysis\n\n"
    cv_md += f"**Successful Training**: {len(successful_symbols)}/{len(cv_results)} symbols\n\n"
    cv_md += f"**Symbols Meeting AUC ≥ 0.55 Threshold**: {len(high_auc_symbols)}\n\n"
    
    if high_auc_symbols:
        cv_md += "**High-Performing Symbols**:\n"
        for symbol, auc in sorted(high_auc_symbols, key=lambda x: x[1], reverse=True):
            cv_md += f"- {symbol}: {auc:.4f}\n"
        cv_md += "\n"
    
    return feature_md, cv_md, high_auc_symbols

def main():
    """Main execution function"""
    print("Enhanced Feature Engineering & Cross-Validation")
    print("=" * 60)
    
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    print(f"Selected symbols: {', '.join(symbols)}")
    
    feature_report = generate_feature_report(symbols)
    
    cv_results = run_lightgbm_cv(symbols, feature_report)
    
    feature_md, cv_md, high_auc_symbols = generate_markdown_reports(feature_report, cv_results)
    
    with open('ENHANCED_FEATURE_VALIDATION_REPORT.md', 'w') as f:
        f.write(feature_md)
    
    with open('LIGHTGBM_CV_RESULTS.md', 'w') as f:
        f.write(cv_md)
    
    results = {
        'feature_report': feature_report,
        'cv_results': cv_results,
        'high_auc_symbols': high_auc_symbols,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('enhanced_features_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("ENHANCED FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    
    print(f"📊 Feature validation report: ENHANCED_FEATURE_VALIDATION_REPORT.md")
    print(f"📈 CV results report: LIGHTGBM_CV_RESULTS.md")
    print(f"📋 Detailed results: enhanced_features_results.json")
    
    print(f"\n🎯 INTEGRATION DECISION:")
    if high_auc_symbols:
        print(f"✅ {len(high_auc_symbols)} symbols achieved AUC ≥ 0.55")
        print("✅ RECOMMEND: Integrate enhanced features into full 46-stock pipeline")
        print("\nNext steps:")
        print("1. Merge enhanced feature groups into main feature engineering pipeline")
        print("2. Update feature metadata and training scripts")
        print("3. Relaunch HPO job for all 46 symbols with enhanced features")
    else:
        print("❌ No symbols achieved AUC ≥ 0.55 threshold")
        print("❌ DO NOT INTEGRATE: Additional feature engineering needed")
        print("\nRecommended actions:")
        print("1. Investigate alternative feature engineering approaches")
        print("2. Consider different target horizons (3-day, 5-day returns)")
        print("3. Explore ensemble methods or different algorithms")
    
    return results

if __name__ == "__main__":
    main()
