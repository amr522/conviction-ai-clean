#!/usr/bin/env python3

import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
from datetime import datetime

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
    """Fetch SPY and QQQ data for cross-asset signals"""
    try:
        print("Fetching cross-asset data (SPY, QQQ)...")
        
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        qqq = yf.download('QQQ', start=start_date, end=end_date, progress=False)
        
        if spy.empty or qqq.empty:
            print("Failed to fetch cross-asset data")
            return None
        
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

def create_lagged_returns_features(df, cross_asset_df=None):
    """Create only lagged returns features"""
    enhanced_df = df.copy()
    
    enhanced_df['ret_1d_lag1'] = enhanced_df['close'].pct_change(1).shift(1)
    enhanced_df['ret_3d_lag1'] = enhanced_df['close'].pct_change(3).shift(1)
    enhanced_df['ret_5d_lag1'] = enhanced_df['close'].pct_change(5).shift(1)
    
    enhanced_df['vol_5d_lag1'] = enhanced_df['close'].pct_change().rolling(5).std().shift(1)
    enhanced_df['vol_10d_lag1'] = enhanced_df['close'].pct_change().rolling(10).std().shift(1)
    
    enhanced_df['price_mom_5d_lag1'] = (enhanced_df['close'] / enhanced_df['close'].shift(5)).shift(1)
    enhanced_df['price_mom_10d_lag1'] = (enhanced_df['close'] / enhanced_df['close'].shift(10)).shift(1)
    
    enhanced_df['target_next_day'] = (enhanced_df['close'].shift(-1) > enhanced_df['close']).astype(int)
    
    enhanced_df = enhanced_df.dropna()
    return enhanced_df

def create_cross_asset_features(df, cross_asset_df):
    """Create only cross-asset signal features"""
    if cross_asset_df is None:
        return None
        
    enhanced_df = df.copy()
    
    enhanced_df = enhanced_df.merge(cross_asset_df, on='date', how='left')
    
    cross_asset_features = ['spy_ret_1d', 'spy_ret_3d', 'spy_ret_5d', 
                           'qqq_ret_1d', 'qqq_ret_3d', 'qqq_ret_5d',
                           'spy_qqq_ratio_change']
    
    for feature in cross_asset_features:
        if feature in enhanced_df.columns:
            enhanced_df[f'{feature}_lag1'] = enhanced_df[feature].shift(1)
            enhanced_df = enhanced_df.drop(columns=[feature])  # Remove non-lagged version
    
    enhanced_df['target_next_day'] = (enhanced_df['close'].shift(-1) > enhanced_df['close']).astype(int)
    
    enhanced_df = enhanced_df.dropna()
    return enhanced_df

def create_combined_features(df, cross_asset_df):
    """Create combined lagged returns + cross-asset features"""
    enhanced_df = df.copy()
    
    enhanced_df['ret_1d_lag1'] = enhanced_df['close'].pct_change(1).shift(1)
    enhanced_df['ret_3d_lag1'] = enhanced_df['close'].pct_change(3).shift(1)
    enhanced_df['ret_5d_lag1'] = enhanced_df['close'].pct_change(5).shift(1)
    
    enhanced_df['vol_5d_lag1'] = enhanced_df['close'].pct_change().rolling(5).std().shift(1)
    enhanced_df['vol_10d_lag1'] = enhanced_df['close'].pct_change().rolling(10).std().shift(1)
    
    enhanced_df['price_mom_5d_lag1'] = (enhanced_df['close'] / enhanced_df['close'].shift(5)).shift(1)
    enhanced_df['price_mom_10d_lag1'] = (enhanced_df['close'] / enhanced_df['close'].shift(10)).shift(1)
    
    if cross_asset_df is not None:
        enhanced_df = enhanced_df.merge(cross_asset_df, on='date', how='left')
        
        cross_asset_features = ['spy_ret_1d', 'spy_ret_3d', 'spy_ret_5d', 
                               'qqq_ret_1d', 'qqq_ret_3d', 'qqq_ret_5d',
                               'spy_qqq_ratio_change']
        
        for feature in cross_asset_features:
            if feature in enhanced_df.columns:
                enhanced_df[f'{feature}_lag1'] = enhanced_df[feature].shift(1)
                enhanced_df = enhanced_df.drop(columns=[feature])
    
    enhanced_df['target_next_day'] = (enhanced_df['close'].shift(-1) > enhanced_df['close']).astype(int)
    
    enhanced_df = enhanced_df.dropna()
    return enhanced_df

def run_cv_for_feature_group(symbols, feature_group_name, feature_creator_func, cross_asset_df=None):
    """Run 3-fold CV for a specific feature group"""
    print(f"\n=== {feature_group_name} ===")
    
    results = {}
    
    for symbol in symbols:
        print(f"\nTraining {symbol} with {feature_group_name}...")
        
        df = load_symbol_data(symbol)
        if df is None:
            results[symbol] = {'auc': 'N/A', 'error': 'Data not found'}
            continue
        
        if feature_group_name == "Cross-Asset Signals Only" and cross_asset_df is None:
            results[symbol] = {'auc': 'N/A', 'error': 'Cross-asset data unavailable'}
            continue
            
        enhanced_df = feature_creator_func(df, cross_asset_df)
        if enhanced_df is None:
            results[symbol] = {'auc': 'N/A', 'error': 'Feature creation failed'}
            continue
        
        basic_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'target_next_day', 'symbol']
        feature_cols = [col for col in enhanced_df.columns if col not in basic_cols]
        
        if len(feature_cols) == 0:
            results[symbol] = {'auc': 'N/A', 'error': 'No features available'}
            continue
        
        X = enhanced_df[feature_cols]
        y = enhanced_df['target_next_day']
        
        # Handle missing values
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
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=1
            )
            
            mean_auc = cv_scores.mean()
            std_auc = cv_scores.std()
            
            results[symbol] = {
                'auc': f"{mean_auc:.4f}",
                'std': f"{std_auc:.4f}",
                'feature_count': len(feature_cols),
                'sample_count': len(X)
            }
            
            print(f"  {symbol}: AUC = {mean_auc:.4f} ± {std_auc:.4f} ({len(feature_cols)} features, {len(X)} samples)")
            
        except Exception as e:
            results[symbol] = {'auc': 'N/A', 'error': str(e)}
            print(f"  {symbol}: Error - {e}")
    
    return results

def generate_comparison_report(all_results):
    """Generate markdown comparison report for all feature groups"""
    
    md_content = "# Feature Group Evaluation Report\n\n"
    md_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
    md_content += "**Symbols**: AAPL, MSFT, AMZN, GOOGL, TSLA\n\n"
    md_content += "**Model**: LightGBM with 3-fold stratified cross-validation\n\n"
    
    md_content += "## Cross-Validation Results by Feature Group\n\n"
    md_content += "| Symbol | Lagged Returns Only | Cross-Asset Signals Only | Combined Features |\n"
    md_content += "|--------|-------------------|--------------------------|------------------|\n"
    
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    
    for symbol in symbols:
        row = f"| {symbol} |"
        
        for group_name in ["Lagged Returns Only", "Cross-Asset Signals Only", "Combined Features"]:
            if group_name in all_results and symbol in all_results[group_name]:
                result = all_results[group_name][symbol]
                if 'error' in result:
                    row += f" N/A |"
                else:
                    auc_val = float(result['auc'])
                    status = "✅" if auc_val >= 0.55 else ""
                    row += f" {result['auc']} {status} |"
            else:
                row += f" N/A |"
        
        md_content += row + "\n"
    
    md_content += "\n## Analysis\n\n"
    
    high_auc_groups = []
    
    for group_name, group_results in all_results.items():
        successful_symbols = []
        high_auc_symbols = []
        
        for symbol, result in group_results.items():
            if 'error' not in result:
                successful_symbols.append(symbol)
                auc_val = float(result['auc'])
                if auc_val >= 0.55:
                    high_auc_symbols.append((symbol, auc_val))
        
        md_content += f"### {group_name}\n"
        md_content += f"- **Successful Training**: {len(successful_symbols)}/{len(group_results)} symbols\n"
        md_content += f"- **Symbols Meeting AUC ≥ 0.55**: {len(high_auc_symbols)}\n"
        
        if high_auc_symbols:
            md_content += f"- **High-Performing Symbols**: "
            symbol_list = [f"{symbol} ({auc:.4f})" for symbol, auc in sorted(high_auc_symbols, key=lambda x: x[1], reverse=True)]
            md_content += ", ".join(symbol_list) + "\n"
            high_auc_groups.append((group_name, high_auc_symbols))
        
        md_content += "\n"
    
    md_content += "## Integration Decision\n\n"
    
    if high_auc_groups:
        md_content += f"✅ **RECOMMEND INTEGRATION**: {len(high_auc_groups)} feature group(s) achieved AUC ≥ 0.55\n\n"
        
        for group_name, symbols_aucs in high_auc_groups:
            md_content += f"**{group_name}**:\n"
            for symbol, auc in symbols_aucs:
                md_content += f"- {symbol}: {auc:.4f}\n"
            md_content += "\n"
        
        md_content += "### Next Steps for Integration:\n"
        md_content += "1. Merge successful feature groups into main feature engineering pipeline\n"
        md_content += "2. Update feature metadata and training scripts\n"
        md_content += "3. Relaunch HPO job for all 46 symbols with enhanced features\n"
        md_content += "4. Monitor validation performance across full symbol universe\n\n"
        
    else:
        md_content += "❌ **DO NOT INTEGRATE**: No feature groups achieved AUC ≥ 0.55 threshold\n\n"
        md_content += "### Recommended Next Steps:\n"
        md_content += "1. Investigate alternative feature engineering approaches:\n"
        md_content += "   - Rolling correlations between symbols\n"
        md_content += "   - Volatility regime indicators\n"
        md_content += "   - Macro-economic indicators\n"
        md_content += "   - Sector rotation signals\n"
        md_content += "2. Consider different target horizons (3-day, 5-day returns)\n"
        md_content += "3. Explore ensemble methods or different algorithms\n"
        md_content += "4. Investigate hyperparameter optimization for LightGBM\n\n"
    
    return md_content, high_auc_groups

def main():
    """Main execution function"""
    print("Feature Group Evaluation: Lagged Returns vs Cross-Asset vs Combined")
    print("=" * 80)
    
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    print(f"Selected symbols: {', '.join(symbols)}")
    
    first_symbol_df = load_symbol_data(symbols[0])
    if first_symbol_df is not None and 'date' in first_symbol_df.columns:
        start_date = first_symbol_df['date'].min()
        end_date = first_symbol_df['date'].max()
        cross_asset_df = fetch_cross_asset_data(start_date, end_date)
    else:
        cross_asset_df = None
    
    all_results = {}
    
    all_results["Lagged Returns Only"] = run_cv_for_feature_group(
        symbols, "Lagged Returns Only", create_lagged_returns_features
    )
    
    all_results["Cross-Asset Signals Only"] = run_cv_for_feature_group(
        symbols, "Cross-Asset Signals Only", create_cross_asset_features, cross_asset_df
    )
    
    all_results["Combined Features"] = run_cv_for_feature_group(
        symbols, "Combined Features", create_combined_features, cross_asset_df
    )
    
    report_md, high_auc_groups = generate_comparison_report(all_results)
    
    with open('FEATURE_GROUP_EVALUATION_REPORT.md', 'w') as f:
        f.write(report_md)
    
    with open('feature_group_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("FEATURE GROUP EVALUATION COMPLETE")
    print("=" * 80)
    
    print(f"📊 Comparison report: FEATURE_GROUP_EVALUATION_REPORT.md")
    print(f"📋 Detailed results: feature_group_results.json")
    
    print(f"\n🎯 INTEGRATION DECISION:")
    if high_auc_groups:
        print(f"✅ {len(high_auc_groups)} feature group(s) achieved AUC ≥ 0.55")
        print("✅ RECOMMEND: Integrate successful feature groups into full 46-stock pipeline")
        for group_name, symbols_aucs in high_auc_groups:
            print(f"   - {group_name}: {len(symbols_aucs)} symbols above threshold")
    else:
        print("❌ No feature groups achieved AUC ≥ 0.55 threshold")
        print("❌ DO NOT INTEGRATE: Additional feature engineering needed")
    
    return all_results, high_auc_groups

if __name__ == "__main__":
    main()
