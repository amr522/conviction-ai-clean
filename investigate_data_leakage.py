#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def analyze_data_structure():
    """Investigate the structure of the real data to identify potential leakage"""
    print("=== Data Leakage Investigation ===")
    
    data_dir = Path("data/sagemaker_input/46_models/2025-07-02-03-05-02")
    
    print("Loading data samples...")
    train_sample = pd.read_csv(data_dir / "train.csv", nrows=1000)
    val_sample = pd.read_csv(data_dir / "validation.csv", nrows=500)
    test_sample = pd.read_csv(data_dir / "test.csv", nrows=500)
    
    print(f"Train sample shape: {train_sample.shape}")
    print(f"Validation sample shape: {val_sample.shape}")
    print(f"Test sample shape: {test_sample.shape}")
    
    with open(data_dir / "feature_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_columns']
    target_name = metadata['target_column']
    
    print(f"\nFeature count: {len(feature_names)}")
    print(f"Target column: {target_name}")
    
    print("\n=== Feature Analysis ===")
    suspicious_features = []
    
    for i, feature in enumerate(feature_names):
        if any(keyword in feature.lower() for keyword in ['next', 'future', 'forward', 'lead', 'ahead']):
            suspicious_features.append((i+1, feature))  # +1 because target is column 0
    
    if suspicious_features:
        print("🚨 SUSPICIOUS FEATURES (potential future information):")
        for col_idx, feature in suspicious_features:
            print(f"  Column {col_idx}: {feature}")
    else:
        print("✅ No obviously suspicious feature names found")
    
    print("\n=== Target Distribution Analysis ===")
    train_targets = train_sample.iloc[:, 0]
    val_targets = val_sample.iloc[:, 0]
    test_targets = test_sample.iloc[:, 0]
    
    train_dist = train_targets.value_counts(normalize=True).sort_index()
    val_dist = val_targets.value_counts(normalize=True).sort_index()
    test_dist = test_targets.value_counts(normalize=True).sort_index()
    
    print(f"Train target distribution: {train_dist.to_dict()}")
    print(f"Validation target distribution: {val_dist.to_dict()}")
    print(f"Test target distribution: {test_dist.to_dict()}")
    
    train_entropy = -sum(p * np.log2(p) for p in train_dist if p > 0)
    val_entropy = -sum(p * np.log2(p) for p in val_dist if p > 0)
    
    print(f"Train target entropy: {train_entropy:.4f} (max=1.0 for balanced)")
    print(f"Validation target entropy: {val_entropy:.4f}")
    
    return {
        'feature_names': feature_names,
        'suspicious_features': suspicious_features,
        'train_dist': train_dist.to_dict(),
        'val_dist': val_dist.to_dict(),
        'test_dist': test_dist.to_dict(),
        'train_entropy': train_entropy,
        'val_entropy': val_entropy
    }

def analyze_feature_correlations():
    """Look for features that are too highly correlated with targets"""
    print("\n=== Feature-Target Correlation Analysis ===")
    
    data_dir = Path("data/sagemaker_input/46_models/2025-07-02-03-05-02")
    
    train_df = pd.read_csv(data_dir / "train.csv", nrows=5000)
    
    target = train_df.iloc[:, 0]
    features = train_df.iloc[:, 1:]
    
    correlations = features.corrwith(target).abs().sort_values(ascending=False)
    
    print("Top 10 features most correlated with target:")
    for i, (feature_idx, corr) in enumerate(correlations.head(10).items()):
        feature_name = f"feature_{feature_idx}" if feature_idx < len(correlations) else "unknown"
        print(f"  {i+1}. Feature {feature_idx}: {corr:.4f}")
    
    high_corr_threshold = 0.8
    high_corr_features = correlations[correlations > high_corr_threshold]
    
    if len(high_corr_features) > 0:
        print(f"\n🚨 HIGH CORRELATION FEATURES (>{high_corr_threshold}):")
        for feature_idx, corr in high_corr_features.items():
            print(f"  Feature {feature_idx}: {corr:.4f}")
        print("  These may indicate data leakage!")
    else:
        print(f"\n✅ No features with correlation >{high_corr_threshold} found")
    
    return {
        'top_correlations': correlations.head(10).to_dict(),
        'high_corr_features': high_corr_features.to_dict() if len(high_corr_features) > 0 else {},
        'max_correlation': correlations.max()
    }

def analyze_feature_distributions():
    """Check if feature distributions are realistic for financial data"""
    print("\n=== Feature Distribution Analysis ===")
    
    data_dir = Path("data/sagemaker_input/46_models/2025-07-02-03-05-02")
    
    train_df = pd.read_csv(data_dir / "train.csv", nrows=2000)
    features = train_df.iloc[:, 1:]
    
    print("Feature statistics:")
    print(f"  Mean of means: {features.mean().mean():.4f}")
    print(f"  Mean of stds: {features.std().mean():.4f}")
    print(f"  Min value: {features.min().min():.4f}")
    print(f"  Max value: {features.max().max():.4f}")
    
    suspicious_stats = []
    
    for col in features.columns:
        col_data = features[col]
        
        if col_data.std() < 1e-6:
            suspicious_stats.append(f"Feature {col}: Nearly constant (std={col_data.std():.6f})")
        
        if abs(col_data.min()) > 10 or abs(col_data.max()) > 10:
            suspicious_stats.append(f"Feature {col}: Extreme values (range=[{col_data.min():.2f}, {col_data.max():.2f}])")
        
        skewness = col_data.skew()
        if abs(skewness) > 5:
            suspicious_stats.append(f"Feature {col}: Highly skewed (skew={skewness:.2f})")
    
    if suspicious_stats:
        print("\n🚨 SUSPICIOUS FEATURE DISTRIBUTIONS:")
        for stat in suspicious_stats[:10]:  # Show first 10
            print(f"  {stat}")
        if len(suspicious_stats) > 10:
            print(f"  ... and {len(suspicious_stats) - 10} more")
    else:
        print("\n✅ Feature distributions appear normal")
    
    return {
        'mean_of_means': features.mean().mean(),
        'mean_of_stds': features.std().mean(),
        'min_value': features.min().min(),
        'max_value': features.max().max(),
        'suspicious_count': len(suspicious_stats)
    }

def check_temporal_consistency():
    """Check if data splits respect temporal ordering"""
    print("\n=== Temporal Consistency Check ===")
    
    data_dir = Path("data/sagemaker_input/46_models/2025-07-02-03-05-02")
    
    
    train_df = pd.read_csv(data_dir / "train.csv", nrows=1000)
    val_df = pd.read_csv(data_dir / "validation.csv", nrows=500)
    
    
    train_features = train_df.iloc[:, 1:]
    val_features = val_df.iloc[:, 1:]
    
    train_means = train_features.mean()
    val_means = val_features.mean()
    
    mean_differences = (val_means - train_means).abs()
    large_differences = mean_differences[mean_differences > 0.5]
    
    print(f"Features with large mean differences between train/val (>{0.5}):")
    if len(large_differences) > 0:
        for feature_idx, diff in large_differences.head(5).items():
            print(f"  Feature {feature_idx}: {diff:.4f}")
        print(f"  Total: {len(large_differences)} features")
    else:
        print("  None found - this might indicate temporal leakage")
    
    train_target_mean = train_df.iloc[:, 0].mean()
    val_target_mean = val_df.iloc[:, 0].mean()
    target_diff = abs(val_target_mean - train_target_mean)
    
    print(f"\nTarget mean difference: {target_diff:.4f}")
    if target_diff < 0.05:
        print("  ⚠️  Very similar target distributions might indicate leakage")
    else:
        print("  ✅ Target distributions appropriately different")
    
    return {
        'features_with_large_diffs': len(large_differences),
        'target_mean_diff': target_diff,
        'max_feature_diff': mean_differences.max()
    }

def main():
    """Run complete data leakage investigation"""
    print("Data Leakage Investigation")
    print("=" * 50)
    
    results = {}
    
    try:
        results['structure'] = analyze_data_structure()
    except Exception as e:
        print(f"Error in structure analysis: {e}")
        results['structure'] = {'error': str(e)}
    
    try:
        results['correlations'] = analyze_feature_correlations()
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
        results['correlations'] = {'error': str(e)}
    
    try:
        results['distributions'] = analyze_feature_distributions()
    except Exception as e:
        print(f"Error in distribution analysis: {e}")
        results['distributions'] = {'error': str(e)}
    
    try:
        results['temporal'] = check_temporal_consistency()
    except Exception as e:
        print(f"Error in temporal analysis: {e}")
        results['temporal'] = {'error': str(e)}
    
    print("\n" + "=" * 50)
    print("LEAKAGE INVESTIGATION SUMMARY")
    print("=" * 50)
    
    leakage_indicators = []
    
    if results.get('structure', {}).get('suspicious_features'):
        leakage_indicators.append("Suspicious feature names found")
    
    max_corr = results.get('correlations', {}).get('max_correlation', 0)
    if max_corr > 0.8:
        leakage_indicators.append(f"High feature-target correlation: {max_corr:.4f}")
    
    target_diff = results.get('temporal', {}).get('target_mean_diff', 1.0)
    if target_diff < 0.05:
        leakage_indicators.append("Suspiciously similar train/val target distributions")
    
    if leakage_indicators:
        print("🚨 POTENTIAL DATA LEAKAGE INDICATORS:")
        for indicator in leakage_indicators:
            print(f"  - {indicator}")
    else:
        print("✅ No obvious data leakage indicators found")
        print("   The overfitting may be due to other factors")
    
    with open('leakage_investigation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to leakage_investigation_results.json")
    
    return results

if __name__ == "__main__":
    main()
