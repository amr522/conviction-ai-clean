#!/usr/bin/env python3
"""
Feature validation utility to ensure all expected features are present in feature tables.
"""

import argparse
import sys
import polars as pl
from pathlib import Path


def load_expected_features(features_list_path: str) -> list:
    """Load expected feature names from markdown file."""
    features = []
    with open(features_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and headers
            if line and not line.startswith('#') and not line.startswith('##'):
                features.append(line)
    return features


def validate_feature_table(feature_table_path: str, expected_features: list) -> tuple:
    """Validate feature table contains all expected features."""
    if not Path(feature_table_path).exists():
        return False, f"Feature table not found: {feature_table_path}"
    
    try:
        df = pl.read_parquet(feature_table_path)
    except Exception as e:
        return False, f"Failed to read feature table: {e}"
    
    actual_features = set(df.columns)
    expected_features_set = set(expected_features)
    
    # Check for missing features
    missing_features = expected_features_set - actual_features
    if missing_features:
        return False, f"Missing features: {sorted(missing_features)}"
    
    # Check for null values in expected features
    null_features = []
    for feature in expected_features:
        if feature in actual_features:
            null_count = df.select(pl.col(feature).is_null().sum()).item()
            if null_count > 0:
                null_features.append(f"{feature} ({null_count} nulls)")
    
    if null_features:
        return False, f"Features with null values: {null_features}"
    
    return True, f"All {len(expected_features)} features validated successfully"


def main():
    parser = argparse.ArgumentParser(description="Validate feature matrix against expected features")
    parser.add_argument("--features-list", required=True, help="Path to features list markdown file")
    parser.add_argument("--feature-table", required=True, help="Path to feature table parquet file")
    
    args = parser.parse_args()
    
    # Load expected features
    try:
        expected_features = load_expected_features(args.features_list)
        print(f"Loaded {len(expected_features)} expected features")
    except Exception as e:
        print(f"Error loading features list: {e}")
        sys.exit(1)
    
    # Validate feature table
    success, message = validate_feature_table(args.feature_table, expected_features)
    
    if success:
        print(f"✅ {message}")
        sys.exit(0)
    else:
        print(f"❌ {message}")
        sys.exit(1)


if __name__ == "__main__":
    main()