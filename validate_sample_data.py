#!/usr/bin/env python3
"""
Dataset sanity check script for pinned HPO dataset
"""

import pandas as pd
import sys
import os

def validate_sample_data(file_path):
    """Validate the downloaded sample data"""
    print("🔍 Dataset Sanity Check")
    print("=" * 40)
    
    if not os.path.exists(file_path):
        print(f"❌ Sample file not found: {file_path}")
        return False
    
    try:
        df = pd.read_csv(file_path)
        
        print(f"📊 File: {file_path}")
        print(f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"💾 File size: {os.path.getsize(file_path):,} bytes")
        
        null_columns = df.columns[df.isna().all()].tolist()
        if null_columns:
            print(f"⚠️ All-null columns detected: {null_columns}")
            return False
        else:
            print("✅ No all-null columns detected")
        
        print(f"📋 Data types:")
        for dtype, count in df.dtypes.value_counts().items():
            print(f"  - {dtype}: {count} columns")
        
        missing_pct = (df.isna().sum() / len(df) * 100).round(2)
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            print(f"⚠️ Columns with >50% missing values:")
            for col, pct in high_missing.items():
                print(f"  - {col}: {pct}%")
        
        print(f"📋 First 3 rows preview:")
        print(df.head(3).to_string())
        
        target_cols = [col for col in df.columns if 'target' in col.lower() or 'direction' in col.lower()]
        if target_cols:
            print(f"🎯 Target columns found: {target_cols}")
            for col in target_cols:
                unique_vals = df[col].nunique()
                print(f"  - {col}: {unique_vals} unique values")
        else:
            print("⚠️ No obvious target columns found")
        
        print("✅ Sample data validation complete")
        return True
        
    except Exception as e:
        print(f"❌ Error reading sample data: {e}")
        return False

if __name__ == "__main__":
    file_path = "sample_train.csv"
    success = validate_sample_data(file_path)
    sys.exit(0 if success else 1)
