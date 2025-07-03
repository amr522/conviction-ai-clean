#!/usr/bin/env python3
"""
Quick script to examine CSV schema without loading entire file
"""

import pandas as pd
import sys

def examine_csv(file_path, num_rows=5):
    """Examine first few rows of a CSV file"""
    print(f"Examining: {file_path}")
    
    try:
        df = pd.read_csv(file_path, nrows=num_rows)
        
        print(f"Shape (first {num_rows} rows): {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        if 'direction' in df.columns:
            print(f"\n✅ 'direction' column found!")
            print(f"Direction values in sample: {df['direction'].unique()}")
        else:
            print(f"\n❌ 'direction' column NOT found!")
            
        target_cols = [col for col in df.columns if 'target' in col.lower()]
        if target_cols:
            print(f"Target columns found: {target_cols}")
        
        return True
        
    except Exception as e:
        print(f"Error examining {file_path}: {e}")
        return False

if __name__ == "__main__":
    files_to_check = [
        "data/sagemaker_input/46_models/2025-07-02-03-05-02/train.csv",
        "data/sagemaker_input/46_models/2025-07-02-03-05-02/validation.csv",
        "data/sagemaker_input/46_models/2025-07-02-03-05-02/test.csv"
    ]
    
    for file_path in files_to_check:
        examine_csv(file_path)
        print("-" * 60)
