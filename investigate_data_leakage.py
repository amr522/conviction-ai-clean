import pandas as pd
import numpy as np
import os
from datetime import datetime

def investigate_training_data():
    """Investigate potential data leakage in training data"""
    
    print("🔍 DATA LEAKAGE INVESTIGATION")
    print("=" * 60)
    
    print("\n1. ANALYZING S3 TRAINING DATA:")
    s3_data_path = "/tmp/hpo_train_data.csv"
    
    if os.path.exists(s3_data_path):
        df_s3 = pd.read_csv(s3_data_path)
        print(f"S3 Training Data Shape: {df_s3.shape}")
        print(f"Columns: {list(df_s3.columns)}")
        
        if 'direction' in df_s3.columns:
            print(f"✅ Direction column found: {df_s3['direction'].value_counts().to_dict()}")
        else:
            print("❌ Direction column MISSING from S3 training data")
        
        leakage_columns = ['target_1d', 'target_3d', 'target_5d', 'target_10d']
        found_leakage = [col for col in leakage_columns if col in df_s3.columns]
        if found_leakage:
            print(f"⚠️ POTENTIAL LEAKAGE COLUMNS FOUND: {found_leakage}")
        else:
            print("✅ No obvious target leakage columns in S3 data")
        
        if len(df_s3.columns) > 1:
            target_col = df_s3.columns[-1]  # Assume last column is target
            print(f"Target column '{target_col}' distribution:")
            if df_s3[target_col].dtype in ['int64', 'float64']:
                print(f"  Mean: {df_s3[target_col].mean():.4f}")
                print(f"  Std: {df_s3[target_col].std():.4f}")
                print(f"  Min: {df_s3[target_col].min():.4f}")
                print(f"  Max: {df_s3[target_col].max():.4f}")
                
                unique_vals = df_s3[target_col].unique()
                if len(unique_vals) == 2:
                    print(f"  Binary target: {sorted(unique_vals)}")
                    print(f"  Class balance: {df_s3[target_col].value_counts().to_dict()}")
    
    print("\n2. ANALYZING LOCAL CSV FILES:")
    local_data_dir = "/home/ubuntu/repos/conviction-ai-clean/data/processed_with_news_20250628"
    
    if os.path.exists(local_data_dir):
        csv_files = [f for f in os.listdir(local_data_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} local CSV files")
        
        if csv_files:
            sample_file = os.path.join(local_data_dir, csv_files[0])
            df_local = pd.read_csv(sample_file)
            
            print(f"\nSample file: {csv_files[0]}")
            print(f"Shape: {df_local.shape}")
            print(f"Columns: {list(df_local.columns)}")
            
            if 'direction' in df_local.columns:
                print(f"✅ Direction column found: {df_local['direction'].value_counts().to_dict()}")
            else:
                print("❌ Direction column MISSING from local data")
            
            found_leakage = [col for col in leakage_columns if col in df_local.columns]
            if found_leakage:
                print(f"⚠️ LEAKAGE COLUMNS FOUND: {found_leakage}")
                
                if 'target_1d' in df_local.columns and 'close' in df_local.columns:
                    df_local['actual_next_return'] = df_local['close'].shift(-1) / df_local['close'] - 1
                    df_local['actual_direction'] = (df_local['actual_next_return'] > 0).astype(int)
                    
                    if not df_local['target_1d'].isna().all():
                        corr_continuous = df_local['target_1d'].corr(df_local['actual_next_return'])
                        print(f"  Correlation target_1d vs actual returns: {corr_continuous:.4f}")
                        
                        if 'direction' in df_local.columns:
                            corr_binary = df_local['direction'].corr(df_local['actual_direction'])
                            print(f"  Correlation direction vs actual direction: {corr_binary:.4f}")
                        
                        if 'close' in df_local.columns:
                            shifted_close_return = df_local['close'].shift(-1) / df_local['close'] - 1
                            corr_shifted = df_local['target_1d'].corr(shifted_close_return)
                            print(f"  Correlation target_1d vs shifted close returns: {corr_shifted:.4f}")
                            
                            if abs(corr_shifted) > 0.99:
                                print("  🚨 SEVERE DATA LEAKAGE: target_1d appears to be future returns!")
            
            if 'date' in df_local.columns:
                df_local['date'] = pd.to_datetime(df_local['date'])
                date_range = f"{df_local['date'].min()} to {df_local['date'].max()}"
                print(f"Date range: {date_range}")
                
                is_sorted = df_local['date'].is_monotonic_increasing
                print(f"Dates properly sorted: {is_sorted}")
    
    print("\n3. DATA SOURCE COMPARISON:")
    print(f"Local CSV files: {len(csv_files) if 'csv_files' in locals() else 0}")
    print(f"S3 training files: 3 (aapl_simple_train.csv, aapl_train.csv, train.csv)")
    print(f"HPO job claimed: 46 stocks")
    
    if 'csv_files' in locals() and len(csv_files) != 46:
        print(f"⚠️ MISMATCH: Local files ({len(csv_files)}) != HPO claim (46 stocks)")
    
    print("\n4. RECOMMENDATIONS:")
    print("Based on the analysis above:")
    
    if 'found_leakage' in locals() and found_leakage:
        print("🚨 HIGH PRIORITY: Data leakage detected")
        print("  - Remove target_1d, target_3d, target_5d, target_10d columns")
        print("  - Regenerate targets using proper temporal alignment")
        print("  - Re-run HPO with clean data")
    
    print("✅ Verify target generation process")
    print("✅ Implement proper train/validation/test splits with temporal ordering")
    print("✅ Add automated data leakage detection to pipeline")
    
    return True

if __name__ == "__main__":
    investigate_training_data()
