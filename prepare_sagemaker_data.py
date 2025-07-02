#!/usr/bin/env python3
"""
prepare_sagemaker_data.py - Prepare the combined dataset for AWS SageMaker training

This script:
1. Loads the combined dataset of all 56 symbols
2. Cleans date formats by:
   - Converting dates to string and stripping whitespace
   - Removing rows with "date" headers or blank dates
   - Properly parsing dates with error handling
3. Creates date-based train/validation/test splits (e.g., 70%/15%/15%)
4. Formats the data for SageMaker training
5. Uploads the prepared data to S3
"""

import pandas as pd
import numpy as np
import os
import argparse
import boto3
import logging
import sys
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_data(input_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Load the combined dataset and prepare it for training
    
    Args:
        input_file: Path to the combined features CSV file
        train_ratio: Percentage of data to use for training
        val_ratio: Percentage of data to use for validation
        test_ratio: Percentage of data to use for testing
        
    Returns:
        Dictionary containing the train, validation, and test dataframes
    """
    # Load the data
    logger.info(f"Loading data from {input_file}")
    try:
        # Check if file exists
        if not os.path.exists(input_file):
            logger.error(f"ERROR: File not found: {input_file}")
            sys.exit(f"ERROR: File not found: {input_file}")
            
        # Try to load the file
        try:
            df = pd.read_csv(input_file)
        except pd.errors.ParserError as e:
            logger.error(f"ERROR: CSV parsing failed: {e}")
            sys.exit(f"ERROR: CSV parsing failed: {e}")
        except MemoryError:
            logger.error("ERROR: Ran out of memory loading the data")
            sys.exit("ERROR: Ran out of memory loading the data. Consider chunked reads or increasing instance size.")
        except Exception as e:
            logger.error(f"ERROR: Failed to read CSV: {e}")
            sys.exit(f"ERROR: Failed to read CSV: {e}")
        
        # Clean and convert date to datetime
        if 'date' in df.columns:
            # Ensure date is string and strip whitespace
            df["date"] = df["date"].astype(str).str.strip()
            
            # Drop rows where date equals "date" (repeated header), blank, or "nan"
            mask_bad = df["date"].str.lower().isin(["date", "", "nan"])
            if mask_bad.sum() > 0:
                logger.warning(f"Dropped {mask_bad.sum()} rows with header 'date', blank values, or 'nan' strings")
                df = df[~mask_bad]
            
            # Parse dates with errors='coerce' to handle invalid formats
            df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")
            
            # Check remaining parse failures
            n_fail = df["parsed_date"].isna().sum()
            logger.info(f"Remaining parse failures after header/blank/nan drop: {n_fail}")
            
            if n_fail > 0:
                logger.warning(f"Dropping {n_fail} rows with unparseable date formats")
                df = df.dropna(subset=["parsed_date"])
            
            # Use the properly parsed date column
            df['date'] = df['parsed_date']
            df = df.drop(columns=['parsed_date'])
        else:
            logger.error(f"Date column not found in {input_file}")
            return None
        
        # Sort by date
        df = df.sort_values('date')
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        logger.info(f"Handling {df.isnull().sum().sum()} missing values")
        # Fill missing values or drop columns based on extent of missingness
        for col in df.columns:
            missing_pct = df[col].isnull().mean()
            if missing_pct > 0:
                if missing_pct > 0.3:  # If more than 30% missing, drop the column
                    logger.warning(f"Dropping column {col} with {missing_pct:.1%} missing values")
                    df = df.drop(columns=[col])
                else:  # Otherwise, fill with median or forward fill
                    if df[col].dtype in [np.float64, np.int64]:
                        logger.info(f"Filling missing values in {col} with median")
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        logger.info(f"Forward filling missing values in {col}")
                        df[col] = df[col].ffill()
    
    # Define features and target
    target_col = 'target_next_day'
    
    # Remove any rows where target is NA
    df = df.dropna(subset=[target_col])
    
    # Ensure target column is numeric
    try:
        print(f"🔢 Starting numeric conversion on {len(df)} rows")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        print("✅ Numeric conversion complete")
        df = df.dropna(subset=[target_col])
        logger.info(f"Converted target column to numeric. {len(df)} rows remaining.")
    except Exception as e:
        logger.error(f"Error converting target column to numeric: {e}")
        return None
    
    # Convert target to binary classification (0 for negative returns, 1 for positive returns)
    df['target_binary'] = (df[target_col] > 0).astype(int)
    
    # Drop columns that shouldn't be used for training
    cols_to_drop = ['timestamp', 'date', 'symbol', target_col]
    X = df.drop(columns=cols_to_drop + ['target_binary'])
    y = df['target_binary']
    
    # Determine split indices
    total_rows = len(df)
    train_idx = int(total_rows * train_ratio)
    val_idx = train_idx + int(total_rows * val_ratio)
    
    # Create the splits
    X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
    X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
    X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]
    
    # Get train/val/test date ranges for logging
    train_dates = df.iloc[:train_idx]['date']
    val_dates = df.iloc[train_idx:val_idx]['date']
    test_dates = df.iloc[val_idx:]['date']
    
    logger.info(f"Train data: {len(X_train)} samples from {train_dates.min().date()} to {train_dates.max().date()}")
    logger.info(f"Validation data: {len(X_val)} samples from {val_dates.min().date()} to {val_dates.max().date()}")
    logger.info(f"Test data: {len(X_test)} samples from {test_dates.min().date()} to {test_dates.max().date()}")
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to dataframes with column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Add target columns back
    X_train_scaled_df['target_binary'] = y_train.values
    X_val_scaled_df['target_binary'] = y_val.values
    X_test_scaled_df['target_binary'] = y_test.values
    
    # Rearrange to put target first (required for SageMaker XGBoost)
    train_df = X_train_scaled_df[['target_binary'] + [col for col in X_train_scaled_df.columns if col != 'target_binary']]
    val_df = X_val_scaled_df[['target_binary'] + [col for col in X_val_scaled_df.columns if col != 'target_binary']]
    test_df = X_test_scaled_df[['target_binary'] + [col for col in X_test_scaled_df.columns if col != 'target_binary']]
    
    # Return the prepared datasets
    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df,
        'feature_columns': X_train.columns.tolist(),
        'target_column': 'target_binary',
        'scaler': scaler
    }

def upload_to_s3(data_dict, s3_bucket, s3_prefix):
    """
    Upload the prepared data to S3
    
    Args:
        data_dict: Dictionary containing the prepared data
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix (folder) to upload to
        
    Returns:
        Dictionary containing the S3 URIs for the uploaded data
    """
    # Create a timestamp for the training job
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # Create S3 client
    s3_client = boto3.client('s3')
    
    # Validate S3 bucket
    try:
        s3_client.head_bucket(Bucket=s3_bucket)
    except Exception as e:
        logger.error(f"ERROR: S3 bucket validation failed: {e}")
        sys.exit(f"ERROR: S3 bucket validation failed: {e}")
    
    # Validate write permissions by creating a test object
    test_key = f"{s3_prefix}/test_write_permission_{timestamp}.txt"
    try:
        s3_client.put_object(Bucket=s3_bucket, Key=test_key, Body="test")
        s3_client.delete_object(Bucket=s3_bucket, Key=test_key)
    except Exception as e:
        logger.error(f"ERROR: Cannot write to S3 path s3://{s3_bucket}/{s3_prefix}: {e}")
        sys.exit(f"ERROR: Cannot write to S3 path s3://{s3_bucket}/{s3_prefix}: {e}")
    
    logger.info(f"Successfully validated S3 bucket and write permissions: s3://{s3_bucket}/{s3_prefix}")
    
    # Create local directory for saving files temporarily
    os.makedirs('data/sagemaker', exist_ok=True)
    
    # Save and upload each dataset
    s3_uris = {}
    
    for dataset_name, df in {k: v for k, v in data_dict.items() if k in ['train', 'validation', 'test']}.items():
        # Save locally
        local_path = f'data/sagemaker/{dataset_name}.csv'
        df.to_csv(local_path, index=False, header=False)  # Remove header for XGBoost
        
        # Upload to S3
        s3_key = f"{s3_prefix}/{timestamp}/{dataset_name}.csv"
        s3_client.upload_file(local_path, s3_bucket, s3_key)
        
        # Store the S3 URI
        s3_uris[dataset_name] = f"s3://{s3_bucket}/{s3_key}"
        
        logger.info(f"Uploaded {dataset_name} data to {s3_uris[dataset_name]}")
    
    # Save and upload feature metadata
    feature_metadata = {
        'feature_columns': data_dict['feature_columns'],
        'target_column': data_dict['target_column']
    }
    
    feature_metadata_path = 'data/sagemaker/feature_metadata.json'
    pd.Series(feature_metadata).to_json(feature_metadata_path)
    
    feature_metadata_key = f"{s3_prefix}/{timestamp}/feature_metadata.json"
    s3_client.upload_file(feature_metadata_path, s3_bucket, feature_metadata_key)
    s3_uris['feature_metadata'] = f"s3://{s3_bucket}/{feature_metadata_key}"
    
    # Save the scaler
    import joblib
    scaler_path = 'data/sagemaker/scaler.joblib'
    joblib.dump(data_dict['scaler'], scaler_path)
    
    scaler_key = f"{s3_prefix}/{timestamp}/scaler.joblib"
    s3_client.upload_file(scaler_path, s3_bucket, scaler_key)
    s3_uris['scaler'] = f"s3://{s3_bucket}/{scaler_key}"
    
    return s3_uris

def main():
    """Main function to prepare data and upload to S3"""
    parser = argparse.ArgumentParser(description='Prepare stock data for SageMaker training')
    parser.add_argument('--input-file', type=str, default='data/processed_features/sample_features.csv',
                        help='Path to the combined features CSV file')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Percentage of data to use for training')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Percentage of data to use for validation')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Percentage of data to use for testing')
    parser.add_argument('--s3-bucket', type=str, required=True,
                        help='S3 bucket to upload data to')
    parser.add_argument('--s3-prefix', type=str, default='stock-prediction',
                        help='S3 prefix (folder) to upload data to')
    
    args = parser.parse_args()
    
    # Prepare the data
    logger.info("Preparing data for SageMaker training")
    data_dict = prepare_data(
        args.input_file,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
    
    # Upload to S3
    logger.info(f"Uploading data to S3 bucket {args.s3_bucket}")
    s3_uris = upload_to_s3(data_dict, args.s3_bucket, args.s3_prefix)
    
    # Save the S3 URIs locally for reference
    with open('data/sagemaker/s3_uris.txt', 'w') as f:
        for dataset_name, uri in s3_uris.items():
            f.write(f"{dataset_name}: {uri}\n")
    
    logger.info(f"S3 URIs saved to data/sagemaker/s3_uris.txt")
    logger.info("Data preparation complete!")
    
    return s3_uris

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        logger.error(f"Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)
