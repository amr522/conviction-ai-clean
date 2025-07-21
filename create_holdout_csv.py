#!/usr/bin/env python3
"""
create_holdout_csv.py - Create a holdout dataset from a Parquet file in S3
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import boto3
from dotenv import load_dotenv
import tempfile
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# S3 path to the Parquet file
S3_BUCKET = "sagemaker-us-east-1-773934887314"
S3_KEY = "conviction-ai/clean/train_dataset/part-00000-ab018799-568c-4eee-a445-74fb49d04e9e-c000.snappy.parquet"
OUTPUT_PATH = "data/holdout_2023.csv"
MAX_ROWS = 2000
CUTOFF_DATE = "2023-01-01"

def main():
    try:
        # Create boto3 session
        session = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )
        
        # Create S3 client
        s3_client = session.client('s3')
        
        # Create temporary file to download the Parquet file
        with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp_file:
            logger.info(f"Downloading Parquet file from s3://{S3_BUCKET}/{S3_KEY}")
            s3_client.download_file(S3_BUCKET, S3_KEY, tmp_file.name)
            
            # Read the Parquet file
            logger.info("Reading Parquet file")
            df = pq.read_table(tmp_file.name).to_pandas()
            
            logger.info(f"Total rows in dataset: {len(df)}")
            
            # Check if timestamp column exists
            if 'timestamp' not in df.columns:
                logger.warning("'timestamp' column not found in dataset. Looking for date-like columns...")
                date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                
                if date_columns:
                    logger.info(f"Found potential date columns: {date_columns}")
                    timestamp_col = date_columns[0]
                    logger.info(f"Using '{timestamp_col}' as timestamp column")
                else:
                    logger.warning("No date-like columns found. Skipping date filtering.")
                    timestamp_col = None
            else:
                timestamp_col = 'timestamp'
                
            # Filter by date if timestamp column exists
            if timestamp_col:
                logger.info(f"Filtering rows where {timestamp_col} >= {CUTOFF_DATE}")
                
                # Convert timestamp column to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
                # Filter rows
                df_filtered = df[df[timestamp_col] >= CUTOFF_DATE]
                logger.info(f"Rows after filtering: {len(df_filtered)}")
                
                # Take first MAX_ROWS rows
                df_holdout = df_filtered.head(MAX_ROWS)
            else:
                # Just take the first MAX_ROWS rows if no timestamp column
                logger.info(f"Taking first {MAX_ROWS} rows from dataset (no date filtering)")
                df_holdout = df.head(MAX_ROWS)
            
            logger.info(f"Holdout dataset size: {len(df_holdout)}")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            
            # Save to CSV
            logger.info(f"Saving holdout dataset to {OUTPUT_PATH}")
            df_holdout.to_csv(OUTPUT_PATH, index=False)
            
            # Identify potential target columns (look for columns with target, label, or price in the name)
            potential_targets = [col for col in df_holdout.columns if any(keyword in col.lower() for keyword in ['target', 'label', 'price', 'return', 'output'])]
            if potential_targets:
                logger.info(f"Potential target columns: {potential_targets}")
            
            logger.info(f"Columns in dataset: {df_holdout.columns.tolist()}")
            logger.info(f"Holdout dataset created successfully with {len(df_holdout)} rows and {len(df_holdout.columns)} columns")
            
            return True
            
    except Exception as e:
        logger.error(f"Error creating holdout dataset: {str(e)}")
        return False

if __name__ == "__main__":
    main()
