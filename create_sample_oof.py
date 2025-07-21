#!/usr/bin/env python3
"""
create_sample_oof.py - Create sample out-of-fold predictions for the stacking pipeline

This script creates sample OOF predictions that simulate what would be produced by a deep learning model.
It uploads these files to S3 where they can be used by the stacking pipeline.

The script performs the following steps:
1. Downloads the cleaned training dataset from S3
2. Creates synthetic OOF predictions for each record
3. Uploads the OOF predictions to the specified S3 location

Usage:
  python create_sample_oof.py
"""

import boto3
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Constants
S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION')
OOF_S3_PREFIX = f"s3://{S3_BUCKET}/conviction-ai/deep-model-oof/"

def download_training_data():
    """Download the cleaned training data from S3"""
    print("Downloading training data from S3...")
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    # Get training data path from the AutoML job
    try:
        with open('endpoint_info.json', 'r') as f:
            import json
            endpoint_info = json.load(f)
            job_name = endpoint_info.get('job_name')
            
        # Construct training data path based on job name
        training_data_key = f"conviction-ai/processed/{job_name}-training-data.csv"
        local_file = "temp_training_data.csv"
        
        s3_client.download_file(S3_BUCKET, training_data_key, local_file)
        return pd.read_csv(local_file)
    except Exception as e:
        print(f"Error downloading training data: {e}")
        print("Using random data instead...")
        # Generate random data if download fails
        return pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])

def create_oof_predictions(data):
    """Create synthetic OOF predictions"""
    print("Creating synthetic OOF predictions...")
    
    # Keep the data but drop the target column if it exists
    X = data.drop(['return'], axis=1) if 'return' in data.columns else data
    
    # Create three different OOF predictions to simulate different models
    # Model 1: Linear prediction with random noise
    oof_1 = X.iloc[:, :5].mean(axis=1) * 0.01 + np.random.normal(0, 0.05, size=len(X))
    
    # Model 2: Non-linear prediction with different random noise
    oof_2 = (X.iloc[:, 5:].mean(axis=1) ** 2) * 0.005 + np.random.normal(0, 0.03, size=len(X))
    
    # Model 3: Another variation
    oof_3 = np.sin(X.iloc[:, :3].mean(axis=1)) * 0.02 + np.random.normal(0, 0.04, size=len(X))
    
    # Create DataFrames with predictions
    oof_df_1 = pd.DataFrame({'id': range(len(X)), 'prediction': oof_1})
    oof_df_2 = pd.DataFrame({'id': range(len(X)), 'prediction': oof_2})
    oof_df_3 = pd.DataFrame({'id': range(len(X)), 'prediction': oof_3})
    
    return oof_df_1, oof_df_2, oof_df_3

def upload_to_s3(oof_dfs):
    """Upload OOF predictions to S3"""
    print("Uploading OOF predictions to S3...")
    s3_resource = boto3.resource('s3', region_name=AWS_REGION)
    
    # Upload each OOF DataFrame
    for i, df in enumerate(oof_dfs, 1):
        filename = f"deep_model_oof_{i}.csv"
        df.to_csv(filename, index=False)
        
        # Upload to S3
        s3_key = f"conviction-ai/deep-model-oof/{filename}"
        s3_resource.meta.client.upload_file(filename, S3_BUCKET, s3_key)
        print(f"Uploaded {filename} to s3://{S3_BUCKET}/{s3_key}")
        
        # Clean up local file
        os.remove(filename)

def main():
    """Main function"""
    print("Creating sample OOF predictions for stacking pipeline...")
    
    # Download training data
    data = download_training_data()
    
    # Create OOF predictions
    oof_dfs = create_oof_predictions(data)
    
    # Upload to S3
    upload_to_s3(oof_dfs)
    
    # Clean up
    if os.path.exists("temp_training_data.csv"):
        os.remove("temp_training_data.csv")
    
    print(f"Done! OOF predictions uploaded to {OOF_S3_PREFIX}")
    print("You can now run the stacking pipeline.")

if __name__ == "__main__":
    main()
