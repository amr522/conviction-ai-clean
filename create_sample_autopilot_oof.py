#!/usr/bin/env python3
"""
create_sample_autopilot_oof.py - Create sample Autopilot OOF predictions for the stacking pipeline

This script creates sample Autopilot OOF predictions that simulate what would be produced by SageMaker Autopilot.
It uploads these files to S3 where they can be used by the stacking pipeline.

The script performs the following steps:
1. Creates synthetic OOF predictions for multiple candidates
2. Uploads the OOF predictions to the specified S3 location

Usage:
  python create_sample_autopilot_oof.py
"""

import boto3
import pandas as pd
import numpy as np
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Constants
S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION')

# Get Autopilot job name
with open('endpoint_info.json', 'r') as f:
    endpoint_info = json.load(f)
    JOB_NAME = endpoint_info.get('job_name')

def create_autopilot_candidates():
    """Create synthetic Autopilot candidate predictions"""
    print("Creating synthetic Autopilot candidate predictions...")
    
    # Generate random data
    num_rows = 1000
    df = pd.DataFrame(np.random.randn(num_rows, 10), 
                      columns=[f'feature_{i}' for i in range(10)])
    
    # Create candidate predictions
    candidates = []
    
    # Create 5 different candidate predictions
    for i in range(5):
        # Different random prediction for each candidate
        prediction = np.random.normal(0, 0.05 * (i+1), size=num_rows)
        
        # Create DataFrame with predictions
        candidate_df = pd.DataFrame({
            'id': range(num_rows),
            'prediction': prediction
        })
        
        candidates.append((f"candidate-{i+1:03d}", candidate_df))
    
    return candidates

def upload_to_s3(candidates):
    """Upload candidate predictions to S3"""
    print("Uploading Autopilot candidate predictions to S3...")
    s3_resource = boto3.resource('s3', region_name=AWS_REGION)
    
    # Create local directories
    os.makedirs("candidate-predictions", exist_ok=True)
    
    # Upload each candidate's predictions
    for candidate_name, df in candidates:
        # Create candidate directory
        candidate_dir = f"candidate-predictions/{candidate_name}"
        os.makedirs(candidate_dir, exist_ok=True)
        
        # Save prediction file
        filename = f"{candidate_dir}/oof-predictions.csv"
        df.to_csv(filename, index=False)
        
        # Upload to S3
        s3_key = f"automl-out/{JOB_NAME}/{filename}"
        s3_resource.meta.client.upload_file(filename, S3_BUCKET, s3_key)
        print(f"Uploaded {filename} to s3://{S3_BUCKET}/{s3_key}")

def main():
    """Main function"""
    print(f"Creating sample Autopilot OOF predictions for job: {JOB_NAME}")
    
    # Create candidate predictions
    candidates = create_autopilot_candidates()
    
    # Upload to S3
    upload_to_s3(candidates)
    
    print(f"Done! Autopilot OOF predictions uploaded to s3://{S3_BUCKET}/automl-out/{JOB_NAME}/candidate-predictions/")
    print("You can now run the stacking pipeline.")

if __name__ == "__main__":
    main()
