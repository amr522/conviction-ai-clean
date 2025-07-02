#!/usr/bin/env python3

import boto3
import yaml
import os
from botocore.client import Config

# Load config from file or environment variables
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except:
    # Fallback to environment variables if config file not available
    config = {
        "flat_access_key": os.environ.get("POLYGON_ACCESS_KEY", ""),
        "flat_secret_key": os.environ.get("POLYGON_SECRET_KEY", ""),
    }

def setup_aws_direct_access():
    """Setup direct access to Polygon S3 data from AWS"""
    
    # Get Polygon credentials from config or environment
    polygon_access_key = config.get("flat_access_key", "") or os.environ.get("POLYGON_ACCESS_KEY", "")
    polygon_secret_key = config.get("flat_secret_key", "") or os.environ.get("POLYGON_SECRET_KEY", "")
    
    # Ensure credentials are provided
    if not polygon_access_key or not polygon_secret_key:
        raise ValueError("Polygon API credentials not found. Set them in config.yaml or as environment variables.")
    
    # Polygon S3 credentials
    polygon_s3 = boto3.client(
        's3',
        aws_access_key_id=polygon_access_key,
        aws_secret_access_key=polygon_secret_key,
        endpoint_url="https://files.polygon.io",
        config=Config(signature_version='s3v4'),
        verify=False
    )
    
    # Your AWS S3 (for training)
    aws_s3 = boto3.client('s3')
    
    print("🚀 Direct AWS-to-AWS data transfer setup")
    print("📦 Polygon S3 bucket: flatfiles")
    print("☁️ Your S3 bucket:", config["s3_bucket"])
    
    # List available data paths
    paths = [
        "us_stocks_sip/day_aggs_v1/",
        "us_options/day_aggs_v1/", 
        "us_indices/day_aggs_v1/",
        "global_forex/day_aggs_v1/",
        "global_crypto/day_aggs_v1/"
    ]
    
    for path in paths:
        try:
            response = polygon_s3.list_objects_v2(
                Bucket="flatfiles", 
                Prefix=path, 
                MaxKeys=5
            )
            count = len(response.get('Contents', []))
            print(f"✅ {path}: {count} files available")
        except Exception as e:
            print(f"❌ {path}: {str(e)}")
    
    return polygon_s3, aws_s3

def copy_data_aws_to_aws(source_path, dest_bucket, dest_prefix="polygon-data/"):
    """Copy data directly from Polygon S3 to your AWS S3"""
    polygon_s3, aws_s3 = setup_aws_direct_access()
    
    # List files in source path
    response = polygon_s3.list_objects_v2(Bucket="flatfiles", Prefix=source_path)
    
    for obj in response.get('Contents', []):
        source_key = obj['Key']
        dest_key = f"{dest_prefix}{source_key}"
        
        # Direct S3-to-S3 copy (fastest method)
        copy_source = {'Bucket': 'flatfiles', 'Key': source_key}
        aws_s3.copy(copy_source, dest_bucket, dest_key)
        print(f"📋 Copied: {source_key}")

if __name__ == "__main__":
    setup_aws_direct_access()
    
    print("\n💡 For AWS training, use these methods:")
    print("1. Direct S3 access in training script:")
    print("   s3://flatfiles/us_stocks_sip/day_aggs_v1/")
    print("2. Copy to your bucket:")
    print("   copy_data_aws_to_aws('us_stocks_sip/day_aggs_v1/', 'your-bucket')")