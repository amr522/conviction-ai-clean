#!/usr/bin/env python3
"""
check_dataset_format.py - Script to check the format of datasets in S3 and convert if necessary

This script provides functions to validate if a dataset is in CSV format with UTF-8 encoding
and convert it from other formats (like Parquet) if needed.
"""

import os
import boto3
import logging
import pandas as pd
import io
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dataset_format(s3_uri, region_name='us-east-1'):
    """
    Check if the dataset at the given S3 URI is in CSV format with UTF-8 encoding.
    
    Args:
        s3_uri (str): The S3 URI of the dataset to check
        region_name (str): AWS region name
    
    Returns:
        bool: True if the dataset is in CSV format with UTF-8 encoding, False otherwise
    """
    # Extract bucket name and key from S3 URI
    if not s3_uri.startswith('s3://'):
        logger.error(f"Invalid S3 URI: {s3_uri}")
        return False
    
    parts = s3_uri[5:].split('/', 1)
    if len(parts) != 2:
        logger.error(f"Invalid S3 URI format: {s3_uri}")
        return False
    
    bucket_name = parts[0]
    prefix = parts[1]
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=region_name)
    
    try:
        # List objects with the given prefix
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' not in response:
            logger.error(f"No files found at {s3_uri}")
            return False
        
        # Check if there are any CSV files
        csv_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.csv')]
        
        if not csv_files:
            logger.warning(f"No CSV files found at {s3_uri}")
            return False
        
        # Check a sample CSV file for UTF-8 encoding
        sample_key = csv_files[0]['Key']
        logger.info(f"Checking encoding of {sample_key}")
        
        try:
            # Get the file content
            response = s3_client.get_object(Bucket=bucket_name, Key=sample_key)
            content = response['Body'].read()
            
            # Try to decode as UTF-8
            content.decode('utf-8')
            logger.info(f"File {sample_key} is in UTF-8 encoding")
            return True
        except UnicodeDecodeError:
            logger.warning(f"File {sample_key} is not in UTF-8 encoding")
            return False
    
    except ClientError as e:
        logger.error(f"Error checking dataset format: {str(e)}")
        return False

def convert_parquet_to_csv(input_s3_uri, output_s3_uri, region_name='us-east-1'):
    """
    Convert Parquet files to CSV format with UTF-8 encoding.
    
    Args:
        input_s3_uri (str): The S3 URI of the Parquet dataset
        output_s3_uri (str): The S3 URI where to save the CSV dataset
        region_name (str): AWS region name
    
    Returns:
        str: The S3 URI of the converted dataset, or None if conversion failed
    """
    # Extract bucket name and key from input S3 URI
    if not input_s3_uri.startswith('s3://'):
        logger.error(f"Invalid input S3 URI: {input_s3_uri}")
        return None
    
    input_parts = input_s3_uri[5:].split('/', 1)
    if len(input_parts) != 2:
        logger.error(f"Invalid input S3 URI format: {input_s3_uri}")
        return None
    
    input_bucket = input_parts[0]
    input_prefix = input_parts[1]
    
    # Extract bucket name and key from output S3 URI
    if not output_s3_uri.startswith('s3://'):
        logger.error(f"Invalid output S3 URI: {output_s3_uri}")
        return None
    
    output_parts = output_s3_uri[5:].split('/', 1)
    if len(output_parts) != 2:
        logger.error(f"Invalid output S3 URI format: {output_s3_uri}")
        return None
    
    output_bucket = output_parts[0]
    output_prefix = output_parts[1]
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=region_name)
    
    try:
        # List Parquet files in the input prefix
        response = s3_client.list_objects_v2(Bucket=input_bucket, Prefix=input_prefix)
        
        if 'Contents' not in response:
            logger.error(f"No files found at {input_s3_uri}")
            return None
        
        # Filter Parquet files
        parquet_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
        
        if not parquet_files:
            logger.warning(f"No Parquet files found at {input_s3_uri}")
            return None
        
        # Process each Parquet file
        for file_obj in parquet_files:
            parquet_key = file_obj['Key']
            
            # Create output key by replacing the extension
            if parquet_key.endswith('.parquet'):
                csv_key = output_prefix + '/' + os.path.basename(parquet_key)[:-8] + '.csv'
            else:
                csv_key = output_prefix + '/' + os.path.basename(parquet_key) + '.csv'
            
            logger.info(f"Converting {parquet_key} to {csv_key}")
            
            # Get the Parquet file
            response = s3_client.get_object(Bucket=input_bucket, Key=parquet_key)
            
            # Read Parquet data
            parquet_data = response['Body'].read()
            df = pd.read_parquet(io.BytesIO(parquet_data))
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')
            
            # Upload CSV file
            s3_client.put_object(
                Bucket=output_bucket,
                Key=csv_key,
                Body=csv_buffer.getvalue().encode('utf-8'),
                ContentType='text/csv'
            )
            
            logger.info(f"Successfully converted {parquet_key} to {csv_key}")
        
        return output_s3_uri
    
    except Exception as e:
        logger.error(f"Error converting Parquet to CSV: {str(e)}")
        return None

def ensure_csv_dataset(s3_uri, region_name='us-east-1'):
    """
    Ensure that the dataset at the given S3 URI is in CSV format with UTF-8 encoding.
    If not, try to convert it from Parquet.
    
    Args:
        s3_uri (str): The S3 URI of the dataset
        region_name (str): AWS region name
    
    Returns:
        str: The S3 URI of the CSV dataset, or None if conversion failed
    """
    # Check if the dataset is already in CSV format
    if check_dataset_format(s3_uri, region_name):
        logger.info(f"Dataset at {s3_uri} is already in CSV format with UTF-8 encoding")
        return s3_uri
    
    # If not, check if there are Parquet files
    # Extract bucket name and key from S3 URI
    parts = s3_uri[5:].split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1]
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=region_name)
    
    try:
        # List objects with the given prefix
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        
        if 'Contents' not in response:
            logger.error(f"No files found at {s3_uri}")
            return None
        
        # Check if there are any Parquet files
        parquet_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
        
        if not parquet_files:
            logger.warning(f"No Parquet files found at {s3_uri}")
            return None
        
        # Convert Parquet files to CSV
        output_s3_uri = s3_uri + '_csv'
        return convert_parquet_to_csv(s3_uri, output_s3_uri, region_name)
    
    except Exception as e:
        logger.error(f"Error ensuring CSV dataset: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check and convert dataset format')
    parser.add_argument('--input-s3-uri', type=str, required=True,
                        help='S3 URI of the input dataset')
    parser.add_argument('--output-s3-uri', type=str, default=None,
                        help='S3 URI where to save the converted dataset')
    parser.add_argument('--region', type=str, default='us-east-1',
                        help='AWS region (default: us-east-1)')
    
    args = parser.parse_args()
    
    if check_dataset_format(args.input_s3_uri, args.region):
        print(f"Dataset at {args.input_s3_uri} is in CSV format with UTF-8 encoding")
    else:
        print(f"Dataset at {args.input_s3_uri} is not in CSV format with UTF-8 encoding")
        
        if args.output_s3_uri:
            output_uri = convert_parquet_to_csv(args.input_s3_uri, args.output_s3_uri, args.region)
            if output_uri:
                print(f"Successfully converted dataset to CSV at {output_uri}")
            else:
                print("Failed to convert dataset to CSV")
