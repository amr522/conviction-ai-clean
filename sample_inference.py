#!/usr/bin/env python3
"""
Sample inference script to test SageMaker endpoint
"""
import boto3
import json
import numpy as np
import pandas as pd
import argparse

def test_endpoint_inference(endpoint_name, sample_count=5):
    """Test endpoint with sample data"""
    print(f"🧪 Testing endpoint: {endpoint_name}")
    
    runtime = boto3.client('sagemaker-runtime')
    
    np.random.seed(42)
    sample_data = np.random.randn(sample_count, 63)
    
    print(f"📊 Testing with {sample_count} samples, {sample_data.shape[1]} features")
    
    try:
        for i in range(sample_count):
            sample_row = sample_data[i]
            payload = ','.join([str(val) for val in sample_row])
            
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='text/csv',
                Body=payload
            )
            
            result = json.loads(response['Body'].read().decode())
            prediction = result.get('predictions', [])
            
            print(f"  Sample {i+1}: Prediction = {prediction}")
            
        print("✅ All inference tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test SageMaker endpoint inference')
    parser.add_argument('--endpoint-name', type=str, 
                        default='conviction-hpo-20250704-064322',
                        help='Name of SageMaker endpoint to test')
    parser.add_argument('--sample-count', type=int, default=5,
                        help='Number of samples to test')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode without making actual inference calls')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🧪 DRY RUN MODE - No inference calls will be made")
        print(f"✅ DRY RUN: Would test endpoint: {args.endpoint_name}")
        print(f"✅ DRY RUN: Would run {args.sample_count} inference samples")
        print("✅ DRY RUN: Would check endpoint status and invoke with sample data")
        return 0
    
    sagemaker = boto3.client('sagemaker')
    try:
        status = sagemaker.describe_endpoint(EndpointName=args.endpoint_name)
        endpoint_status = status['EndpointStatus']
        print(f"📊 Endpoint status: {endpoint_status}")
        
        if endpoint_status != 'InService':
            print(f"⚠️ Endpoint not ready for inference (status: {endpoint_status})")
            print("Please wait for endpoint to reach 'InService' status")
            return 1
            
    except Exception as e:
        print(f"❌ Could not check endpoint status: {e}")
        return 1
    
    success = test_endpoint_inference(args.endpoint_name, args.sample_count)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
