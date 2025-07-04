#!/usr/bin/env python3
"""Sample inference test for deployed HPO model endpoint"""

import boto3
import json
import csv
import io
from sagemaker.predictor import Predictor

def test_endpoint_inference():
    endpoint_name = "conviction-hpo-20250704-064322"
    
    # Create sample CSV data (matching training format)
    sample_data = "1.0,0.5,0.3,0.8,0.2,0.7,0.4,0.9,0.1,0.6"
    
    try:
        # Test with SageMaker Predictor
        predictor = Predictor(endpoint_name=endpoint_name)
        
        print(f"🔍 Testing endpoint: {endpoint_name}")
        print(f"📊 Sample input: {sample_data}")
        
        # Send prediction request
        result = predictor.predict(sample_data)
        
        print(f"✅ Prediction result: {result}")
        print("🎉 Endpoint inference test successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_endpoint_inference()
