#!/usr/bin/env python3
"""Deploy the best model from pinned HPO artifacts to SageMaker endpoint"""

import os
import sys
import json
import argparse
import tempfile
import tarfile
import boto3
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_xgboost_inference_script():
    """Create inference script for XGBoost model"""
    return '''
import json
import pickle
import numpy as np
import xgboost as xgb

def model_fn(model_dir):
    """Load the XGBoost model"""
    model_path = os.path.join(model_dir, "xgboost-model")
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == "text/csv":
        return np.array([float(x) for x in request_body.strip().split(",")])
    elif request_content_type == "application/json":
        input_data = json.loads(request_body)
        return np.array(input_data["instances"])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    dmatrix = xgb.DMatrix(input_data.reshape(1, -1))
    prediction = model.predict(dmatrix)
    return prediction[0]

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == "application/json":
        return json.dumps({"prediction": float(prediction)})
    else:
        return str(prediction)
'''

def deploy_xgboost_model(model_file, endpoint_name):
    """Deploy XGBoost model to SageMaker endpoint"""
    print(f"🚀 Deploying XGBoost model to endpoint: {endpoint_name}")
    
    sagemaker_client = boto3.client('sagemaker')
    s3_client = boto3.client('s3')
    
    bucket = 'hpo-bucket-773934887314'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(model_file, 'r:gz') as tar:
            tar.extractall(temp_dir)
        
        inference_script = create_xgboost_inference_script()
        with open(os.path.join(temp_dir, 'inference.py'), 'w') as f:
            f.write(inference_script)
        
        model_tar_path = os.path.join(temp_dir, 'model.tar.gz')
        with tarfile.open(model_tar_path, 'w:gz') as tar:
            tar.add(temp_dir, arcname='.')
        
        s3_key = f"models/best-hpo/{timestamp}/model.tar.gz"
        s3_client.upload_file(model_tar_path, bucket, s3_key)
        model_uri = f"s3://{bucket}/{s3_key}"
        
        print(f"📤 Model uploaded to: {model_uri}")
        
        model_name = f"best-hpo-model-{timestamp}"
        config_name = f"best-hpo-config-{timestamp}"
        
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3',
                'ModelDataUrl': model_uri,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                }
            },
            ExecutionRoleArn='arn:aws:iam::773934887314:role/SageMakerExecutionRole'
        )
        
        sagemaker_client.create_endpoint_configuration(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.t2.medium',
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        
        print(f"✅ Endpoint creation initiated: {endpoint_name}")
        print("⏳ Endpoint is being created... This may take several minutes.")
        
        return {
            'endpoint_name': endpoint_name,
            'model_name': model_name,
            'config_name': config_name,
            'model_uri': model_uri
        }

def extract_and_deploy_pinned_model(endpoint_name="conviction-best-model"):
    """Extract and deploy the pinned best model"""
    pinned_dir = "models/pinned_successful_hpo"
    config_file = os.path.join(pinned_dir, "hpo_config_pinned.json")
    model_file = None
    
    for file in os.listdir(pinned_dir):
        if file.endswith(".tar.gz") and "best_model" in file:
            model_file = os.path.join(pinned_dir, file)
            break
    
    if not model_file or not os.path.exists(model_file):
        print("❌ Best model artifact not found in pinned directory")
        return None
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"🚀 Deploying best model from HPO job: {config['successful_hpo_job']}")
    print(f"📊 Model achieved validation AUC: {config['validation_auc']}")
    
    result = deploy_xgboost_model(model_file, endpoint_name)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deploy best HPO model to SageMaker endpoint')
    parser.add_argument('--endpoint-name', default='conviction-best-model', help='Endpoint name')
    args = parser.parse_args()
    
    result = extract_and_deploy_pinned_model(args.endpoint_name)
    if result:
        print("✅ Best model deployment completed successfully!")
    else:
        print("❌ Deployment failed")
        sys.exit(1)
