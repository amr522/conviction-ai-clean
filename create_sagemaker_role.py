#!/usr/bin/env python3
import boto3
import json
import os
from os import environ

# AWS Credentials - Use environment variables
# AWS credentials should be set up via:
# - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
# - AWS credentials file (~/.aws/credentials)
# - IAM Role attached to the instance (if running on EC2/ECS)
# NEVER hardcode credentials in source code
ACCOUNT_ID = environ.get('AWS_ACCOUNT_ID', '')  # Get from env var or update with your account
REGION = environ.get('AWS_REGION', 'us-east-1')

def create_sagemaker_role():
    """Create SageMaker execution role"""
    
    # Create IAM client - will use credentials from environment or credential file
    iam = boto3.client('iam', region_name=REGION)
    
    role_name = 'SageMakerExecutionRole'
    
    # Trust policy for SageMaker
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    print("🔧 Creating SageMaker execution role...")
    
    try:
        # Create role
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='SageMaker execution role for HPO training'
        )
        
        role_arn = response['Role']['Arn']
        print(f"✅ Created role: {role_arn}")
        
        # Attach required policies
        policies = [
            'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess'
        ]
        
        for policy in policies:
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy
            )
            print(f"✅ Attached policy: {policy}")
        
        print(f"\n🎯 Role ARN: {role_arn}")
        return role_arn
        
    except Exception as e:
        if 'EntityAlreadyExists' in str(e):
            print("⚠️ Role already exists")
            role_arn = f"arn:aws:iam::{ACCOUNT_ID}:role/{role_name}"
            print(f"🎯 Using existing role: {role_arn}")
            return role_arn
        else:
            print(f"❌ Error creating role: {e}")
            return None

if __name__ == "__main__":
    create_sagemaker_role()