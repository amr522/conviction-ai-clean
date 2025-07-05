#!/usr/bin/env python3
"""
AWS Utilities for Secure Secret Management
"""

import json
import boto3
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_secret(secret_name: str, region_name: str = 'us-east-1') -> Optional[Dict[str, Any]]:
    """
    Retrieve secret from AWS Secrets Manager
    
    Args:
        secret_name: Name of the secret in AWS Secrets Manager
        region_name: AWS region name
        
    Returns:
        Dictionary containing secret values or None if failed
    """
    try:
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        
        response = client.get_secret_value(SecretId=secret_name)
        secret_string = response['SecretString']
        
        return json.loads(secret_string)
        
    except Exception as e:
        logger.error(f"Failed to retrieve secret {secret_name}: {e}")
        return None

def store_secret(secret_name: str, secret_dict: Dict[str, Any], region_name: str = 'us-east-1') -> bool:
    """
    Store secret in AWS Secrets Manager
    
    Args:
        secret_name: Name of the secret in AWS Secrets Manager
        secret_dict: Dictionary containing secret key-value pairs
        region_name: AWS region name
        
    Returns:
        True if successful, False otherwise
    """
    try:
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        
        secret_string = json.dumps(secret_dict)
        
        try:
            client.create_secret(
                Name=secret_name,
                SecretString=secret_string,
                Description=f'API keys for {secret_name}'
            )
            logger.info(f"Created new secret: {secret_name}")
        except client.exceptions.ResourceExistsException:
            client.update_secret(
                SecretId=secret_name,
                SecretString=secret_string
            )
            logger.info(f"Updated existing secret: {secret_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to store secret {secret_name}: {e}")
        return False

def get_api_keys() -> Dict[str, str]:
    """
    Get all API keys from AWS Secrets Manager with fallback to environment variables
    """
    import os
    
    api_keys = {}
    
    secrets_to_retrieve = [
        'twitter-api-keys',
        'polygon-api-keys', 
        'xai-api-keys',
        'fred-api-keys'
    ]
    
    for secret_name in secrets_to_retrieve:
        secret_data = get_secret(secret_name)
        if secret_data:
            api_keys.update(secret_data)
        else:
            logger.warning(f"Failed to retrieve {secret_name} from Secrets Manager")
    
    env_fallbacks = {
        'TWITTER_BEARER_TOKEN': os.environ.get('TWITTER_BEARER_TOKEN'),
        'POLYGON_API_KEY': os.environ.get('POLYGON_API_KEY'),
        'XAI_API_KEY': os.environ.get('XAI_API_KEY'),
        'FRED_API_KEY': os.environ.get('FRED_API_KEY')
    }
    
    for key, value in env_fallbacks.items():
        if value and key.lower() not in api_keys:
            api_keys[key.lower()] = value
            logger.info(f"Using environment variable for {key}")
    
    return api_keys
