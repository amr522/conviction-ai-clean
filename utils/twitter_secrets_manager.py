#!/usr/bin/env python3
"""
Twitter API Secrets Manager Integration
Secure retrieval of Twitter API credentials from AWS Secrets Manager with fallbacks
"""

import boto3
import json
import os
import logging
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class TwitterSecretsManager:
    """Manages Twitter API credentials with AWS Secrets Manager integration"""
    
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize Twitter Secrets Manager
        
        Args:
            region_name: AWS region for Secrets Manager
        """
        self.region_name = region_name
        self.secret_name = "twitter-api-credentials"
        self._secrets_client = None
        self._cached_credentials = None
        
    @property
    def secrets_client(self):
        """Lazy initialization of AWS Secrets Manager client"""
        if self._secrets_client is None:
            try:
                self._secrets_client = boto3.client(
                    'secretsmanager',
                    region_name=self.region_name
                )
            except NoCredentialsError:
                logger.warning("AWS credentials not configured. Falling back to environment variables.")
                self._secrets_client = None
        return self._secrets_client
    
    def get_twitter_credentials(self) -> Dict[str, str]:
        """
        Get Twitter API credentials with fallback strategy:
        1. Try AWS Secrets Manager
        2. Fall back to environment variables
        3. Raise error if neither available
        
        Returns:
            Dict containing Twitter API credentials
            
        Raises:
            ValueError: If credentials cannot be retrieved from any source
        """
        if self._cached_credentials:
            return self._cached_credentials
            
        credentials = self._get_from_secrets_manager()
        
        if not credentials:
            credentials = self._get_from_environment()
            
        if not credentials or not self._validate_credentials(credentials):
            raise ValueError(
                "Twitter API credentials not found or incomplete. "
                "Set them in AWS Secrets Manager or as environment variables:\n"
                "- TWITTER_BEARER_TOKEN\n"
                "- TWITTER_API_KEY\n"
                "- TWITTER_API_SECRET\n"
                "- TWITTER_ACCESS_TOKEN\n"
                "- TWITTER_ACCESS_TOKEN_SECRET"
            )
            
        self._cached_credentials = credentials
        logger.info("Twitter API credentials loaded successfully")
        return credentials
    
    def _get_from_secrets_manager(self) -> Optional[Dict[str, str]]:
        """
        Retrieve Twitter credentials from AWS Secrets Manager
        
        Returns:
            Dict of credentials or None if unavailable
        """
        if not self.secrets_client:
            return None
            
        try:
            logger.info(f"Attempting to retrieve Twitter credentials from AWS Secrets Manager: {self.secret_name}")
            
            response = self.secrets_client.get_secret_value(SecretId=self.secret_name)
            secret_string = response['SecretString']
            
            credentials = json.loads(secret_string)
            
            logger.info("Successfully retrieved Twitter credentials from AWS Secrets Manager")
            return credentials
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.warning(f"Secret {self.secret_name} not found in AWS Secrets Manager")
            elif error_code == 'InvalidRequestException':
                logger.warning(f"Invalid request to AWS Secrets Manager: {e}")
            elif error_code == 'InvalidParameterException':
                logger.warning(f"Invalid parameter for AWS Secrets Manager: {e}")
            else:
                logger.warning(f"AWS Secrets Manager error: {e}")
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Twitter credentials JSON from Secrets Manager: {e}")
            return None
            
        except Exception as e:
            logger.warning(f"Unexpected error retrieving from AWS Secrets Manager: {e}")
            return None
    
    def _get_from_environment(self) -> Optional[Dict[str, str]]:
        """
        Retrieve Twitter credentials from environment variables
        Following the pattern from aws_direct_access.py
        
        Returns:
            Dict of credentials or None if unavailable
        """
        logger.info("Attempting to retrieve Twitter credentials from environment variables")
        
        bearer_token = os.environ.get("TWITTER_BEARER_TOKEN", "")
        api_key = os.environ.get("TWITTER_API_KEY", "")
        api_secret = os.environ.get("TWITTER_API_SECRET", "")
        access_token = os.environ.get("TWITTER_ACCESS_TOKEN", "")
        access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET", "")
        
        if not any([bearer_token, api_key, api_secret, access_token, access_token_secret]):
            logger.warning("No Twitter credentials found in environment variables")
            return None
            
        credentials = {
            "bearer_token": bearer_token,
            "api_key": api_key,
            "api_secret": api_secret,
            "access_token": access_token,
            "access_token_secret": access_token_secret
        }
        
        logger.info("Retrieved Twitter credentials from environment variables")
        return credentials
    
    def _validate_credentials(self, credentials: Dict[str, str]) -> bool:
        """
        Validate that all required Twitter credentials are present
        
        Args:
            credentials: Dict of credential key-value pairs
            
        Returns:
            True if all required credentials are present and non-empty
        """
        required_keys = [
            "bearer_token",
            "api_key", 
            "api_secret",
            "access_token",
            "access_token_secret"
        ]
        
        for key in required_keys:
            if not credentials.get(key, "").strip():
                logger.error(f"Missing or empty Twitter credential: {key}")
                return False
                
        return True
    
    def create_secrets_manager_secret(self, credentials: Dict[str, str]) -> bool:
        """
        Create or update Twitter credentials in AWS Secrets Manager
        
        Args:
            credentials: Dict containing Twitter API credentials
            
        Returns:
            True if successful, False otherwise
        """
        if not self.secrets_client:
            logger.error("AWS Secrets Manager client not available")
            return False
            
        if not self._validate_credentials(credentials):
            logger.error("Invalid credentials provided")
            return False
            
        try:
            secret_value = json.dumps(credentials)
            
            try:
                self.secrets_client.update_secret(
                    SecretId=self.secret_name,
                    SecretString=secret_value
                )
                logger.info(f"Updated existing secret: {self.secret_name}")
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    self.secrets_client.create_secret(
                        Name=self.secret_name,
                        Description="Twitter API credentials for sentiment analysis",
                        SecretString=secret_value
                    )
                    logger.info(f"Created new secret: {self.secret_name}")
                else:
                    raise
                    
            self._cached_credentials = None
            return True
            
        except Exception as e:
            logger.error(f"Failed to create/update secret in AWS Secrets Manager: {e}")
            return False
    
    def test_credentials(self) -> bool:
        """
        Test if Twitter credentials can be retrieved successfully
        
        Returns:
            True if credentials are available and valid
        """
        try:
            credentials = self.get_twitter_credentials()
            return self._validate_credentials(credentials)
        except Exception as e:
            logger.error(f"Failed to test Twitter credentials: {e}")
            return False


def get_twitter_credentials(region_name: str = 'us-east-1') -> Dict[str, str]:
    """
    Convenience function to get Twitter credentials
    
    Args:
        region_name: AWS region for Secrets Manager
        
    Returns:
        Dict containing Twitter API credentials
        
    Raises:
        ValueError: If credentials cannot be retrieved
    """
    manager = TwitterSecretsManager(region_name=region_name)
    return manager.get_twitter_credentials()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        manager = TwitterSecretsManager()
        
        if manager.test_credentials():
            print("✅ Twitter credentials are available and valid")
            credentials = manager.get_twitter_credentials()
            
            for key, value in credentials.items():
                if value:
                    masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "*" * len(value)
                    print(f"   {key}: {masked_value}")
                else:
                    print(f"   {key}: <empty>")
        else:
            print("❌ Twitter credentials are not available or invalid")
            print("\nTo set up credentials:")
            print("1. AWS Secrets Manager (recommended):")
            print("   aws secretsmanager create-secret --name twitter-api-credentials --secret-string '{\"bearer_token\":\"...\",\"api_key\":\"...\",\"api_secret\":\"...\",\"access_token\":\"...\",\"access_token_secret\":\"...\"}'")
            print("\n2. Environment variables:")
            print("   export TWITTER_BEARER_TOKEN='...'")
            print("   export TWITTER_API_KEY='...'")
            print("   export TWITTER_API_SECRET='...'")
            print("   export TWITTER_ACCESS_TOKEN='...'")
            print("   export TWITTER_ACCESS_TOKEN_SECRET='...'")
            
    except Exception as e:
        print(f"❌ Error testing Twitter credentials: {e}")
