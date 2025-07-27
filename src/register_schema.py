#!/usr/bin/env python3
"""Register JSON schema in AWS Glue Schema Registry."""

import json
import os
from pathlib import Path

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def register_schema_in_glue():
    """Register feature schema in AWS Glue Schema Registry."""
    if not BOTO3_AVAILABLE:
        print("⚠️  boto3 not available, skipping AWS Glue registration")
        return
    
    schema_path = Path("schemas/feature_schema.json")
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return
    
    schema_content = schema_path.read_text()
    
    try:
        client = boto3.client('glue', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        
        response = client.create_schema(
            RegistryId={'RegistryName': 'conviction-ai-registry'},
            SchemaName='feature-schema',
            DataFormat='JSON',
            SchemaDefinition=schema_content,
            Description='Feature schema for conviction-ai pipeline'
        )
        
        print(f"✅ Schema registered in AWS Glue: {response['SchemaArn']}")
        
    except client.exceptions.AlreadyExistsException:
        print("✅ Schema already exists in AWS Glue")
    except Exception as e:
        print(f"⚠️  Failed to register schema in AWS Glue: {e}")


if __name__ == "__main__":
    register_schema_in_glue()