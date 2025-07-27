#!/usr/bin/env python3
"""Register JSON schema in AWS Glue Schema Registry with versioning support."""

import argparse
import json
import sys
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception


def load_schema(path):
    """Load JSON schema from file."""
    with open(path) as f:
        return json.load(f)


def register_schema(registry, name, schema_path, compatibility):
    """Register or update schema in AWS Glue Schema Registry."""
    if not BOTO3_AVAILABLE:
        print("⚠️  boto3 not available, skipping AWS Glue registration")
        return
    
    client = boto3.client("glue")
    schema_def = load_schema(schema_path)
    
    try:
        # Try to create new schema
        response = client.create_schema(
            RegistryId={"RegistryName": registry},
            SchemaName=name,
            DataFormat="JSON",
            Compatibility=compatibility,
            SchemaDefinition=json.dumps(schema_def),
            Description=f"Feature schema for {name} with {compatibility} compatibility"
        )
        print(f"✅ Registered new schema {name} in {registry}")
        print(f"   Schema ARN: {response.get('SchemaArn', 'N/A')}")
        
    except ClientError as e:
        if e.response["Error"]["Code"] == "AlreadyExistsException":
            print(f"ℹ️  Schema {name} already exists in {registry}, updating...")
            try:
                # Update existing schema
                response = client.update_schema(
                    RegistryId={"RegistryName": registry},
                    SchemaName=name,
                    Compatibility=compatibility,
                    SchemaDefinition=json.dumps(schema_def),
                    Description=f"Updated feature schema for {name}"
                )
                print(f"✅ Updated schema {name} to version {response.get('SchemaVersionNumber', 'N/A')}")
            except ClientError as update_error:
                print(f"❌ Failed to update schema: {update_error}")
                sys.exit(1)
        else:
            print(f"❌ AWS Glue error: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Register schema in AWS Glue Schema Registry")
    parser.add_argument("--registry", default="ConvictionAIPipelineRegistry", 
                       help="AWS Glue registry name")
    parser.add_argument("--schema-name", default="feature_schema", 
                       help="Schema name")
    parser.add_argument("--schema-path", default="schemas/feature_schema.json", 
                       help="Path to JSON schema")
    parser.add_argument("--compat", default="BACKWARD", 
                       choices=["NONE", "DISABLED", "BACKWARD", "FORWARD", "FULL"],
                       help="Schema compatibility mode")
    
    args = parser.parse_args()
    
    if not Path(args.schema_path).exists():
        print(f"❌ Schema file not found: {args.schema_path}")
        sys.exit(1)
    
    register_schema(args.registry, args.schema_name, args.schema_path, args.compat)


if __name__ == "__main__":
    main()