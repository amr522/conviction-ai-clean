#!/usr/bin/env python3
"""
AWS Glue Data Catalog registration utilities
"""
import boto3
import logging
import os
from typing import Dict, List, Optional
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.getLogger(__name__)

def get_parquet_schema(s3_path: str) -> List[Dict[str, str]]:
    """
    Extract schema from Parquet file and convert to Glue format
    
    Args:
        s3_path: S3 path to Parquet file or directory
        
    Returns:
        List of column definitions for Glue table
    """
    try:
        # Read Parquet schema
        dataset = pq.ParquetDataset(s3_path)
        schema = dataset.schema.to_arrow_schema()
        
        # Convert Arrow types to Hive/Glue types
        type_mapping = {
            pa.int64(): "bigint",
            pa.int32(): "int",
            pa.float64(): "double",
            pa.float32(): "float",
            pa.string(): "string",
            pa.bool_(): "boolean",
            pa.timestamp('us'): "timestamp",
            pa.date32(): "date"
        }
        
        columns = []
        for name, arrow_type in zip(schema.names, schema.types):
            # Map Arrow type to Hive type
            hive_type = type_mapping.get(arrow_type, "string")
            
            # Handle special cases
            if str(arrow_type).startswith("timestamp"):
                hive_type = "timestamp"
            elif str(arrow_type).startswith("decimal"):
                hive_type = "decimal(10,2)"
            
            columns.append({
                "Name": name,
                "Type": hive_type
            })
        
        return columns
        
    except Exception as e:
        logger.error(f"Error reading Parquet schema from {s3_path}: {str(e)}")
        # Fallback to basic schema
        return [
            {"Name": "timestamp", "Type": "timestamp"},
            {"Name": "symbol", "Type": "string"},
            {"Name": "value", "Type": "double"}
        ]

def register_parquet_table(
    database: str, 
    table: str, 
    s3_path: str, 
    region: str = None,
    partition_keys: Optional[List[Dict[str, str]]] = None,
    description: str = None
) -> bool:
    """
    Register Parquet table in AWS Glue Data Catalog
    
    Args:
        database: Glue database name
        table: Table name
        s3_path: S3 location of Parquet files
        region: AWS region (defaults to environment)
        partition_keys: Optional partition key definitions
        description: Table description
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize Glue client
        glue_client = boto3.client("glue", region_name=region)
        
        # Ensure database exists
        try:
            glue_client.get_database(Name=database)
        except glue_client.exceptions.EntityNotFoundException:
            logger.info(f"Creating Glue database: {database}")
            glue_client.create_database(
                DatabaseInput={
                    "Name": database,
                    "Description": f"Conviction AI data catalog - {database}"
                }
            )
        
        # Get Parquet schema
        columns = get_parquet_schema(s3_path)
        
        # Prepare table input
        table_input = {
            "Name": table,
            "StorageDescriptor": {
                "Columns": columns,
                "Location": s3_path,
                "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                "SerdeInfo": {
                    "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                    "Parameters": {"serialization.format": "1"}
                },
                "Compressed": True,
                "StoredAsSubDirectories": False
            },
            "TableType": "EXTERNAL_TABLE",
            "Parameters": {
                "classification": "parquet",
                "compressionType": "snappy",
                "typeOfData": "file"
            }
        }
        
        # Add partition keys if provided
        if partition_keys:
            table_input["PartitionKeys"] = partition_keys
        
        # Add description if provided
        if description:
            table_input["Description"] = description
        
        # Try to update existing table first
        try:
            glue_client.update_table(
                DatabaseName=database,
                TableInput=table_input
            )
            logger.info(f"Updated Glue table: {database}.{table}")
            
        except glue_client.exceptions.EntityNotFoundException:
            # Create new table
            glue_client.create_table(
                DatabaseName=database,
                TableInput=table_input
            )
            logger.info(f"Created Glue table: {database}.{table}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error registering table {database}.{table}: {str(e)}")
        return False

def register_pipeline_tables(
    s3_bucket: str, 
    s3_prefix: str = "", 
    database: str = "conviction_ai",
    region: str = None
) -> Dict[str, bool]:
    """
    Register all pipeline output tables in Glue catalog
    
    Args:
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix for data files
        database: Glue database name
        region: AWS region
        
    Returns:
        Dictionary of table registration results
    """
    results = {}
    
    # Define pipeline tables
    tables = [
        {
            "name": "stocks_daily",
            "path": f"s3://{s3_bucket}/{s3_prefix}stocks_daily_clean.parquet",
            "description": "Daily stock price and volume data with technical indicators",
            "partitions": [{"Name": "year", "Type": "string"}, {"Name": "month", "Type": "string"}]
        },
        {
            "name": "options_daily", 
            "path": f"s3://{s3_bucket}/{s3_prefix}options_daily_clean.parquet",
            "description": "Daily options data with Greeks and volatility metrics",
            "partitions": [{"Name": "year", "Type": "string"}, {"Name": "month", "Type": "string"}]
        },
        {
            "name": "stocks_30min",
            "path": f"s3://{s3_bucket}/{s3_prefix}stocks_30min_clean.parquet", 
            "description": "30-minute aggregated stock data with intraday indicators",
            "partitions": [{"Name": "year", "Type": "string"}, {"Name": "month", "Type": "string"}]
        },
        {
            "name": "options_30min",
            "path": f"s3://{s3_bucket}/{s3_prefix}options_30min_clean.parquet",
            "description": "30-minute options data with flow analysis and gamma metrics",
            "partitions": [{"Name": "year", "Type": "string"}, {"Name": "month", "Type": "string"}]
        },
        {
            "name": "intraday_master",
            "path": f"s3://{s3_bucket}/{s3_prefix}intraday_master.parquet",
            "description": "Master dataset combining stocks and options for ML training",
            "partitions": [{"Name": "year", "Type": "string"}, {"Name": "month", "Type": "string"}]
        }
    ]
    
    # Register each table
    for table_config in tables:
        success = register_parquet_table(
            database=database,
            table=table_config["name"],
            s3_path=table_config["path"],
            region=region,
            partition_keys=table_config.get("partitions"),
            description=table_config.get("description")
        )
        results[table_config["name"]] = success
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Register Parquet tables in AWS Glue")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket name")
    parser.add_argument("--s3-prefix", default="", help="S3 prefix for data files")
    parser.add_argument("--database", default="conviction_ai", help="Glue database name")
    parser.add_argument("--region", help="AWS region")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Register all tables
    results = register_pipeline_tables(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        database=args.database,
        region=args.region
    )
    
    # Print results
    print("Glue table registration results:")
    for table, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {table}: {status}")
    
    # Exit with error code if any failed
    if not all(results.values()):
        exit(1)