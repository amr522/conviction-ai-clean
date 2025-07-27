#!/usr/bin/env python3
"""
Delta Lake utilities for ACID-compliant data storage
"""
import os
import logging
from typing import Optional
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from delta import configure_spark_with_delta_pip

logger = logging.getLogger(__name__)

def get_delta_spark_session(app_name: str = "ConvictionAI-Delta") -> SparkSession:
    """
    Create Spark session configured for Delta Lake
    
    Args:
        app_name: Spark application name
        
    Returns:
        Configured SparkSession
    """
    try:
        # Configure Spark with Delta Lake
        builder = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        
        # Configure for S3 if needed
        if os.getenv('S3_BUCKET_NAME'):
            builder = builder \
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
                .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                       "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
        
        spark = configure_spark_with_delta_pip(builder).getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        
        logger.info(f"Delta Spark session created: {spark.version}")
        return spark
        
    except Exception as e:
        logger.error(f"Failed to create Delta Spark session: {str(e)}")
        raise

def write_delta_table(
    df: pd.DataFrame,
    path: str,
    mode: str = "overwrite",
    partition_cols: Optional[list] = None,
    merge_schema: bool = True,
    overwrite_schema: bool = False
) -> bool:
    """
    Write pandas DataFrame as Delta table
    
    Args:
        df: Pandas DataFrame to write
        path: Delta table path (local or S3)
        mode: Write mode (overwrite, append, merge)
        partition_cols: Columns to partition by
        merge_schema: Allow schema evolution
        overwrite_schema: Overwrite existing schema
        
    Returns:
        True if successful, False otherwise
    """
    try:
        spark = get_delta_spark_session()
        
        # Convert pandas to Spark DataFrame
        spark_df = spark.createDataFrame(df)
        
        # Configure write operation
        writer = spark_df.write.format("delta").mode(mode)
        
        if merge_schema:
            writer = writer.option("mergeSchema", "true")
        
        if overwrite_schema:
            writer = writer.option("overwriteSchema", "true")
        
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        
        # Write Delta table
        writer.save(path)
        
        logger.info(f"Delta table written successfully: {path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to write Delta table {path}: {str(e)}")
        return False
    
    finally:
        if 'spark' in locals():
            spark.stop()

def read_delta_table(
    path: str,
    timestamp_as_of: Optional[str] = None,
    version_as_of: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Read Delta table with optional time-travel
    
    Args:
        path: Delta table path
        timestamp_as_of: Read table as of timestamp (ISO format)
        version_as_of: Read table as of version number
        
    Returns:
        Pandas DataFrame or None if failed
    """
    try:
        spark = get_delta_spark_session()
        
        # Configure reader
        reader = spark.read.format("delta")
        
        if timestamp_as_of:
            reader = reader.option("timestampAsOf", timestamp_as_of)
        elif version_as_of is not None:
            reader = reader.option("versionAsOf", str(version_as_of))
        
        # Read Delta table
        spark_df = reader.load(path)
        pandas_df = spark_df.toPandas()
        
        logger.info(f"Delta table read successfully: {path} ({len(pandas_df)} rows)")
        return pandas_df
        
    except Exception as e:
        logger.error(f"Failed to read Delta table {path}: {str(e)}")
        return None
    
    finally:
        if 'spark' in locals():
            spark.stop()

def get_delta_table_history(path: str) -> Optional[pd.DataFrame]:
    """
    Get Delta table history/versions
    
    Args:
        path: Delta table path
        
    Returns:
        DataFrame with version history or None if failed
    """
    try:
        spark = get_delta_spark_session()
        
        # Get table history
        history_df = spark.sql(f"DESCRIBE HISTORY delta.`{path}`")
        pandas_history = history_df.toPandas()
        
        logger.info(f"Retrieved {len(pandas_history)} versions for {path}")
        return pandas_history
        
    except Exception as e:
        logger.error(f"Failed to get Delta table history {path}: {str(e)}")
        return None
    
    finally:
        if 'spark' in locals():
            spark.stop()

def optimize_delta_table(path: str, z_order_cols: Optional[list] = None) -> bool:
    """
    Optimize Delta table (compaction and Z-ordering)
    
    Args:
        path: Delta table path
        z_order_cols: Columns for Z-ordering optimization
        
    Returns:
        True if successful, False otherwise
    """
    try:
        spark = get_delta_spark_session()
        
        # Run OPTIMIZE command
        optimize_sql = f"OPTIMIZE delta.`{path}`"
        
        if z_order_cols:
            z_order_str = ", ".join(z_order_cols)
            optimize_sql += f" ZORDER BY ({z_order_str})"
        
        spark.sql(optimize_sql)
        
        logger.info(f"Delta table optimized: {path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to optimize Delta table {path}: {str(e)}")
        return False
    
    finally:
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Delta Lake utilities")
    parser.add_argument("--action", choices=["history", "optimize"], required=True)
    parser.add_argument("--path", required=True, help="Delta table path")
    parser.add_argument("--z-order", nargs="+", help="Columns for Z-ordering")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    if args.action == "history":
        history = get_delta_table_history(args.path)
        if history is not None:
            print(history.to_string())
    
    elif args.action == "optimize":
        success = optimize_delta_table(args.path, args.z_order)
        if success:
            print(f"✅ Optimized: {args.path}")
        else:
            print(f"❌ Failed to optimize: {args.path}")
            exit(1)