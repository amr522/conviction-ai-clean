import polars as pl
from prefect import task
from prefect.tasks import task_input_hash
from datetime import timedelta
import os
from pathlib import Path
import argparse

@task(
    name="partition_datasets",
    description="Partition daily and intraday datasets by date",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
    retries=3,
    retry_delay_seconds=60,
)
def run(date: str, dry_run: bool = False) -> dict:
    """
    Partition daily and intraday master datasets by date.
    
    Args:
        date: The processing date (for logging/tracking)
        dry_run: If True, simulate processing without writing files
    
    Returns:
        dict: Status information about the processing
    """
    try:
        print(f"Starting dataset partitioning for date: {date}")
        
        # Input/output paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        daily_input = os.path.join(project_root, "staged/daily_master.parquet")
        intraday_input = os.path.join(project_root, "staged/intraday_master.parquet")
        
        daily_output_dir = os.path.join(project_root, "partitioned/daily")
        intraday_output_dir = os.path.join(project_root, "partitioned/intraday")
        
        # Create output directories if they don't exist and we're not in dry-run mode
        if not dry_run:
            for directory in [daily_output_dir, intraday_output_dir]:
                Path(directory).mkdir(parents=True, exist_ok=True)

        # Process daily data
        print("\nProcessing daily master dataset...")
        daily_stats = partition_dataset(daily_input, daily_output_dir, "date", dry_run)
        
        # Process intraday data
        print("\nProcessing intraday master dataset...")
        intraday_stats = partition_dataset(intraday_input, intraday_output_dir, "timestamp", dry_run)

        return {
            "status": "success",
            "date": date,
            "daily_stats": daily_stats,
            "intraday_stats": intraday_stats
        }

    except Exception as e:
        print(f"Error partitioning datasets: {str(e)}")
        raise

def partition_dataset(input_path: str, output_dir: str, partition_col: str, dry_run: bool) -> dict:
    """
    Partition a dataset by the specified column.
    
    Args:
        input_path: Path to input parquet file
        output_dir: Directory to write partitioned files
        partition_col: Column to partition by
        dry_run: If True, don't write files
        
    Returns:
        dict: Statistics about the partitioning
    """
    # Validate input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    # Load and validate data
    print(f"Loading data from: {input_path}")
    df = pl.scan_parquet(input_path, extra_columns='ignore')
    data = df.collect()
    
    if data.shape[0] == 0:
        raise ValueError("Input data is empty")
        
    if partition_col not in data.columns:
        raise ValueError(f"Partition column '{partition_col}' not found in data")
        
    # Get unique partition values
    partition_values = data[partition_col].unique().sort()
    
    print(f"\nFound {len(partition_values)} unique {partition_col} values")
    print(f"Date range: {partition_values[0]} to {partition_values[-1]}")
    
    total_rows = 0
    total_files = 0
    
    # Partition and write the data
    for val in partition_values:
        partition_data = data.filter(pl.col(partition_col) == val)
        
        if partition_data.shape[0] == 0:
            continue
            
        # Create partition path
        val_str = str(val).split(" ")[0]  # Get date part only
        partition_path = os.path.join(output_dir, f"{val_str}.parquet")
        
        print(f"\nPartition {val_str}:")
        print(f"Rows: {partition_data.shape[0]}")
        
        if not dry_run:
            # Write the partition
            partition_data.write_parquet(
                partition_path,
                compression="zstd",
                statistics=True,
                use_pyarrow=True,
                pyarrow_options={"compression_level": 3}
            )
            print(f"Wrote partition to: {partition_path}")
        else:
            print("DRY RUN: Skipping file write")
        
        total_rows += partition_data.shape[0]
        total_files += 1
    
    print(f"\nPartitioning complete:")
    print(f"Total rows processed: {total_rows}")
    print(f"Total files created: {total_files}")
    
    return {
        "total_rows": total_rows,
        "total_files": total_files,
        "partition_values": len(partition_values),
        "date_range": [str(partition_values[0]), str(partition_values[-1])]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition master datasets")
    parser.add_argument("--date", type=str, required=True, help="Processing date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing files")
    
    args = parser.parse_args()
    result = run(args.date, dry_run=args.dry_run)
    print(f"Task result: {result}")
