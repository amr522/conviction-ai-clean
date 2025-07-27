#!/usr/bin/env python3
import argparse
import polars as pl
from pathlib import Path

def generate_training_dataset(feature_path: str, label_path: str, output_path: str) -> None:
    """Generate training dataset by joining features with labels"""
    print(f"Loading features from {feature_path}")
    feats = pl.read_parquet(feature_path)
    
    print(f"Loading labels from {label_path}")
    labels = pl.read_parquet(label_path)
    
    # Join on date & ticker
    print("Joining features with labels...")
    train = feats.join(labels, on=["date", "ticker"], how="inner")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write out training dataset
    train.write_parquet(output_path)
    print(f"✅ Training dataset written to {output_path}")
    print(f"Dataset shape: {train.shape}")
    print(f"Columns: {len(train.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training dataset from features and labels")
    parser.add_argument("--feature-path", required=True, help="Path to features parquet file")
    parser.add_argument("--label-path", required=True, help="Path to labels parquet file")
    parser.add_argument("--output-path", default="data/Parquet_data/train_dataset_{date}.parquet", help="Output path template")
    
    args = parser.parse_args()
    
    # Replace {date} placeholder with actual date if present
    if "{date}" in args.output_path:
        date_part = args.feature_path.split('_')[-1].replace('.parquet', '')
        output = args.output_path.format(date=date_part)
    else:
        output = args.output_path
    
    generate_training_dataset(args.feature_path, args.label_path, output)