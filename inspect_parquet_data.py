#!/usr/bin/env python3
# inspect_parquet_data.py

import argparse
import sys
import pandas as pd
import s3fs


def inspect_parquet(s3_uri, sample_size):
    """
    Load the first Parquet file under s3_uri, display schema, null percentages,
    sample rows, and suggest numeric columns.
    """
    fs = s3fs.S3FileSystem()

    # List all parquet files under the prefix
    try:
        paths = fs.glob(f"{s3_uri.rstrip('/')}/**/*.parquet", detail=False)
    except Exception as e:
        print(f"❌ Failed to list parquet files under {s3_uri}: {e}", file=sys.stderr)
        sys.exit(1)

    if not paths:
        print(f"❌ No parquet files found under {s3_uri}", file=sys.stderr)
        sys.exit(1)

    # Load the first file
    path = paths[0]
    print(f"🔍 Inspecting file: {path}\n")
    try:
        df = pd.read_parquet(path, filesystem=fs)
    except Exception as e:
        print(f"❌ Failed to read parquet file {path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Display columns and dtypes
    print("ℹ️ Column list (name: dtype):")
    for name, dtype in df.dtypes.items():
        print(f" - {name} ({dtype})")
    print()

    # Null percentage per column
    total = len(df)
    nulls = df.isna().sum()
    print("ℹ️ Null percentages:")
    for col, cnt in nulls.items():
        pct = cnt / total * 100
        print(f" - {col}: {pct:.2f}%")
    print()

    # Show sample rows
    print(f"ℹ️ Sample data (first {sample_size} rows):")
    print(df.head(sample_size).to_string(index=False))
    print()

    # Suggest numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print("✅ Potential numeric/target columns:")
    for col in numeric_cols:
        print(f" - {col}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a Parquet dataset on S3: schema, null rates, samples, and numeric columns"
    )
    parser.add_argument(
        "--s3-uri", required=True,
        help="S3 prefix where parquet files reside (e.g. s3://bucket/path/clean/)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=5,
        help="Number of rows to show in sample output"
    )
    args = parser.parse_args()

    inspect_parquet(args.s3_uri, args.sample_size)


if __name__ == "__main__":
    try:
        import pandas  # ensure pandas is installed
        import s3fs    # ensure s3fs is installed
    except ImportError as e:
        print(f"Required package missing: {e.name}. Install with 'pip install pandas s3fs'.", file=sys.stderr)
        sys.exit(1)
    main()