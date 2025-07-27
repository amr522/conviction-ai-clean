#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import polars as pl
from fastjsonschema import compile


def load_schema(schema_file):
    """Load JSON schema from file."""
    return json.loads(Path(schema_file).read_text())


def validate_parquet_against_schema(parquet_path, schema_file):
    """Validate parquet file against JSON schema."""
    df = pl.read_parquet(parquet_path)
    schema = load_schema(schema_file)
    validator = compile(schema)
    
    # Convert to dicts for validation
    records = df.to_dicts()
    errors = []
    
    for i, row in enumerate(records[:100]):  # Validate first 100 rows
        try:
            validator(row)
        except Exception as e:
            errors.append(f"Row {i}: {e}")
            if len(errors) >= 10:  # Limit error output
                break
    
    if errors:
        print("❌ Schema validation failed:")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)
    
    print(f"✅ Schema validation passed for {parquet_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate parquet against JSON schema")
    parser.add_argument("--parquet", required=True, help="Path to parquet file")
    parser.add_argument("--schema", required=True, help="Path to JSON schema file")
    
    args = parser.parse_args()
    validate_parquet_against_schema(args.parquet, args.schema)


if __name__ == "__main__":
    main()