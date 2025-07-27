#!/usr/bin/env python3
"""
Parquet schema validation helper for the conviction-ai pipeline.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    pass


def load_expected_schema(spec_file: str, dataset_type: str) -> Dict[str, str]:
    """
    Load expected schema from JSON specification file.

    Args:
        spec_file: Path to schema specification JSON file
        dataset_type: Type of dataset (options_daily, options_30min, etc.)

    Returns:
        Dictionary mapping column names to expected types
    """
    with open(spec_file, "r") as f:
        schemas = json.load(f)

    if dataset_type not in schemas:
        raise ValueError(f"Dataset type '{dataset_type}' not found in schema spec")

    return schemas[dataset_type]


def get_parquet_schema(parquet_path: str) -> Dict[str, str]:
    """
    Read schema from Parquet file using PyArrow.

    Args:
        parquet_path: Path to Parquet file

    Returns:
        Dictionary mapping column names to PyArrow types
    """
    table = pq.read_table(parquet_path)
    schema = table.schema

    return {field.name: str(field.type) for field in schema}


def compare_schemas(
    actual: Dict[str, str], expected: Dict[str, str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Compare actual vs expected schemas.

    Args:
        actual: Actual schema from Parquet file
        expected: Expected schema from specification

    Returns:
        Tuple of (missing_columns, extra_columns, type_mismatches)
    """
    actual_cols = set(actual.keys())
    expected_cols = set(expected.keys())

    missing_columns = list(expected_cols - actual_cols)
    extra_columns = list(actual_cols - expected_cols)

    type_mismatches = []
    for col in actual_cols & expected_cols:
        actual_type = actual[col]
        expected_type = expected[col]

        # Normalize type comparisons (PyArrow vs simplified types)
        if not _types_match(actual_type, expected_type):
            type_mismatches.append(
                f"{col}: expected {expected_type}, got {actual_type}"
            )

    return missing_columns, extra_columns, type_mismatches


def _types_match(actual_type: str, expected_type: str) -> bool:
    """
    Check if PyArrow type matches expected simplified type.

    Args:
        actual_type: PyArrow type string
        expected_type: Simplified type from schema spec

    Returns:
        True if types match
    """
    # Type mapping from PyArrow to simplified types
    type_mappings = {
        "double": "float64",
        "float": "float32",
        "int64": "int64",
        "uint64": "uint64",
        "uint32": "uint32",
        "int32": "int32",
        "bool": "bool",
        "string": "string",
        "date32[day]": "date32",
        "timestamp[ns]": "timestamp[ns]",
        "timestamp[ns, tz=UTC]": "timestamp[ns]",
    }

    # Normalize actual type
    normalized_actual = type_mappings.get(actual_type, actual_type)

    return normalized_actual == expected_type


def validate_parquet_schema(
    parquet_path: str, spec_file: str, dataset_type: str
) -> None:
    """
    Validate Parquet file schema against specification.

    Args:
        parquet_path: Path to Parquet file to validate
        spec_file: Path to schema specification JSON file
        dataset_type: Type of dataset (options_daily, options_30min, etc.)

    Raises:
        SchemaValidationError: If schema validation fails
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    if not os.path.exists(spec_file):
        raise FileNotFoundError(f"Schema specification file not found: {spec_file}")

    print(f"Validating schema for {dataset_type}: {parquet_path}")

    # Load schemas
    expected_schema = load_expected_schema(spec_file, dataset_type)
    actual_schema = get_parquet_schema(parquet_path)

    # Compare schemas
    missing_cols, extra_cols, type_mismatches = compare_schemas(
        actual_schema, expected_schema
    )

    # Report results
    errors = []

    if missing_cols:
        errors.append(f"Missing columns: {', '.join(missing_cols)}")

    if extra_cols:
        errors.append(f"Extra columns: {', '.join(extra_cols)}")

    if type_mismatches:
        errors.append(f"Type mismatches: {'; '.join(type_mismatches)}")

    if errors:
        error_msg = f"Schema validation failed for {dataset_type}:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        print(error_msg)
        raise SchemaValidationError(error_msg)

    print(f"✅ Schema validation passed for {dataset_type}")


def validate_all_schemas(base_path: str, spec_file: str) -> None:
    """
    Validate all Parquet files in the pipeline output directory.

    Args:
        base_path: Base directory containing Parquet files
        spec_file: Path to schema specification JSON file
    """
    # Define expected files and their types
    file_mappings = {
        "options_daily_clean.parquet": "options_daily",
        "options_30min_clean.parquet": "options_30min",
        "stocks_daily_clean.parquet": "stocks_daily",
        "stocks_30min_clean.parquet": "stocks_30min",
    }

    validated_count = 0

    for filename, dataset_type in file_mappings.items():
        file_path = os.path.join(base_path, filename)

        if os.path.exists(file_path):
            validate_parquet_schema(file_path, spec_file, dataset_type)
            validated_count += 1
        else:
            print(f"⚠️  File not found (skipping): {file_path}")

    print(f"\n🎉 Schema validation completed: {validated_count} files validated")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate Parquet schema against specification"
    )
    parser.add_argument("--parquet-path", required=True, help="Path to Parquet file")
    parser.add_argument(
        "--spec-file",
        default="schemas/option_parquet_schema.json",
        help="Schema specification file",
    )
    parser.add_argument(
        "--dataset-type",
        required=True,
        help="Dataset type (options_daily, options_30min, etc.)",
    )

    args = parser.parse_args()

    try:
        validate_parquet_schema(args.parquet_path, args.spec_file, args.dataset_type)
    except (SchemaValidationError, FileNotFoundError) as e:
        print(f"❌ Validation failed: {e}")
        exit(1)
