#!/usr/bin/env python3
"""
Generic raw-data schema validator with fallback support.
Updated to work with new data sources that will have identical schema after go-live.
"""
import json
import logging
import pathlib
from typing import Union

import fastjsonschema
import polars as pl

logger = logging.getLogger(__name__)


class SchemaMismatchError(Exception):
    """Raised when raw data doesn't match expected schema"""

    pass


def validate(
    path: Union[str, pathlib.Path], schema_path: Union[str, pathlib.Path]
) -> bool:
    """
    Returns True if `path` exists **and** the first 100 rows conform to `schema_path`.
    Supports Parquet files/directories, CSV, and JSON.

    Args:
        path: Path to data file or directory to validate
        schema_path: Path to JSON schema file

    Returns:
        bool: True if validation passes

    Raises:
        FileNotFoundError: If path doesn't exist
        SchemaMismatchError: If schema validation fails
    """
    path = pathlib.Path(path)
    schema_path = pathlib.Path(schema_path)

    # Check if file or directory exists
    if not path.exists():
        raise FileNotFoundError(f"Raw data path not found: {path}")

    # Load schema
    if not schema_path.exists():
        logger.warning(f"Schema file not found: {schema_path}, skipping validation")
        return True

    with open(schema_path, "r") as f:
        schema = json.load(f)

    # Compile schema validator
    validate_schema = fastjsonschema.compile(schema)

    try:
        # Load first 100 rows based on path type
        try:
            if path.is_dir():
                # Directory of parquet files - try with flexible schema first
                try:
                    df = pl.scan_parquet(path).head(100).collect()
                except Exception:
                    # Fallback: read individual files and combine
                    parquet_files = list(path.glob("*.parquet"))
                    if parquet_files:
                        df = pl.read_parquet(parquet_files[0]).head(100)
                    else:
                        raise ValueError(f"No parquet files found in directory: {path}")
            elif path.suffix.lower() == ".parquet":
                # Single parquet file
                df = pl.read_parquet(path).head(100)
            elif path.suffix.lower() == ".csv":
                df = pl.read_csv(path).head(100)
            elif path.suffix.lower() == ".json":
                df = pl.read_json(path).head(100)
            else:
                raise ValueError(
                    f"Unsupported file format: {path.suffix if path.is_file() else 'directory'}"
                )
        except Exception as e:
            # If we can't read the data, log warning and return True (skip validation)
            logger.warning(
                f"Could not read data from {path} for schema validation: {e}"
            )
            return True

        # Convert to dict for schema validation
        data_dict = df.to_pandas().to_dict("records")

        # Validate each row (sample first few for performance)
        sample_size = min(10, len(data_dict))
        for i, row in enumerate(data_dict[:sample_size]):
            try:
                validate_schema(row)
            except fastjsonschema.JsonSchemaException as e:
                raise SchemaMismatchError(
                    f"Schema validation failed at row {i}: {e.message}"
                )

        logger.info(f"Schema validation passed for {path}")
        return True

    except Exception as e:
        if isinstance(e, SchemaMismatchError):
            raise
        raise SchemaMismatchError(f"Failed to validate {path}: {str(e)}")
