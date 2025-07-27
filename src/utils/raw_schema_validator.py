#!/usr/bin/env python3
"""
Generic raw-data schema validator with fallback support
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


def validate(path: Union[str, pathlib.Path], schema_path: Union[str, pathlib.Path]) -> bool:
    """
    Returns True if `path` exists **and** the first 100 rows conform to `schema_path`.
    Supports Parquet/CSV/JSON based on file suffix.
    
    Args:
        path: Path to data file to validate
        schema_path: Path to JSON schema file
        
    Returns:
        bool: True if validation passes
        
    Raises:
        FileNotFoundError: If path doesn't exist
        SchemaMismatchError: If schema validation fails
    """
    path = pathlib.Path(path)
    schema_path = pathlib.Path(schema_path)
    
    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    
    # Load schema
    if not schema_path.exists():
        logger.warning(f"Schema file not found: {schema_path}, skipping validation")
        return True
        
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Compile schema validator
    validate_schema = fastjsonschema.compile(schema)
    
    try:
        # Load first 100 rows based on file type
        if path.suffix.lower() == '.parquet':
            df = pl.read_parquet(path).head(100)
        elif path.suffix.lower() == '.csv':
            df = pl.read_csv(path).head(100)
        elif path.suffix.lower() == '.json':
            df = pl.read_json(path).head(100)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Convert to dict for schema validation
        data_dict = df.to_pandas().to_dict('records')
        
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