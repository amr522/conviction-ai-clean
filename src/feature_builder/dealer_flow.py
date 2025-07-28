#!/usr/bin/env python3
"""
Dealer flow (GEX) feature computation
"""
from pathlib import Path

import polars as pl


def compute_gex_spx(input_path: str, output_path: str) -> pl.DataFrame:
    """
    Compute GEX SPX features from CBOE dealer positioning data.

    Args:
        input_path: Path to raw CBOE CSV with date, gex_spx columns
        output_path: Path to write processed parquet

    Returns:
        Polars DataFrame with date, gex_spx, gex_spx_lag1
    """
    if not Path(input_path).exists():
        # Return empty schema if no data
        return pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Date),
                "gex_spx": pl.Series([], dtype=pl.Float64),
                "gex_spx_lag1": pl.Series([], dtype=pl.Float64),
            }
        )

    # Load raw dealer flow data
    df = pl.read_csv(input_path, try_parse_dates=True)

    # Ensure date column is properly typed
    if "date" in df.columns:
        df = df.with_columns(pl.col("date").cast(pl.Date))

    # Sort by date and forward-fill missing values
    df = (
        df.sort("date")
        .with_columns([pl.col("gex_spx").forward_fill().alias("gex_spx")])
        .with_columns([pl.col("gex_spx").shift(1).alias("gex_spx_lag1")])
    )

    # Write to output
    if output_path:
        df.write_parquet(output_path)

    return df
