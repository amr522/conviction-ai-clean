#!/usr/bin/env python3
"""
Options ticker parsing utilities

Parses option tickers in format: O:AAPL250404C00130000
- O: = Options prefix
- AAPL = Underlying symbol
- 250404 = Expiration date (YYMMDD)
- C = Call/Put (C/P)
- 00130000 = Strike price (multiply by 1000)
"""

import re
from datetime import datetime

import pandas as pd
import polars as pl


def parse_option_ticker(ticker: str) -> dict:
    """Parse option ticker into components"""
    if not ticker.startswith("O:"):
        return {
            "underlying": None,
            "exp_date": None,
            "option_type": None,
            "strike": None,
        }

    # Remove O: prefix
    ticker_clean = ticker[2:]

    # Pattern: SYMBOL + YYMMDD + C/P + STRIKE
    # Example: AAPL250404C00130000
    pattern = r"^([A-Z]+)(\d{6})([CP])(\d{8})$"
    match = re.match(pattern, ticker_clean)

    if not match:
        return {
            "underlying": None,
            "exp_date": None,
            "option_type": None,
            "strike": None,
        }

    symbol, date_str, option_type, strike_str = match.groups()

    # Parse expiration date
    try:
        exp_date = datetime.strptime(date_str, "%y%m%d").date()
    except:
        exp_date = None

    # Parse strike (divide by 1000)
    try:
        strike = float(strike_str) / 1000.0
    except:
        strike = None

    return {
        "underlying": symbol,
        "exp_date": exp_date,
        "option_type": option_type,
        "strike": strike,
    }


def add_option_fields(df: pl.DataFrame) -> pl.DataFrame:
    """Add parsed option fields to DataFrame"""

    # Convert to pandas for complex parsing
    df_pandas = df.to_pandas()

    # Parse all tickers
    parsed_data = []
    for ticker in df_pandas["ticker"]:
        parsed = parse_option_ticker(ticker)
        parsed_data.append(parsed)

    # Add parsed fields
    parsed_df = pd.DataFrame(parsed_data)
    df_pandas = pd.concat([df_pandas, parsed_df], axis=1)

    # Convert back to polars
    return pl.from_pandas(df_pandas)


def validate_option_parsing(df: pl.DataFrame) -> dict:
    """Validate option parsing results"""
    total_rows = df.height

    # Count successful parses
    valid_underlying = df.filter(pl.col("underlying").is_not_null()).height
    valid_exp_date = df.filter(pl.col("exp_date").is_not_null()).height
    valid_option_type = df.filter(pl.col("option_type").is_not_null()).height
    valid_strike = df.filter(pl.col("strike").is_not_null()).height

    return {
        "total_rows": total_rows,
        "valid_underlying": valid_underlying,
        "valid_exp_date": valid_exp_date,
        "valid_option_type": valid_option_type,
        "valid_strike": valid_strike,
        "parse_success_rate": valid_underlying / total_rows if total_rows > 0 else 0,
    }


if __name__ == "__main__":
    # Test parsing
    test_tickers = [
        "O:AAPL250404C00130000",
        "O:TSLA250404P00200000",
        "O:SPY250404C00450000",
    ]

    for ticker in test_tickers:
        parsed = parse_option_ticker(ticker)
        print(f"{ticker} -> {parsed}")
