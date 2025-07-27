from typing import Dict

import pandas as pd
from prefect import task


@task
def cast_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast raw input columns to correct data types."""
    df = df.astype({
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'UInt64',
        'transactions': 'UInt32',
    })
    df['window_start'] = pd.to_datetime(df['window_start'], unit='ns')
    df['ticker'] = df['ticker'].astype(str)

    # Handle options-specific columns
    if 'underlying_symbol' in df.columns:
        df['underlying_symbol'] = df['underlying_symbol'].astype(str)
    if 'strike' in df.columns:
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')

    return df


@task
def cast_feature_dtypes(df: pd.DataFrame, dtypes: Dict[str, str]) -> pd.DataFrame:
    """Cast computed features to correct data types."""
    for col, dtype in dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


# Standard feature data type mappings
STOCKS_DAILY_DTYPES = {
    'stockd_close': 'float64',
    'stockd_volume': 'UInt64',
    'stockd_return_1d': 'float64',
    'stockd_return_1d_lag1': 'float64',
    'stockd_vol_7d': 'float64',
    'stockd_vol_7d_lag1': 'float64',
    'stockd_volume_pct_change': 'float64',
    'stockd_beta_spy': 'float64',
    'stockd_days_to_earnings': 'Int64',
    'stockd_earnings_flag': 'boolean'
}

STOCKS_30MIN_DTYPES = {
    'stock30_close_return': 'float64',
    'stock30_rolling_vol_5': 'float64',
    'stock30_is_last_30min': 'boolean',
    'stock30_close': 'float64',
    'stock30_volume': 'UInt64',
    'stock30_open': 'float64',
    'stock30_high': 'float64',
    'stock30_low': 'float64'
}

OPTIONS_DAILY_DTYPES = {
    'optd_strike': 'float64',
    'optd_moneyness': 'float64',
    'optd_iv30': 'float64',
    'optd_hv30': 'float64',
    'optd_iv30_lag1': 'float64',
    'optd_hv30_lag1': 'float64',
    'optd_iv_percentile': 'float64',
    'optd_iv_percentile_lag1': 'float64',
    'optd_vrp_30d': 'float64',
    'optd_vrp_30d_lag1': 'float64',
    'optd_iv_skew_slope': 'float64',
    'optd_vol_surprise': 'float64',
    'optd_put_call_ratio': 'float64',
    'optd_volume': 'UInt64'
}

OPTIONS_30MIN_DTYPES = {
    'opt30_strike': 'float64',
    'opt30_moneyness': 'float64',
    'opt30_mid_price_return': 'float64',
    'opt30_bid_ask_spread': 'float64',
    'opt30_implied_volatility': 'float64',
    'opt30_delta': 'float64',
    'opt30_theta': 'float64',
    'opt30_volume_return': 'float64',
    'opt30_rolling_vol_5': 'float64',
    'opt30_call_flow': 'float64',
    'opt30_put_flow': 'float64',
    'opt30_flow_divergence': 'float64',
    'opt30_net_gamma': 'float64',
    'opt30_gamma_mean_5': 'float64',
    'opt30_gamma_std_5': 'float64',
    'opt30_gamma_squeeze': 'boolean'
}
