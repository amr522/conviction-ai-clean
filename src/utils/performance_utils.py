"""Performance optimization utilities for ETL operations."""

import polars as pl

from utils.profiling import profile_time

# Centralized performance configuration
PERF_CONFIG = {
    "window_sizes": {
        "flow_window": 5,
        "gamma_window": 5,
        "volume_window": 5,
        "volatility_window": 5,
    },
    "join_hints": {
        "broadcast_threshold": 100000,  # rows
        "streaming": True,
        "join_nulls": False,
    },
    "multipliers": {"gamma_squeeze": 2.0, "volume_spike": 2.0},
}


@profile_time
def optimize_join_performance(
    stocks_df: pl.LazyFrame, options_df: pl.LazyFrame, on: list = None
) -> pl.DataFrame:
    """
    Optimized join with broadcast hints and performance tuning.

    Args:
        stocks_df: Larger stocks DataFrame (left side)
        options_df: Smaller options DataFrame (right side, will be broadcast)
        on: Join columns (defaults to timestamp, ticker)

    Returns:
        Joined DataFrame with optimized performance
    """
    if on is None:
        on = ["timestamp", "ticker"]

    print("Executing optimized broadcast join...")

    # Perform join with optimization hints from config
    result = stocks_df.join(
        options_df,
        left_on=on,
        right_on=on,
        how="inner",
        join_nulls=PERF_CONFIG["join_hints"]["join_nulls"],
    ).collect(streaming=PERF_CONFIG["join_hints"]["streaming"])

    print(f"Join completed: {result.shape[0]} rows")
    return result


@profile_time
def compute_flow_signals_optimized(
    df: pl.DataFrame, window: int = None
) -> pl.DataFrame:
    """
    Compute flow divergence signals using native Polars operations.

    Args:
        df: Input DataFrame with options data

    Returns:
        DataFrame with flow signal columns added
    """
    print("Computing flow signals with native Polars...")

    # Extract option type if not present
    if "opt30_type" not in df.columns:
        df = df.with_columns(
            [
                pl.col("ticker")
                .str.extract(r"([CP])", 1)
                .fill_null("C")
                .alias("opt30_type")
            ]
        )

    # Compute flow aggregations
    flow_agg = (
        df.group_by(["underlying", "timestamp", "opt30_type"])
        .agg([pl.col("opt30_volume").sum().alias("flow_volume")])
        .pivot(
            index=["underlying", "timestamp"],
            columns="opt30_type",
            values="flow_volume",
        )
        .with_columns(
            [
                pl.col("C").fill_null(0).alias("opt30_call_flow"),
                pl.col("P").fill_null(0).alias("opt30_put_flow"),
            ]
        )
        .with_columns(
            [
                (pl.col("opt30_call_flow") - pl.col("opt30_put_flow")).alias(
                    "opt30_flow_divergence"
                )
            ]
        )
        .select(
            [
                "underlying",
                "timestamp",
                "opt30_call_flow",
                "opt30_put_flow",
                "opt30_flow_divergence",
            ]
        )
    )

    # Join back to main dataset
    result = df.join(flow_agg, on=["underlying", "timestamp"], how="left")
    print(f"Flow signals computed for {result.shape[0]} rows")
    return result


@profile_time
def compute_gamma_signals_optimized(
    df: pl.DataFrame, window: int = None, gamma_squeeze_multiplier: float = None
) -> pl.DataFrame:
    """
    Compute gamma squeeze signals using optimized window functions.

    Args:
        df: Input DataFrame with options data
        gamma_squeeze_multiplier: Threshold multiplier for squeeze detection

    Returns:
        DataFrame with gamma signal columns added
    """
    if window is None:
        window = PERF_CONFIG["window_sizes"]["gamma_window"]
    if gamma_squeeze_multiplier is None:
        gamma_squeeze_multiplier = PERF_CONFIG["multipliers"]["gamma_squeeze"]

    print(
        f"Computing gamma signals with window={window}, multiplier={gamma_squeeze_multiplier}..."
    )

    result = (
        df.with_columns(
            [
                # Simplified gamma and open interest proxies
                pl.lit(0.01).alias("opt30_gamma"),
                (pl.col("opt30_volume") * 10).alias("opt30_open_interest"),
            ]
        )
        .with_columns(
            [
                # Net gamma calculation
                (pl.col("opt30_gamma") * pl.col("opt30_open_interest")).alias(
                    "opt30_net_gamma"
                )
            ]
        )
        .with_columns(
            [
                # Rolling statistics using native window functions
                pl.col("opt30_net_gamma")
                .rolling_mean(window_size=window, min_periods=1)
                .over("underlying")
                .alias(f"opt30_gamma_mean_{window}"),
                pl.col("opt30_net_gamma")
                .rolling_std(window_size=window, min_periods=1)
                .over("underlying")
                .alias(f"opt30_gamma_std_{window}"),
            ]
        )
        .with_columns(
            [
                # Gamma squeeze detection
                (
                    pl.col("opt30_net_gamma")
                    > (
                        pl.col(f"opt30_gamma_mean_{window}")
                        + gamma_squeeze_multiplier * pl.col(f"opt30_gamma_std_{window}")
                    )
                ).alias("opt30_gamma_squeeze")
            ]
        )
    )

    print(f"Gamma signals computed for {result.shape[0]} rows")
    return result


@profile_time
def optimize_signal_generation(
    df: pl.DataFrame, window_size: int = 5
) -> pl.DataFrame:
    """
    Applies optimized rolling mean & std on volume and gamma.
    """
    return df.with_columns([
        pl.col("opt30_volume").rolling_mean(window_size).alias("rolling_vol_mean"),
        pl.col("opt30_volume").rolling_std(window_size).alias("rolling_vol_std"),
        pl.col("opt30_net_gamma").rolling_mean(window_size).alias("rolling_gamma_mean"),
        pl.col("opt30_net_gamma").rolling_std(window_size).alias("rolling_gamma_std")
    ])


@profile_time
def enhance_gamma_detection(
    df: pl.DataFrame, multiplier: float = 2.0
) -> pl.DataFrame:
    """
    Flags gamma squeezes where net_gamma > rolling_gamma_mean * multiplier.
    """
    return df.with_columns([
        (pl.col("opt30_net_gamma") > pl.col("rolling_gamma_mean") * multiplier)
        .alias("gamma_squeeze_enhanced")
    ])
