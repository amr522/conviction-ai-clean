"""Tests for performance optimization utilities."""

import pytest
import polars as pl
from src.utils.performance_utils import (
    optimize_join_performance,
    compute_flow_signals_optimized,
    compute_gamma_signals_optimized,
    PERF_CONFIG
)

@pytest.fixture
def sample_stocks_data():
    """Sample stocks data for testing."""
    return pl.DataFrame({
        'timestamp': ['2025-01-01 09:30:00', '2025-01-01 10:00:00'] * 2,
        'ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
        'stock30_close': [150.0, 151.0, 300.0, 301.0],
        'stock30_volume': [1000, 1100, 2000, 2100]
    }).with_columns([
        pl.col('timestamp').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S')
    ])

@pytest.fixture
def sample_options_data():
    """Sample options data for testing."""
    return pl.DataFrame({
        'timestamp': ['2025-01-01 09:30:00', '2025-01-01 10:00:00'] * 2,
        'ticker': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
        'underlying': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
        'opt30_volume': [100, 110, 200, 210],
        'opt30_strike': [150.0, 150.0, 300.0, 300.0],
        'opt30_type': ['C', 'P', 'C', 'P']
    }).with_columns([
        pl.col('timestamp').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S')
    ])

def test_optimize_join_performance(sample_stocks_data, sample_options_data):
    """Test optimized join performance."""
    stocks_lazy = sample_stocks_data.lazy()
    options_lazy = sample_options_data.lazy()
    
    result = optimize_join_performance(stocks_lazy, options_lazy)
    
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] == 4  # All rows should join
    assert 'stock30_close' in result.columns
    assert 'opt30_volume' in result.columns

def test_compute_flow_signals_optimized(sample_options_data):
    """Test optimized flow signal computation."""
    result = compute_flow_signals_optimized(sample_options_data, window=5)
    
    assert 'opt30_call_flow' in result.columns
    assert 'opt30_put_flow' in result.columns
    assert 'opt30_flow_divergence' in result.columns
    assert result.shape[0] == sample_options_data.shape[0]

def test_compute_gamma_signals_optimized(sample_options_data):
    """Test optimized gamma signal computation."""
    result = compute_gamma_signals_optimized(sample_options_data, window=5)
    
    assert 'opt30_gamma' in result.columns
    assert 'opt30_net_gamma' in result.columns
    assert 'opt30_gamma_mean_5' in result.columns
    assert 'opt30_gamma_std_5' in result.columns
    assert 'opt30_gamma_squeeze' in result.columns
    assert result.shape[0] == sample_options_data.shape[0]

def test_perf_config_structure():
    """Test that performance config has expected structure."""
    assert 'window_sizes' in PERF_CONFIG
    assert 'join_hints' in PERF_CONFIG
    assert 'multipliers' in PERF_CONFIG
    
    assert 'flow_window' in PERF_CONFIG['window_sizes']
    assert 'gamma_window' in PERF_CONFIG['window_sizes']
    assert 'streaming' in PERF_CONFIG['join_hints']
    assert 'gamma_squeeze' in PERF_CONFIG['multipliers']

def test_config_defaults():
    """Test that config defaults are reasonable."""
    assert PERF_CONFIG['window_sizes']['flow_window'] > 0
    assert PERF_CONFIG['multipliers']['gamma_squeeze'] > 0
    assert isinstance(PERF_CONFIG['join_hints']['streaming'], bool)