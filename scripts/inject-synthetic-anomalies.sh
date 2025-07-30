#!/usr/bin/env bash
set -euo pipefail

DATE=${1:-$(date +%F)}
IN_PQ="data/Parquet_data/options_30min_clean_${DATE}.parquet"
OUT_PQ="data/Parquet_data/options_30min_anomalies_${DATE}.parquet"

echo "🧪 Injecting anomalies into $IN_PQ"

# Create input data if it doesn't exist
if [[ ! -f "$IN_PQ" ]]; then
    mkdir -p data/Parquet_data
    python3 - <<'PYCODE'
import polars as pl
from datetime import date
import numpy as np

# Create synthetic options data
np.random.seed(42)
df = pl.DataFrame({
    'date': [str(date(2025, 1, 1))] * 1000,
    'ticker': ['AAPL'] * 1000,
    'opt30_call_flow': np.random.normal(1000, 200, 1000),
    'opt30_put_flow': np.random.normal(800, 150, 1000),
    'opt30_flow_divergence': np.random.normal(0, 100, 1000),
    'opt30_net_gamma': np.random.normal(0.05, 0.01, 1000),
    'opt30_close': np.random.normal(150, 10, 1000),
    'opt30_vol_spike': [False] * 1000
})
df.write_parquet("$IN_PQ")
print("✅ Created synthetic input data")
PYCODE
fi

python3 - <<'PYCODE'
import polars as pl
import random

df = pl.read_parquet("$IN_PQ")
# Mark 5% of rows as anomalies with 10x call flow spikes
indices = random.sample(range(df.height), int(df.height * 0.05))

df = df.with_columns([
    pl.when(pl.Series("idx", list(range(df.height))).is_in(indices))
      .then(pl.col("opt30_call_flow") * 10)
      .otherwise(pl.col("opt30_call_flow"))
      .alias("opt30_call_flow"),
    pl.Series("true_spike", [i in indices for i in range(df.height)])
])

df.write_parquet("$OUT_PQ")
print(f"✅ Wrote anomalies to $OUT_PQ")
print(f"   Injected {len(indices)} anomalies ({len(indices)/df.height*100:.1f}%)")
PYCODE