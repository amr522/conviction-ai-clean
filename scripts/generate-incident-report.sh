#!/usr/bin/env bash
set -euo pipefail

EVENT=${1:-"AnomalyTestFailed"}
mkdir -p reports
OUT="reports/incident_$(date +%F_%H%M%S).md"

cat <<EOF > $OUT
# Incident Report: $EVENT

**Timestamp:** $(date -u +"%Y-%m-%dT%H:%M:%SZ")  
**Triggered Alert:** $EVENT  

## 1. Context
- Job: anomaly-test cronjob
- Threshold: \${ANOMALY_THRESHOLD:-0.1}
- CI run: \${GITHUB_RUN_ID:-local}

## 2. Logs & Metrics
### Recent Pod Logs
\`\`\`bash
kubectl logs job/conviction-ai-pipeline-anomaly-test --tail=100
\`\`\`

### Prometheus Query
\`\`\`bash
promql='signal_validation_passed{job="anomaly-test"}'
curl -s 'http://prometheus:9090/api/v1/query?query='\"\$promql\"
\`\`\`

## 3. Data Snapshot
\`\`\`bash
python3 - <<'PYCODE'
import polars as pl
df = pl.read_parquet("data/Parquet_data/options_30min_clean_$(date +%Y-%m-%d).parquet")
print(df.tail(100))
PYCODE
\`\`\`

## 4. Next Steps
1. Investigate recent code changes in signal-generation  
2. Re-run local smoke test: \`scripts/smoke-test-staging.sh\`  
3. Rollback helm release if needed: \`helm rollback\`  

EOF

echo "✅ Incident report generated at $OUT"