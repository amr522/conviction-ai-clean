# Incident Response Runbook

## AnomalyTestFailed

### Overview
Critical alert triggered when synthetic anomaly injection test fails in staging.

### Response Steps

1. **View incident report artifact in CI**
   - Download from GitHub Actions artifacts
   - Contains timestamp, context, and diagnostic commands

2. **Check pod logs**
   ```bash
   kubectl logs job/conviction-ai-pipeline-anomaly-test --tail=100
   ```

3. **Inspect Prometheus metrics**
   ```bash
   # Query anomaly test status
   curl -s 'http://prometheus:9090/api/v1/query?query=signal_validation_passed{job="anomaly-test"}'

   # Check detection rates
   curl -s 'http://prometheus:9090/api/v1/query?query=vol_spike_detection{job="anomaly-test"}'
   ```

4. **Remediate**
   - Fix data/signal logic or rollback deployment
   - Re-trigger cronjob manually:
     ```bash
     kubectl create job --from=cronjob/conviction-ai-pipeline-anomaly-test manual-test-$(date +%s)
     ```

5. **Close Incident**
   - Verify metrics return to normal
   - Update incident report with resolution

## GammaCoverageLow

### Response Steps
1. Check gamma calculation logic in `src/clean_options_30min.py`
2. Validate input data quality
3. Adjust threshold if needed via Helm values

## FlowAccuracyLow / VolSpikeDetectionLow

### Response Steps
1. Review signal detection algorithms
2. Check for data quality issues
3. Consider threshold adjustments based on market conditions
