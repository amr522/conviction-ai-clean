# Troubleshooting Guide

This guide provides step-by-step troubleshooting procedures for common issues in the Conviction AI pipeline.

## Risk Assessment

We compute three risk scores to ensure pipeline health and reliability:

### 1. Data Gaps (Threshold: 0.95)
**Description**: Fraction of rows with non-null core features  
**Calculation**: Complete rows / Total rows across `optd_`, `opt30_`, `stock_` columns  
**Risk**: Data quality issues, incomplete feature sets

### 2. Signal Noise (Threshold: 0.90)
**Description**: Average of signal-quality metrics  
**Calculation**: Mean of gamma coverage, flow accuracy, volume spike detection  
**Risk**: Poor signal quality, unreliable predictions

### 3. Performance (Threshold: 0.50)
**Description**: Inverse scaled latency score  
**Calculation**: max(0, 1 - (latency_ms / 1000))  
**Risk**: System performance degradation, timeout issues

### Running Risk Assessment

Run locally to assess current pipeline health:

```bash
./src/risk_assessment.py \
  --parquet data/Parquet_data/daily_master.parquet \
  --signal-metrics results/signal_metrics.json \
  --latency-ms 150 \
  --output-path risk_report.json
```

### Risk Report Format

```json
{
  "scores": {
    "data_gaps": 0.96,
    "signal_noise": 0.91,
    "performance": 0.85
  },
  "risks": {},
  "thresholds": {
    "data_gaps": 0.95,
    "signal_noise": 0.90,
    "performance": 0.50
  }
}
```

### Remediation Actions

**Data Gaps < 0.95**:
- Check data ingestion pipeline
- Validate upstream data sources
- Review ETL transformation logic

**Signal Noise < 0.90**:
- Run signal validation diagnostics
- Check anomaly detection accuracy
- Review feature engineering pipeline

**Performance < 0.50**:
- Optimize query performance
- Scale compute resources
- Review system bottlenecks

## Common Issues

### Pipeline Failures

1. **Check CI status**: Review GitHub Actions for failed jobs
2. **Validate data**: Run `./scripts/run-all-validations.sh`
3. **Check logs**: Review application and system logs
4. **Run diagnostics**: Execute risk assessment module

### Signal Quality Issues

1. **Gamma Coverage Low**: Check options data completeness
2. **Flow Accuracy Poor**: Validate flow calculation logic
3. **Volume Spikes Missed**: Review spike detection thresholds

### Performance Issues

1. **High Latency**: Check database query performance
2. **Memory Usage**: Monitor resource consumption
3. **Timeout Errors**: Review timeout configurations

## Emergency Procedures

### Critical System Failure

1. **Immediate**: Stop data ingestion
2. **Assess**: Run full validation suite
3. **Isolate**: Identify failing components
4. **Restore**: Rollback to last known good state
5. **Monitor**: Continuous health checks

### Data Quality Emergency

1. **Quarantine**: Isolate affected data
2. **Validate**: Run schema and quality checks
3. **Notify**: Alert stakeholders
4. **Remediate**: Fix data quality issues
5. **Verify**: Confirm resolution

## Monitoring and Alerts

### Key Metrics to Monitor

- Risk assessment scores
- Signal quality metrics
- System performance indicators
- Data completeness rates

### Alert Thresholds

- **Critical**: Any risk score below threshold
- **Warning**: Degrading trend in risk scores
- **Info**: Successful risk assessment completion

## Contact Information

For escalation and support:
- **On-call Engineer**: [Contact details]
- **Team Lead**: [Contact details]
- **System Admin**: [Contact details]