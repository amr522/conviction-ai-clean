# GitHub Actions Workflows

This directory contains automated workflows for the Conviction AI trading system.

## Workflows

### 1. Nightly Strategy Recommendations (`nightly_recommendations.yml`)

**Schedule:** Daily at 15:00 UTC (11:00 ET / 10:00 ET during DST)

**Purpose:** Generates daily strategy recommendations based on current market conditions and IV/HV analysis.

**Triggers:**
- Scheduled: `cron: '0 15 * * *'`
- Manual: `workflow_dispatch`

**Outputs:**
- `strategy_recs.json` - JSON file containing daily recommendations
- Log files for debugging
- Workflow summary with preview of recommendations

### 2. Weekly Backtest Analysis (`weekly_backtest.yml`)

**Schedule:** Monday at 22:00 UTC (18:00 ET / 17:00 ET during DST)

**Purpose:** Runs weekly backtest analysis on the previous week's trading period (Monday to Monday).

**Triggers:**
- Scheduled: `cron: '0 22 * * MON'`
- Manual: `workflow_dispatch`

**Outputs:**
- `backtest_*.json` - Backtest results in JSON format
- `backtest_*.csv` - Detailed trade data (if generated)
- Log files for debugging
- Workflow summary with performance metrics

## Required GitHub Secrets

To enable these workflows, configure the following secrets in your GitHub repository settings:

### AWS Credentials
- `AWS_ACCESS_KEY_ID` - AWS access key for S3 and other AWS services
- `AWS_SECRET_ACCESS_KEY` - AWS secret access key
- `AWS_REGION` - AWS region (e.g., `us-east-1`)

### Market Data API
- `POLYGON_API_KEY` - Polygon.io API key for option chain data

## Setup Instructions

1. **Configure GitHub Secrets:**
   - Go to repository Settings → Secrets and variables → Actions
   - Add all required secrets listed above

2. **Verify Workflows:**
   - Check the Actions tab after pushing these files
   - Workflows should appear and show scheduled runs

3. **Test Manually:**
   - Use "Run workflow" button for immediate testing
   - Check artifact downloads and summary reports

## Artifacts

Both workflows upload artifacts that are retained for different periods:

- **Nightly Recommendations:** 30 days retention
- **Weekly Backtests:** 90 days retention

Artifacts can be downloaded from the workflow run page in GitHub Actions.

## Troubleshooting

### Common Issues

1. **Missing Secrets:** Ensure all required secrets are configured
2. **API Rate Limits:** Check Polygon API usage if workflows fail
3. **S3 Access:** Verify AWS credentials have proper S3 permissions
4. **Time Zone Issues:** All schedules use UTC time

### Debugging

- Check workflow logs in GitHub Actions tab
- Download artifacts to review generated files
- Use manual triggers to test changes

## Time Zone Reference

| Schedule | UTC Time | ET (Standard) | ET (Daylight) |
|----------|----------|---------------|---------------|
| Nightly Recs | 15:00 | 11:00 AM | 10:00 AM |
| Weekly Backtest | 22:00 Mon | 6:00 PM Mon | 5:00 PM Mon |

## Workflow Dependencies

Both workflows require:
- Python 3.10
- `requirements.txt` dependencies
- Access to `strategy_selector.py` and `backtest_harness.py`
- AWS S3 for data storage
- Polygon API for market data
