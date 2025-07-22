#!/bin/bash

# GitHub Actions Workflow Validation Script
# Tests the commands that will be used in the workflows

echo "=== GitHub Actions Workflows Validation ==="
echo ""

echo "1. NIGHTLY RECOMMENDATIONS WORKFLOW"
echo "   Schedule: 0 15 * * * (15:00 UTC daily)"
echo "   Current UTC time: $(date -u)"
echo ""

# Test nightly recommendations command
CURRENT_DATE=$(date -u +'%Y-%m-%d')
echo "   Command: python strategy_selector.py --date $CURRENT_DATE"
echo "   Date parameter: $CURRENT_DATE"
echo ""

echo "2. WEEKLY BACKTEST WORKFLOW"
echo "   Schedule: 0 22 * * MON (22:00 UTC every Monday)"
echo ""

# Test weekly backtest date calculation (Linux/Ubuntu compatible)
echo "   Calculating backtest date range..."
LAST_MONDAY=$(date -u -d 'last monday' +'%Y-%m-%d' 2>/dev/null)
if [ $? -eq 0 ]; then
    START_DATE=$(date -u -d "$LAST_MONDAY - 7 days" +'%Y-%m-%d')
    echo "   Last Monday: $LAST_MONDAY"
    echo "   Previous Monday: $START_DATE"
    echo "   Command: python backtest_harness.py --start $START_DATE --end $LAST_MONDAY"
else
    echo "   Note: GNU date not available (MacOS), but workflows use Ubuntu which has GNU date"
    echo "   Example: python backtest_harness.py --start 2025-07-14 --end 2025-07-21"
fi

echo ""
echo "3. REQUIRED GITHUB SECRETS"
echo "   - AWS_ACCESS_KEY_ID"
echo "   - AWS_SECRET_ACCESS_KEY" 
echo "   - AWS_REGION"
echo "   - POLYGON_API_KEY"
echo ""

echo "4. ARTIFACTS GENERATED"
echo "   Nightly: strategy_recs.json (30 days retention)"
echo "   Weekly: backtest_*.json, backtest_*.csv (90 days retention)"
echo ""

echo "=== Validation Complete ==="
