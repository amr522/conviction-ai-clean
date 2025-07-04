#!/bin/bash
set -euo pipefail

EMAIL=${1:-"amr522@gmail.com"}

echo "🚀 Enhancing HPO Pipeline with all 5 suggested improvements..."

cd "$(dirname "$0")/.."

echo "1️⃣ Setting up S3 versioning and dataset backup..."
./scripts/backup_successful_dataset.sh

echo "2️⃣ Setting up CloudWatch monitoring for HPO failures..."
./scripts/setup_hpo_monitoring.sh "$EMAIL"

echo "3️⃣ Deploying best model to SageMaker endpoint..."
python scripts/deploy_best_model.py --endpoint-name conviction-best-hpo-model

echo "4️⃣ Running initial cleanup (dry-run first)..."
python scripts/automated_cleanup.py --dry-run
echo "Running actual cleanup..."
python scripts/automated_cleanup.py

echo "5️⃣ Optimal hyperparameters documented in docs/optimal_hyperparameters.md"

echo "✅ All 5 HPO pipeline enhancements completed successfully!"
echo ""
echo "📋 Summary:"
echo "  - S3 versioning enabled with dataset backup"
echo "  - CloudWatch alarms monitoring HPO job failures"
echo "  - Best model deployed to SageMaker endpoint"
echo "  - Automated cleanup configured"
echo "  - Optimal hyperparameters documented"
