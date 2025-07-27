# Quickstart Guide

Welcome to the Conviction AI Pipeline! This guide takes you from zero to a running pipeline in minutes using DevContainer and local Kubernetes (Kind).

## Prerequisites
- Git (v2.25+)
- Docker (v20.10+)
- Kind (v0.17+) or Minikube
- VS Code with Remote - Containers extension
- kubectl (v1.21+)

## 1. Clone the Repository

```bash
git clone https://github.com/amr522/conviction-ai-clean.git
cd conviction-ai-clean
```

## 2. Launch DevContainer
1. Open VS Code in this folder: `code .`
2. When prompted, **Reopen in Container**.
3. DevContainer will provision:
   - Python 3.10 virtualenv
   - All required dependencies
   - Pre-commit hooks
   - Dev tooling (Black, isort, mypy)

## 3. Local Environment Setup

In the DevContainer terminal:

```bash
# One-time bootstrap
./scripts/setup-env.sh
source .venv/bin/activate
```

## 4. Run the Full Pipeline Locally

```bash
# Replace YYYY-MM-DD with a sample date
./scripts/evaluate_pipeline.sh 2025-01-15
```

This runs:
- ETL dry-run with schema validation
- Parquet schema inspection
- Feature smoke tests
- Quick training smoke-test

## 5. Generate Features & Training Dataset

```bash
# Generate feature Parquet
python src/calculate_features.py --date 2025-01-15

# Generate training dataset
python src/generate_training_dataset.py \
  --feature-path data/Parquet_data/features_2025-01-15.parquet \
  --label-path data/Parquet_data/labels_2025-01-15.parquet
```

## 6. Deploy to Staging Smoke Test

```bash
# Use local Kind cluster and run end-to-end smoke test
./scripts/smoke-test-staging.sh
```

## 7. Promote to Production

```bash
# Manual approval or run script
env \
  KUBECONFIG=~/.kube/config \
  SLACK_WEBHOOK_URL=... \
  S3_BUCKET=... \
  PUSHGATEWAY_URL=... \
  helm upgrade conviction-ai-pipeline charts/conviction-ai-pipeline --install --namespace production

# Verify rollout
./scripts/promote-production.sh
```

## 8. Demo Quickstart Script

For a live 2-minute walkthrough, run:

```bash
./scripts/demo-quickstart.sh
```

This interactive script steps you through:
1. DevContainer setup
2. Pipeline evaluation
3. Staging smoke test
4. Production promotion

**Enjoy your Conviction AI end-to-end workflow!**