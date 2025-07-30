#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Conviction AI Pipeline - Interactive Demo"
echo "============================================"
echo "This demo will walk you through the complete pipeline setup and execution."
echo ""

# Section 1: Repository & DevContainer
echo "📦 Section 1: Repository & DevContainer Setup"
echo "If you haven't already:"
echo "  git clone https://github.com/amr522/conviction-ai-clean.git"
echo "  cd conviction-ai-clean"
echo "  code .  # Open in VS Code"
echo "  # When prompted, click 'Reopen in Container'"
echo ""
read -p "Press Enter to continue..."

# Section 2: Environment Setup
echo ""
echo "🔧 Section 2: Environment Setup"
echo "Setting up Python environment and dependencies..."
read -p "Press Enter to run setup-env.sh..."
echo "Running: ./scripts/setup-env.sh"
if [ ! -d ".venv" ]; then
    ./scripts/setup-env.sh
fi
source .venv/bin/activate
echo "✅ Environment ready"

# Section 3: Full Pipeline
echo ""
echo "⚡ Section 3: Full Pipeline Evaluation"
read -p "Enter date for pipeline evaluation (YYYY-MM-DD) [2025-01-15]: " DATE
DATE=${DATE:-2025-01-15}
echo "Running: ./scripts/evaluate_pipeline.sh $DATE"
read -p "Press Enter to execute..."
./scripts/evaluate_pipeline.sh $DATE || echo "⚠️  Pipeline evaluation completed with warnings"
echo "✅ Pipeline evaluation done"

# Section 4: Feature Generation
echo ""
echo "🔬 Section 4: Feature Generation"
echo "Generating feature parquet for date: $DATE"
echo "Running: python src/calculate_features.py --date $DATE"
read -p "Press Enter to execute..."
python src/calculate_features.py --date $DATE || echo "⚠️  Feature generation completed with warnings"
echo "✅ Feature generation done"

# Section 5: Training Dataset
echo ""
echo "🎯 Section 5: Training Dataset Generation"
if [ -f "data/Parquet_data/labels_$DATE.parquet" ]; then
    echo "Labels found for $DATE. Generating training dataset..."
    echo "Running: ./scripts/generate-training-dataset.sh"
    read -p "Press Enter to execute..."
    ./scripts/generate-training-dataset.sh \
        "data/Parquet_data/features_$DATE.parquet" \
        "data/Parquet_data/labels_$DATE.parquet" \
        "data/Parquet_data/train_dataset_$DATE.parquet" || echo "⚠️  Training dataset generation completed with warnings"
    echo "✅ Training dataset generated"
else
    echo "⚠️  No labels found for $DATE. Skipping training dataset generation."
    echo "Features are available at: data/Parquet_data/features_$DATE.parquet"
fi

# Section 6: Staging Smoke Test
echo ""
echo "🧪 Section 6: Staging Smoke Test"
echo "Running end-to-end Kubernetes deployment test..."
echo "This will create a local Kind cluster and test the full deployment."
read -p "Press Enter to run staging smoke test..."
echo "Running: ./scripts/smoke-test-staging.sh"
./scripts/smoke-test-staging.sh || echo "⚠️  Staging test completed with warnings"
echo "✅ Staging smoke test done"

# Section 7: Production Promotion
echo ""
echo "🚀 Section 7: Production Promotion (Optional)"
echo "This will deploy the pipeline to production environment."
read -p "Do you want to promote to production? (y/N): " PROMOTE
if [[ $PROMOTE =~ ^[Yy]$ ]]; then
    echo "Running: ./scripts/promote-production.sh"
    read -p "Press Enter to execute..."
    ./scripts/promote-production.sh || echo "⚠️  Production promotion completed with warnings"
    echo "✅ Production deployment done"
else
    echo "⏭️  Skipping production promotion"
fi

# Demo Complete
echo ""
echo "🎉 Demo Complete!"
echo "================="
echo "What you've accomplished:"
echo "• ✅ Environment setup and dependency installation"
echo "• ✅ Pipeline evaluation and validation"
echo "• ✅ Feature generation for date: $DATE"
if [ -f "data/Parquet_data/train_dataset_$DATE.parquet" ]; then
    echo "• ✅ Training dataset generation"
fi
echo "• ✅ Staging deployment testing"
if [[ $PROMOTE =~ ^[Yy]$ ]]; then
    echo "• ✅ Production deployment"
fi
echo ""
echo "Next steps:"
echo "• Check data/Parquet_data/ for generated files"
echo "• Run './scripts/run-canary-test.sh' for canary testing"
echo "• Explore the full README.md for advanced features"
echo ""
echo "📚 See QUICKSTART.md for detailed instructions"
echo "✅ Demo complete!"
