#!/usr/bin/env bash
set -euo pipefail

# Kubernetes deployment script for Conviction AI Pipeline
NAMESPACE=${NAMESPACE:-conviction-ai}
RELEASE_NAME=${RELEASE_NAME:-conviction-ai-pipeline}
CHART_PATH=${CHART_PATH:-charts/conviction-ai-pipeline}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    if ! command -v helm &> /dev/null; then
        log_error "helm not found. Please install Helm 3.0+."
        exit 1
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
}

# Create secrets
create_secrets() {
    log_info "Creating pipeline secrets..."

    # Check if secrets already exist
    if kubectl get secret pipeline-secrets -n "$NAMESPACE" &> /dev/null; then
        log_warn "Secret 'pipeline-secrets' already exists. Skipping creation."
        return
    fi

    # Prompt for required values
    read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
    read -s -p "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
    echo
    read -p "S3 Bucket: " S3_BUCKET
    read -p "Slack Webhook URL (optional): " SLACK_WEBHOOK_URL
    read -p "MLflow Tracking URI (optional): " MLFLOW_TRACKING_URI

    kubectl create secret generic pipeline-secrets \
        --from-literal=aws-access-key-id="$AWS_ACCESS_KEY_ID" \
        --from-literal=aws-secret-access-key="$AWS_SECRET_ACCESS_KEY" \
        --from-literal=s3-bucket="$S3_BUCKET" \
        --from-literal=slack-webhook-url="${SLACK_WEBHOOK_URL:-}" \
        --from-literal=mlflow-tracking-uri="${MLFLOW_TRACKING_URI:-}" \
        -n "$NAMESPACE"

    log_info "Secrets created successfully"
}

# Deploy chart
deploy_chart() {
    log_info "Deploying Helm chart..."

    # Get current date for default run date
    RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}

    helm upgrade --install "$RELEASE_NAME" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --set runDate="$RUN_DATE" \
        --set nTrials="${N_TRIALS:-50}" \
        --set nJobs="${N_JOBS:-8}" \
        --set image.tag="${IMAGE_TAG:-latest}" \
        --wait \
        --timeout=10m

    log_info "Chart deployed successfully"
}

# Show status
show_status() {
    log_info "Deployment status:"

    echo "Namespace: $NAMESPACE"
    echo "Release: $RELEASE_NAME"
    echo

    kubectl get all -n "$NAMESPACE" -l app.kubernetes.io/name=conviction-ai-pipeline

    echo
    log_info "To view logs:"
    echo "kubectl logs -n $NAMESPACE -l component=pipeline -f"

    echo
    log_info "To run tests:"
    echo "helm test $RELEASE_NAME -n $NAMESPACE"
}

# Main execution
main() {
    log_info "🚀 Deploying Conviction AI Pipeline to Kubernetes"

    check_prerequisites
    create_namespace
    create_secrets
    deploy_chart
    show_status

    log_info "✅ Deployment completed successfully!"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "status")
        kubectl get all -n "$NAMESPACE" -l app.kubernetes.io/name=conviction-ai-pipeline
        ;;
    "logs")
        kubectl logs -n "$NAMESPACE" -l component=pipeline -f
        ;;
    "delete")
        log_warn "Deleting deployment..."
        helm uninstall "$RELEASE_NAME" -n "$NAMESPACE"
        kubectl delete namespace "$NAMESPACE"
        log_info "Deployment deleted"
        ;;
    *)
        echo "Usage: $0 [deploy|status|logs|delete]"
        exit 1
        ;;
esac
