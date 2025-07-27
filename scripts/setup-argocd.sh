#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setting up ArgoCD for GitOps deployment..."

echo "1️⃣ Installing ArgoCD"
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

echo "2️⃣ Waiting for ArgoCD to be ready"
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n argocd

echo "3️⃣ Creating ArgoCD Applications"
kubectl apply -f gitops/argocd-app-staging.yaml
kubectl apply -f gitops/argocd-app-production.yaml

echo "4️⃣ Getting ArgoCD admin password"
ARGOCD_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)
echo "ArgoCD admin password: $ARGOCD_PASSWORD"

echo "5️⃣ Port-forward ArgoCD server (optional)"
echo "Run: kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo "Access: https://localhost:8080 (admin / $ARGOCD_PASSWORD)"

echo "✅ ArgoCD setup complete! Applications will sync automatically."