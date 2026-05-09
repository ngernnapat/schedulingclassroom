#!/bin/bash

# Script to allow public access to all Firebase Functions
# Firebase Functions v2 (2nd gen) run on Cloud Run, so we use Cloud Run IAM policy binding

set -e

# Get project ID from gcloud config or use default
PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "schedule-optimization-d83ea")
REGION="us-central1"

echo "🔓 Allowing public access to all functions..."
echo "📋 Project: ${PROJECT_ID}"
echo "🌍 Region: ${REGION}"
echo ""

FAILED=0
while IFS= read -r SERVICE; do
    [ -z "$SERVICE" ] && continue
    echo "Setting public access for ${SERVICE}..."
    if gcloud run services add-iam-policy-binding "$SERVICE" \
        --region="${REGION}" \
        --member="allUsers" \
        --role="roles/run.invoker" \
        --project="${PROJECT_ID}"; then
        echo "   ✅ ${SERVICE}"
    else
        echo "   ⚠️  ${SERVICE} (skipped or already public)"
        FAILED=$((FAILED + 1))
    fi
done < <(gcloud run services list --region="${REGION}" --project="${PROJECT_ID}" --format="value(metadata.name)")

echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "⚠️  Some services could not be updated (may already allow public access)."
fi
echo "✅ Done. List services with:"
echo "   gcloud run services list --region=${REGION} --project=${PROJECT_ID}"
