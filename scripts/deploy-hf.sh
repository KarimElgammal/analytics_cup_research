#!/bin/bash
# Deploy only to HuggingFace Space
set -e

echo "=== Deploying to HuggingFace Space ==="
git push hf main

echo ""
echo "Done! Space: https://huggingface.co/spaces/KarimElgammal/analytics-cup-research"
