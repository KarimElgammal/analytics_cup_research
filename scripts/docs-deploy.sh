#!/bin/bash
# Deploy MkDocs documentation to GitHub Pages
# Usage: ./scripts/docs-deploy.sh

set -e

cd "$(dirname "$0")/.."

echo "Building and deploying documentation to GitHub Pages..."
echo "Target: https://karimelgammal.github.io/analytics_cup_research/"
echo ""

# Build first to catch any errors
echo "Step 1: Building documentation..."
PYTHONPATH=. uv run mkdocs build

echo ""
echo "Step 2: Deploying to gh-pages branch..."
PYTHONPATH=. uv run mkdocs gh-deploy --force

echo ""
echo "Deployment complete!"
echo "Visit: https://karimelgammal.github.io/analytics_cup_research/"
