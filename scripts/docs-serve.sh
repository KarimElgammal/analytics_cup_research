#!/bin/bash
# Serve MkDocs documentation locally
# Usage: ./scripts/docs-serve.sh

set -e

cd "$(dirname "$0")/.."

echo "Starting MkDocs development server..."
echo "Documentation will be available at: http://127.0.0.1:8000"
echo "Press Ctrl+C to stop"
echo ""

PYTHONPATH=. uv run mkdocs serve
