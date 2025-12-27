#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PYTHONPATH=. uv run mkdocs serve
