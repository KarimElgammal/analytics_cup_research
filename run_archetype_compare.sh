#!/bin/bash
# Run the Archetype Comparison Streamlit app

cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run archetype_compare.py "$@"
