#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <pdf_path> <working_dir> <output_dir>"
  exit 1
fi

PDF_PATH="$1"
WORKING_DIR="$2"
OUTPUT_DIR="$3"

# More conservative defaults for long circuit lecture PDFs.
# Smaller chunks reduce entity-extraction timeouts on dense teaching material.
export MAX_ASYNC="${MAX_ASYNC:-2}"
export EMBEDDING_FUNC_MAX_ASYNC="${EMBEDDING_FUNC_MAX_ASYNC:-4}"
export MAX_PARALLEL_INSERT="${MAX_PARALLEL_INSERT:-1}"
export CHUNK_SIZE="${CHUNK_SIZE:-800}"
export CHUNK_OVERLAP_SIZE="${CHUNK_OVERLAP_SIZE:-120}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-300}"

exec ./.venv/bin/python -u reproduce/01_run_pipeline.py \
  "$PDF_PATH" \
  --working-dir "$WORKING_DIR" \
  --output "$OUTPUT_DIR"
