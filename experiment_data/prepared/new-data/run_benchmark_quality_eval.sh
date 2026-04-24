#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"

python_bin="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

"$python_bin" "$ROOT_DIR/export_ablation_results.py" \
  --benchmark-master "$ROOT_DIR/benchmark_master.jsonl" \
  --experiment "${EXPERIMENT:-E1}" \
  --baseline-condition "${BASELINE_CONDITION:-naive}" \
  --enhanced-condition "${ENHANCED_CONDITION:-hybrid}" \
  --topic-result "bjt=${BJT_RESULT:-$REPO_ROOT/output_ablation_benchmark_bjt_v1/ablation_results.json}" \
  --topic-result "fet=${FET_RESULT:-$REPO_ROOT/output_ablation_benchmark_fet_v1/ablation_results.json}" \
  --topic-result "freq_domain=${FREQ_RESULT:-$REPO_ROOT/output_ablation_benchmark_freq_domain_v1/ablation_results.json}" \
  --topic-result "opamp=${OPAMP_RESULT:-$REPO_ROOT/output_ablation_benchmark_opamp_v1/ablation_results.json}" \
  --baseline-out "$ROOT_DIR/results/results_baseline.json" \
  --enhanced-out "$ROOT_DIR/results/results_enhanced.json"

"$python_bin" "$ROOT_DIR/benchmark_eval.py" \
  --baseline "$ROOT_DIR/results/results_baseline.json" \
  --enhanced "$ROOT_DIR/results/results_enhanced.json" \
  --output "$ROOT_DIR/results/benchmark_report.json"
