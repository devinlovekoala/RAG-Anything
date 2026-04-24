#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"

python_bin="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
build_script="$REPO_ROOT/reproduce/09_build_topic_store.sh"
ablation_script="$REPO_ROOT/reproduce/04_ablation_experiments.py"

run_build="${RUN_BUILD:-1}"
run_ablation="${RUN_ABLATION:-1}"
experiments="${EXPERIMENTS:-E1 E2}"

declare -a topics=(
  "bjt|experiment_data/2024-ch2-BJTs.pdf|./rag_storage_benchmark_bjt_v1|./output_benchmark_bjt_v1|$ROOT_DIR/runtime/benchmark_runtime_bjt.jsonl|./output_ablation_benchmark_bjt_v1"
  "fet|experiment_data/2024-ch3-FETs-Enhance.pdf|./rag_storage_benchmark_fet_v1|./output_benchmark_fet_v1|$ROOT_DIR/runtime/benchmark_runtime_fet.jsonl|./output_ablation_benchmark_fet_v1"
  "freq_domain|experiment_data/2024-ch5-frequency.pdf|./rag_storage_benchmark_freq_domain_v1|./output_benchmark_freq_domain_v1|$ROOT_DIR/runtime/benchmark_runtime_freq_domain.jsonl|./output_ablation_benchmark_freq_domain_v1"
  "opamp|experiment_data/2024-ch8-op amp.pdf|./rag_storage_benchmark_opamp_v1|./output_benchmark_opamp_v1|$ROOT_DIR/runtime/benchmark_runtime_opamp.jsonl|./output_ablation_benchmark_opamp_v1"
)

cd "$REPO_ROOT"

for entry in "${topics[@]}"; do
  IFS='|' read -r topic doc_path working_dir topic_output qa_file ablation_output <<< "$entry"

  echo
  echo "============================================================"
  echo "TOPIC: $topic"
  echo "============================================================"

  if [[ "$run_build" == "1" ]]; then
    bash "$build_script" "$doc_path" "$working_dir" "$topic_output"
  fi

  if [[ "$run_ablation" == "1" ]]; then
    "$python_bin" "$ablation_script" \
      --working-dir "$working_dir" \
      --output "$ablation_output" \
      --qa-file "$qa_file" \
      --experiments $experiments
  fi
done
