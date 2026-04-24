# Circuit-Domain RAG Evaluation

> Branch: `feat/circuit-domain`
> Base project: [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything)

This branch extends RAG-Anything for electronic-circuit document QA and provides a reproducible evaluation package for undergraduate electronics lecture materials. The current stage focuses on circuit-domain ingestion, topic-isolated RAG stores, E1/E2 ablation experiments, and an 80-question answer-quality benchmark.

The latest stage result is documented in:

```text
experiment_data/prepared/new-data/reports/stage_result_report.md
```

## Current Stage Result

The frozen stage benchmark contains 80 Chinese QA items across four electronics topics:

| Topic | Source document | Questions |
|---|---|---:|
| BJT | `experiment_data/2024-ch2-BJTs.pdf` | 20 |
| FET | `experiment_data/2024-ch3-FETs-Enhance.pdf` | 20 |
| Frequency domain | `experiment_data/2024-ch5-frequency.pdf` | 20 |
| Op-Amp | `experiment_data/2024-ch8-op amp.pdf` | 20 |

Main E1 answer-quality comparison:

| Metric | Baseline (`naive`) | Enhanced (`hybrid`) | Delta |
|---|---:|---:|---:|
| Overall rule-based score | 0.413 | 0.443 | +0.030 |

Topic-level deltas:

| Topic | Baseline | Enhanced | Delta |
|---|---:|---:|---:|
| BJT | 0.347 | 0.381 | +0.034 |
| FET | 0.404 | 0.407 | +0.003 |
| Frequency domain | 0.417 | 0.492 | +0.075 |
| Op-Amp | 0.482 | 0.493 | +0.011 |

Category-level improvements are concentrated in topology, reasoning, concept, and cross-modal questions. Factoid and mapping questions are largely tied, which is expected because many of them are answerable from direct textual evidence.

All four topic stores completed successfully with `multimodal_processed=true`. E1 and E2 reached `20/20` query success rate for all four topics, and no query-level `error` was found in the generated ablation outputs.

## How To Reproduce

### 1. Prepare Environment

From the repository root:

```bash
cd /home/devin/Workspace/HKUDS-project/RAG-Anything
bash reproduce/00_setup_env.sh
cp reproduce/env.example reproduce/.env
```

Then edit `reproduce/.env` and set the LLM, VLM, and embedding credentials used by your environment.

Run a quick stack check:

```bash
./.venv/bin/python reproduce/06_smoke_test_model_stack.py
```

### 2. Prepare Input Documents

The stage benchmark expects the source PDFs below to exist under `experiment_data/`:

```text
experiment_data/2024-ch2-BJTs.pdf
experiment_data/2024-ch3-FETs-Enhance.pdf
experiment_data/2024-ch5-frequency.pdf
experiment_data/2024-ch8-op amp.pdf
```

These PDFs are local experiment inputs. The versioned benchmark artifacts preserve the question set, reference answers, scoring protocol, generated answers, and report.

### 3. Run The Full Benchmark

This command builds topic-isolated stores and runs E1/E2 ablations for all four benchmark topics:

```bash
cd /home/devin/Workspace/HKUDS-project/RAG-Anything
bash experiment_data/prepared/new-data/run_benchmark_experiments.sh
```

Expected output directories:

```text
rag_storage_benchmark_bjt_v1/
rag_storage_benchmark_fet_v1/
rag_storage_benchmark_freq_domain_v1/
rag_storage_benchmark_opamp_v1/
output_ablation_benchmark_bjt_v1/
output_ablation_benchmark_fet_v1/
output_ablation_benchmark_freq_domain_v1/
output_ablation_benchmark_opamp_v1/
```

### 4. Run Local Answer-Quality Scoring

This step does not call the LLM. It exports answers from the ablation files and runs local rule-based scoring.

```bash
cd /home/devin/Workspace/HKUDS-project/RAG-Anything
bash experiment_data/prepared/new-data/run_benchmark_quality_eval.sh
```

Generated artifacts:

```text
experiment_data/prepared/new-data/results/results_baseline.json
experiment_data/prepared/new-data/results/results_enhanced.json
experiment_data/prepared/new-data/results/benchmark_report.json
```

## Single-Topic Commands

The full benchmark script is recommended for report-level reproduction. For debugging, a single topic can be run manually.

Example: BJT topic store

```bash
cd /home/devin/Workspace/HKUDS-project/RAG-Anything
bash reproduce/09_build_topic_store.sh \
  "experiment_data/2024-ch2-BJTs.pdf" \
  "./rag_storage_benchmark_bjt_v1" \
  "./output_benchmark_bjt_v1"
```

Example: BJT E1/E2 ablation

```bash
cd /home/devin/Workspace/HKUDS-project/RAG-Anything
./.venv/bin/python reproduce/04_ablation_experiments.py \
  --working-dir ./rag_storage_benchmark_bjt_v1 \
  --output ./output_ablation_benchmark_bjt_v1 \
  --qa-file ./experiment_data/prepared/new-data/runtime/benchmark_runtime_bjt.jsonl \
  --experiments E1 E2
```

## Benchmark Assets

| Path | Purpose |
|---|---|
| `experiment_data/prepared/new-data/benchmark_master.jsonl` | Frozen 80-question master benchmark with references and rubrics |
| `experiment_data/prepared/new-data/benchmark_manifest.json` | Topic-to-document and output-directory mapping |
| `experiment_data/prepared/new-data/runtime/` | Per-topic JSONL files consumed by ablation scripts |
| `experiment_data/prepared/new-data/results/` | Exported baseline/enhanced answers and score report |
| `experiment_data/prepared/new-data/reports/stage_result_report.md` | Professional stage report |
| `experiment_data/prepared/new-data/prepare_benchmark_assets.py` | Regenerates runtime files and result templates from the master benchmark |
| `experiment_data/prepared/new-data/export_ablation_results.py` | Converts ablation outputs into standardized result files |
| `experiment_data/prepared/new-data/benchmark_eval.py` | Rule-based answer-quality scorer |

## Legacy 4-Topic Smoke Eval

Before the frozen 80-question benchmark, the repository also contains lighter per-topic eval files:

| File | Topic | Questions |
|---|---|---:|
| `experiment_data/prepared/eval/week5_opamp.jsonl` | Op-Amp (Week 5) | 12 |
| `experiment_data/prepared/eval/2024_bjts.jsonl` | BJT (ch2) | 12 |
| `experiment_data/prepared/eval/2024_fets.jsonl` | FET / MOSFET (ch3) | 12 |
| `experiment_data/prepared/eval/week9_freq_domain.jsonl` | Frequency-domain analysis (Week 9) | 11 |

These files are useful for fast smoke tests and development checks. The report-level result should use `experiment_data/prepared/new-data/`.

## Reproduction Script Index

| Script | Purpose |
|---|---|
| `reproduce/00_setup_env.sh` | Install dependencies and validate environment setup |
| `reproduce/01_run_pipeline.py` | Full document ingestion pipeline |
| `reproduce/02_test_modal_processors.py` | Modal processor routing test |
| `reproduce/03_circuit_processor.py` | Manual circuit extraction test |
| `reproduce/04_ablation_experiments.py` | E1/E2 ablation runner with JSON output |
| `reproduce/05_prepare_circuit_experiment_data.py` | Prepare circuit experiment data and query templates |
| `reproduce/06_smoke_test_model_stack.py` | Verify LLM, VLM, and embedding stack availability |
| `reproduce/07_smoke_test_circuit_processor.py` | End-to-end circuit extraction smoke test |
| `reproduce/08_build_eval_questions.py` | Build lightweight eval question sets |
| `reproduce/09_build_topic_store.sh` | Build topic-isolated RAG store; signature is `pdf_path working_dir output_dir` |

## Data Limitations

The current benchmark is a stage-level validation, not a final large-scale professional benchmark. The source documents are undergraduate lecture slides, so the data scale, document quality, and difficulty are limited:

- The benchmark contains 80 QA items across four lecture documents.
- Slide-style teaching materials often have sparse explanations, incomplete derivations, and limited visual annotations.
- The difficulty level is closer to undergraduate instruction than professional engineering datasets such as datasheets, design notes, SPICE reports, or advanced circuit design cases.
- Some questions are answerable from local textual evidence alone, which naturally compresses the gap between baseline and enhanced retrieval.

The observed improvement should therefore be read conservatively as a lower-bound stage signal. Larger, higher-quality, and more professionally demanding circuit datasets may better reveal the value of circuit-aware multimodal processing and graph retrieval.
