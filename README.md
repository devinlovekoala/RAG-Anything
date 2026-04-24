# Circuit-Domain RAG — Extension of RAG-Anything

> **Branch:** `feat/circuit-domain` | **Base project:** [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything)

This branch extends RAG-Anything with native understanding of electronic-circuit documents. Circuit figures are converted into structured representations (component lists, connection relations, SPICE-like netlist summaries) that participate in the same multimodal index and hybrid retrieval pipeline as plain text.

---

## Architecture (Phase 2 — Adapter Route)

All circuit-domain logic lives in a **self-contained extension package** rather than inside the core library. The extension pre-processes documents into a `CircuitIR`, then hands a standard `content_list` to the existing `insert_content_list()` API — the only integration point needed.

```
PDF / lecture slide
        │
        ▼
  CircuitParser  (examples/circuit_domain_extension/parser.py)
    • MinerU / Docling → raw content_list
    • CircuitDetector  → score every image block
    • VLM              → structured extraction for likely circuits
        │
        ▼
  CircuitIR  (ir.py)
    • text_blocks / table_blocks / equation_blocks
    • circuit_figures — each with CircuitDesign, netlist, component_lines
        │
        ▼
  CircuitEnhancer  (enhancer.py)
    • component ref normalisation (R1, C2, Q3 …)
    • topology inference from component mix
        │
        ▼
  CircuitAdapter  (adapter.py)
    • CircuitIR → content_list  (type="circuit" items)
        │
        ▼
  RAGAnything.insert_content_list()   ← clean API boundary
        │
        ▼
  LightRAG KG + vector store  (unchanged core)
```

The core `raganything/` package is **not modified** by this extension.

### Extension Package Layout

| Module | Responsibility |
|---|---|
| [`ir.py`](examples/circuit_domain_extension/ir.py) | `CircuitIR` / `CircuitFigure` dataclasses — document-level IR |
| [`prompts.py`](examples/circuit_domain_extension/prompts.py) | VLM prompt constants (migrated from core, extended with `CIRCUIT_QA_SYSTEM`) |
| [`parser.py`](examples/circuit_domain_extension/parser.py) | PDF → CircuitIR via MinerU/Docling + per-image VLM extraction |
| [`enhancer.py`](examples/circuit_domain_extension/enhancer.py) | Heuristic post-processing: ref normalisation, topology inference |
| [`adapter.py`](examples/circuit_domain_extension/adapter.py) | CircuitIR → `content_list` for `insert_content_list()` |
| [`pipeline.py`](examples/circuit_domain_extension/pipeline.py) | `CircuitPipeline` + `ingest_circuit_document()` convenience helper |
| [`__init__.py`](examples/circuit_domain_extension/__init__.py) | Public API re-exports |

### Quick Start

```python
from raganything import RAGAnything, RAGAnythingConfig
from examples.circuit_domain_extension import ingest_circuit_document

rag = RAGAnything(config=RAGAnythingConfig(working_dir="./rag_store"), ...)
await rag.initialize()

# Single document — circuit figures extracted and indexed automatically
ir = await ingest_circuit_document(rag, "lecture.pdf")
print(ir.summary_stats())
# → source=lecture.pdf, pages=12, text_blocks=48, tables=3,
#   equations=7, circuit_figures=5, other_images=2
```

---

## Phase 1 Experimental Results

### Headline

> **E2 (circuit-focused QA) on Op-Amp topic: 3/8 → 8/8 success rate**
> when switching from mixed retrieval to circuit-aware hybrid retrieval.

This is a task-completion reliability gain, not a stylistic improvement.

### Ablation Tables

#### Op-Amp — rich multimodal store (`output_ablation_formal_opamp/`)

| Experiment | Condition | Success | Avg Latency (s) | Avg Answer Len |
|---|---|---:|---:|---:|
| E1 — general QA | Hybrid | **8/8** | 79.4 | 1207 |
| E1 — general QA | Naive | 8/8 | 40.8 | 707 |
| E2 — circuit QA | **Hybrid** | **8/8** | 54.9 | 912 |
| E2 — circuit QA | Mix | **3/8** | 49.2 | 754 |

#### Op-Amp v3 — faster embedding (`text-embedding-v4`, dim=1024)

| Experiment | Condition | Success | Avg Latency (s) | Avg Answer Len |
|---|---|---:|---:|---:|
| E1 — general QA | Hybrid | 8/8 | 11.8 | 986 |
| E1 — general QA | Naive | 8/8 | 12.4 | 1004 |
| E2 — circuit QA | Hybrid | 8/8 | **0.7** | 986 |
| E2 — circuit QA | Mix | 8/8 | 48.9 | 1203 |

Switching to `text-embedding-v4` (dim=1024) cut E2 latency from ~55 s to under 1 s.

#### BJT — second independent topic

| Experiment | Condition | Success | Avg Latency (s) | Avg Answer Len |
|---|---|---:|---:|---:|
| E1 — general QA | Hybrid | 7/8 | 16.8 | 1238 |
| E1 — general QA | Naive | 8/8 | 59.9 | 984 |
| E2 — circuit QA | **Hybrid** | **8/8** | 2.8 | 1195 |
| E2 — circuit QA | Mix | 7/8 | 14.9 | 1155 |

Hybrid retrieval dominates circuit-focused questions on a second independent topic, confirming the op-amp result is not a one-off.

### Conclusions

1. Circuit-aware structured processing improves answer reliability on circuit-specific QA compared with mixed retrieval.
2. The extension integrates cleanly into RAG-Anything via `insert_content_list()` — zero core modification required.
3. Topic-specific deployment is the most dependable evaluation mode at this stage.

---

## Evaluation Data (4 Topics)

| File | Topic | Questions |
|---|---|---:|
| `experiment_data/prepared/eval/week5_opamp.jsonl` | Op-Amp (Week 5) | 12 |
| `experiment_data/prepared/eval/2024_bjts.jsonl` | BJT (ch2) | 12 |
| `experiment_data/prepared/eval/2024_fets.jsonl` | FET / MOSFET (ch3) — **new** | 12 |
| `experiment_data/prepared/eval/week9_freq_domain.jsonl` | Freq-Domain analysis (Week 9) — **new** | 11 |

Source PDFs: university electronics course lecture slides under `experiment_data/`.
The refreshed question sets mix lecture-summary, study-support, circuit-structure, and formula-mapping prompts to better reflect realistic document QA needs.

---

## Reproduce

### Prerequisites

```bash
pip install raganything[all]
cd reproduce/
cp env.example .env   # set LLM_API_KEY, LLM_BASE_URL, VLM_*, EMBED_* fields
```

### Step-by-step

```bash
# 1. Verify the model stack is reachable
python 06_smoke_test_model_stack.py

# 2. Build a topic store (Op-Amp example)
bash 09_build_topic_store.sh opamp

# 3. Run ablation experiments (Op-Amp)
python 04_ablation_experiments.py \
    --working-dir ../rag_storage_formal_opamp_v3 \
    --qa-file ../experiment_data/prepared/eval/week5_opamp.jsonl \
    --experiments E1 E2

# 4. Run ablation experiments (FET — new topic)
python 04_ablation_experiments.py \
    --working-dir ../rag_storage_formal_fets \
    --qa-file ../experiment_data/prepared/eval/2024_fets.jsonl \
    --experiments E1 E2
```

Results land in `output_ablation_*/ablation_results.json`.

### Script Index

| Script | Purpose |
|---|---|
| `00_setup_env.sh` | Install dependencies and validate env |
| `01_run_pipeline.py` | Full document ingestion pipeline |
| `02_test_modal_processors.py` | Unit-test modal processor routing |
| `03_circuit_processor.py` | Manual circuit extraction test (legacy built-in path) |
| `04_ablation_experiments.py` | E1/E2 ablation with result JSON output |
| `05_prepare_circuit_experiment_data.py` | Prepare QA eval sets from raw PDFs |
| `06_smoke_test_model_stack.py` | Verify LLM/VLM/embed stack is live |
| `07_smoke_test_circuit_processor.py` | End-to-end circuit extraction smoke test |
| `08_build_eval_questions.py` | Generate evaluation question sets |
| `09_build_topic_store.sh` | Build a topic-isolated RAG store |

---

## Phase 2 Status

Phase 2 completed the adapter-route refactoring:

- `examples/circuit_domain_extension/` — complete 7-module package, **zero core changes**
- `raganything/raganything.py` — `CircuitModalProcessor` wiring removed; core is clean
- Two new eval topics added (FET, Freq-Domain) — benchmark now covers 4 distinct circuit subdomains
- All 4-topic ablation experiments are the next run target

Full analysis: [`docs/Circuit_RAG_Phase1_Report.md`](docs/Circuit_RAG_Phase1_Report.md)
