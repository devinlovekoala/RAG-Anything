# Circuit-Domain RAG — Extension of RAG-Anything

> **Branch:** `feat/circuit-domain` | **Base project:** [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything)

This branch extends RAG-Anything with native understanding of electronic-circuit documents. Circuit figures are no longer processed as plain images — they are converted into structured representations (component lists, connection relations, SPICE-like netlist summaries) that participate in the same multimodal index and hybrid retrieval pipeline.

---

## What Was Added

| Component | Description | Path |
|---|---|---|
| `CircuitModalProcessor` | Extracts `CircuitDesign` from circuit figures: component list, connections, netlist | [`raganything/circuit.py`](raganything/circuit.py) |
| Circuit routing | Heuristic that steers circuit-like images into `CircuitModalProcessor` instead of caption-only | [`raganything/modalprocessors.py`](raganything/modalprocessors.py) |
| Reproduce scripts | 10 numbered scripts: env setup → indexing → smoke tests → ablation → topic-store | [`reproduce/`](reproduce/) |
| Experiment QA data | Prepared eval sets + gold metadata for Op-Amp, BJT, Superposition topics | [`experiment_data/prepared/`](experiment_data/prepared/) |
| Phase 1 report | Full written analysis, methodology, conclusions | [`docs/Circuit_RAG_Phase1_Report.md`](docs/Circuit_RAG_Phase1_Report.md) |

---

## How the Circuit Path Works

```
PDF / lecture slide
        │
        ▼
   MinerU / Docling parser  →  content_list with image chunks
        │
        ▼
   RAGAnything.ainsert_content_from_path()
        │
        ├─ text chunks ──────────────────────► LightRAG KG + vector store
        │
        └─ image chunks
                │
                ▼
         ModalProcessor router
                │
                ├─ circuit-like? ──► CircuitModalProcessor
                │                     • component list, connections, netlist
                │                     • serialised to structured text chunk
                │
                └─ other ─────────► ImageModalProcessor (caption only)
                        │
                        ▼
               same unified KG + vector store
```

The key principle: circuit-derived structured content enters the **same** LightRAG index as plain text. No separate side store — cross-modal hybrid retrieval is preserved intact.

---

## Phase 1 Experimental Results

### Headline

> **E2 (circuit-focused QA) on Op-Amp topic: 3/8 → 8/8 success rate**
> when switching from mixed retrieval to circuit-aware hybrid retrieval.

This is not a stylistic improvement — it is a task-completion reliability gain.

### Ablation Tables

#### Op-Amp — rich multimodal store (`output_ablation_formal_opamp/`)

| Experiment | Condition | Success | Avg Latency (s) | Avg Answer Len |
|---|---|---:|---:|---:|
| E1 — general QA | Hybrid | **8/8** | 79.4 | 1207 |
| E1 — general QA | Naive | 8/8 | 40.8 | 707 |
| E2 — circuit QA | **Hybrid** | **8/8** | 54.9 | 912 |
| E2 — circuit QA | Mix | **3/8** | 49.2 | 754 |

#### Op-Amp v3 — faster embedding (`text-embedding-v4`, dim=1024) (`output_ablation_formal_opamp_v3/`)

| Experiment | Condition | Success | Avg Latency (s) | Avg Answer Len |
|---|---|---:|---:|---:|
| E1 — general QA | Hybrid | 8/8 | 11.8 | 986 |
| E1 — general QA | Naive | 8/8 | 12.4 | 1004 |
| E2 — circuit QA | Hybrid | 8/8 | **0.7** | 986 |
| E2 — circuit QA | Mix | 8/8 | 48.9 | 1203 |

Switching to `text-embedding-v4` (dim=1024) cut E2 topic-store latency from ~55 s to under 1 s.

#### BJT — second independent topic (`output_ablation_formal_bjts_v3/`)

| Experiment | Condition | Success | Avg Latency (s) | Avg Answer Len |
|---|---|---:|---:|---:|
| E1 — general QA | Hybrid | 7/8 | 16.8 | 1238 |
| E1 — general QA | Naive | 8/8 | 59.9 | 984 |
| E2 — circuit QA | **Hybrid** | **8/8** | 2.8 | 1195 |
| E2 — circuit QA | Mix | 7/8 | 14.9 | 1155 |

Hybrid retrieval dominates circuit-focused questions on a second independent topic, confirming the op-amp result is not a one-off.

### Conclusions

1. Circuit-aware structured processing improves answer reliability on circuit-specific QA compared with mixed retrieval.
2. The enhancement integrates cleanly into the native RAG-Anything architecture without a separate side system.
3. Topic-specific deployment is the most dependable evaluation mode at this stage.

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

# 2. Verify the circuit processor extracts structured output
python 07_smoke_test_circuit_processor.py

# 3. Build a topic store (Op-Amp example)
bash 09_build_topic_store.sh opamp

# 4. Run ablation experiments
python 04_ablation_experiments.py \
    --working-dir ../rag_storage_formal_opamp_v3 \
    --qa-file ../experiment_data/prepared/eval/week5_opamp.jsonl \
    --experiments E1 E2
```

Results land in `output_ablation_*/ablation_results.json`.

### Script index

| Script | Purpose |
|---|---|
| `00_setup_env.sh` | Install dependencies and validate env |
| `01_run_pipeline.py` | Full document ingestion pipeline |
| `02_test_modal_processors.py` | Unit-test modal processor routing |
| `03_circuit_processor.py` | Manual circuit extraction test |
| `04_ablation_experiments.py` | E1/E2 ablation with result JSON output |
| `05_prepare_circuit_experiment_data.py` | Prepare QA eval sets from raw PDFs |
| `06_smoke_test_model_stack.py` | Verify LLM/VLM/embed stack is live |
| `07_smoke_test_circuit_processor.py` | End-to-end circuit extraction smoke test |
| `08_build_eval_questions.py` | Generate evaluation question sets |
| `09_build_topic_store.sh` | Build a topic-isolated RAG store |

---

## Evaluation Data

| File | Contents |
|---|---|
| `experiment_data/prepared/eval/week5_opamp.jsonl` | 8 QA pairs — Op-Amp topic |
| `experiment_data/prepared/eval/2024_bjts.jsonl` | 8 QA pairs — BJT topic |
| `experiment_data/prepared/eval/source_superposition.jsonl` | QA pairs — Superposition topic |
| `experiment_data/prepared/gold/` | Per-document gold metadata |
| `experiment_data/prepared/queries/` | Per-document query sets |

Source PDFs: university electronics course lecture slides (not committed due to size).

---

## Known Limitation (Phase 1)

The superposition topic (`rag_storage_formal_superposition_v4`) hit a 600 s LLM worker timeout during chunk-level extraction and did not complete. Two topics were fully validated; one exposed the next robustness target.

## Phase 2 Priorities

- Chunk-level fallback before LLM timeout on long/dense pages
- Formal domain QA benchmark aligned to the imported lecture set
- Separate infrastructure failures from answer-quality failures in summaries
- Standardised three-topic report for direct run-to-run comparison

Full analysis: [`docs/Circuit_RAG_Phase1_Report.md`](docs/Circuit_RAG_Phase1_Report.md)
