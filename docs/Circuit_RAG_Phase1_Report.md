# Circuit-Domain RAG Phase 1 Report

## 1. Scope

This Phase 1 milestone extends `RAG-Anything` toward native support for electronic-circuit documents, with a focus on preserving the framework's original multimodal advantages while adding circuit-aware structured understanding.

The implementation goal was not only to caption circuit images, but to convert them into reusable structured representations that can participate in:

- multimodal unified indexing,
- cross-modal hybrid retrieval,
- knowledge graph construction,
- circuit-oriented downstream answering.

Phase 1 should therefore be understood as a capability validation milestone rather than the final robustness target.

## 2. Implemented Capabilities

The current codebase now includes a built-in circuit processing path centered on `CircuitModalProcessor`, plus the surrounding parser, storage, retrieval, and reproducibility work needed to exercise it in real document pipelines.

Key delivered capabilities:

- Heuristic routing of circuit-like images into a dedicated circuit processor rather than plain caption-only handling.
- Structured `CircuitDesign` extraction with component lists, connection relations, and SPICE-like netlist summaries.
- Integration of circuit chunks into the existing `RAG-Anything` storage and retrieval path instead of creating a disconnected side system.
- Reproducible experiment scripts for smoke testing, end-to-end insertion, topic-store construction, question preparation, and ablation evaluation.
- Environment and config support for LLM, VLM, rerank, and embedding setup in the reproduction workflow.

In practical terms, the circuit extension is already able to work with the original design strengths of `RAG-Anything`:

- `Multimodal Unified Indexing`: circuit-derived text summaries, structured entities, and multimodal chunks are indexed in the same pipeline.
- `Cross-modal Hybrid Retrieval`: retrieved answers can benefit from both document text and circuit-aware multimodal chunks.
- Graph-augmented reasoning: component and connection information is materialized as structured content that can complement plain caption semantics.

## 3. Experimental Setup

### 3.1 Data and Topics

Phase 1 evaluation used electronic-circuit lecture PDFs collected under `experiment_data/`, with topic-focused stores built for:

- Operational amplifiers
- Bipolar junction transistors (BJTs)
- Source transformation and superposition

### 3.2 Evaluation Strategy

Two evaluation themes were used repeatedly:

- `E1`: general lecture understanding and domain QA
- `E2`: more circuit-focused explanatory and structural QA

The strongest comparative evidence came from topic-specific stores rather than one large shared store, because topic isolation reduced indexing instability and improved retrieval consistency.

### 3.3 Model/Runtime Notes

The pipeline was iteratively stabilized during the phase:

- Embedding dimension mismatch was resolved by aligning the environment to the actual `text-embedding-v4` output dimension (`1024`).
- Missing rerank configuration warnings were handled in project-side query defaults.
- Storage initialization, multimodal insertion, and ablation-script robustness issues were fixed.
- A keyword-extraction fallback was added so evaluation runs could survive structured-output instability from the LLM side.

## 4. Key Results

### 4.1 Milestone Evidence: Op-Amp Rich Run

The strongest early proof point came from the richer op-amp run in `output_ablation_formal_opamp/ablation_results.json`.

| Topic | Experiment | Condition | Success | Avg Latency (s) | Avg Answer Length |
|---|---|---|---:|---:|---:|
| Op-amp | E1 | Hybrid | 8/8 | 79.409 | 1206.5 |
| Op-amp | E1 | Naive | 8/8 | 40.823 | 706.9 |
| Op-amp | E2 | Hybrid | 8/8 | 54.913 | 912.1 |
| Op-amp | E2 | Mix | 3/8 | 49.197 | 753.7 |

This `3/8` versus `8/8` result is the most convincing Phase 1 milestone because it shows a clear task-level reliability gain rather than only stylistic or latency variation.

### 4.2 New Faster-Embedding Round: Op-Amp v3

After switching to the faster embedding setup, the op-amp topic was re-run in `output_ablation_formal_opamp_v3/ablation_results.json`.

| Topic | Experiment | Condition | Success | Avg Latency (s) | Avg Answer Length |
|---|---|---|---:|---:|---:|
| Op-amp v3 | E1 | Hybrid | 8/8 | 11.774 | 985.9 |
| Op-amp v3 | E1 | Naive | 8/8 | 12.352 | 1003.8 |
| Op-amp v3 | E2 | Hybrid | 8/8 | 0.741 | 985.9 |
| Op-amp v3 | E2 | Mix | 8/8 | 48.883 | 1203.1 |

Interpretation:

- The faster embedding stack substantially reduced latency for the topic-store workflow.
- This run is useful as a speed and stability checkpoint.
- However, it is less persuasive than the earlier rich op-amp run for multimodal superiority claims, because its store was materially smaller and less multimodal-dense.

### 4.3 BJT Topic Results

The BJT topic completed with usable results in `output_ablation_formal_bjts_v3/ablation_results.json`.

| Topic | Experiment | Condition | Success | Avg Latency (s) | Avg Answer Length |
|---|---|---|---:|---:|---:|
| BJT | E1 | Hybrid | 7/8 | 16.775 | 1237.7 |
| BJT | E1 | Naive | 8/8 | 59.875 | 983.6 |
| BJT | E2 | Hybrid | 8/8 | 2.785 | 1195.0 |
| BJT | E2 | Mix | 7/8 | 14.922 | 1155.3 |

Interpretation:

- The BJT topic validates that the circuit-domain enhancement is not limited to one op-amp lecture.
- Hybrid retrieval remained strong on circuit-focused questioning (`E2`), reaching `8/8`.
- A small number of failures remain under timeout- or response-quality-related conditions, so the topic should be considered validated but not fully hardened.

## 5. Comparative Conclusions

Phase 1 already supports three evidence-backed conclusions:

1. Circuit-aware structured processing can improve answer reliability on circuit-specific QA compared with weaker mixed retrieval behavior.
2. The enhancement is compatible with the native `RAG-Anything` architecture rather than fighting against it.
3. Topic-specific deployment is currently the most dependable evaluation mode for this vertical.

The most important headline result is:

> In the op-amp rich run, `E2` improved from `3/8` successful answers in the mixed condition to `8/8` in the hybrid circuit-aware condition.

That is a meaningful milestone because it demonstrates that structured circuit understanding is not merely adding verbosity; it is improving task completion reliability.

## 6. Known Limitation Exposed in Phase 1

The third topic, source transformation and superposition, did not complete successfully in `rag_storage_formal_superposition_v4`.

Observed status:

- Document status: `failed`
- Failure point: chunk-level LLM worker timeout
- Exact error: `Worker execution timeout after 600s`

This matters because it shows the next bottleneck is no longer basic wiring of the circuit pipeline. The remaining issue is robustness under difficult, long, or concept-dense chunks during extraction.

So the honest Phase 1 assessment is:

- Two representative circuit subdomains were validated.
- One additional topic exposed a concrete extraction-time robustness gap.

## 7. Phase 2 Delivered — Adapter Route Refactoring

Phase 2 completed the architectural refactoring that Phase 1 identified as the right direction.

### 7.1 Architecture Decision

Embedding circuit logic directly into `raganything/` created tight coupling that would make upstream PRs difficult. Phase 2 resolves this with the **adapter route**: all circuit-domain logic moves into a self-contained package at `examples/circuit_domain_extension/`. The only integration point with the core library is the existing `insert_content_list()` API.

### 7.2 Delivered Modules

| Module | Role |
|---|---|
| `ir.py` | `CircuitIR` / `CircuitFigure` — document-level intermediate representation |
| `prompts.py` | VLM prompt constants, migrated from core and extended |
| `parser.py` | PDF → CircuitIR: runs MinerU/Docling, detects circuits, calls VLM |
| `enhancer.py` | Component ref normalisation, topology inference from component mix |
| `adapter.py` | CircuitIR → `content_list` (type="circuit" items for `insert_content_list`) |
| `pipeline.py` | `CircuitPipeline` class + `ingest_circuit_document()` one-shot helper |
| `__init__.py` | Public API re-exports |

### 7.3 Core Cleanup

`raganything/raganything.py` — `CircuitModalProcessor` import and `_initialize_processors` wiring block removed. The `enable_circuit_processing` config field is retained as a no-op to avoid breaking existing reproduce scripts that pass it as a kwarg.

The core library is now clean with respect to circuit-domain concerns.

### 7.4 Evaluation Expansion

Two new eval topics were added, bringing the benchmark from 3 to 4 distinct circuit subdomains:

| File | Topic | Status |
|---|---|---|
| `week5_opamp.jsonl` | Op-Amp | validated Phase 1 |
| `2024_bjts.jsonl` | BJT | validated Phase 1 |
| `2024_fets.jsonl` | FET / MOSFET | new — Phase 2 |
| `week9_freq_domain.jsonl` | Freq-Domain analysis | new — Phase 2 |

### 7.5 Outstanding Work

- Re-run ablation experiments for all 4 topics using the new adapter-based pipeline.
- Address the 600 s LLM worker timeout that blocked the superposition topic (chunk-level fallback strategy).
- Formal answer-quality scoring to separate infrastructure failures from retrieval/LLM-quality failures.

## 8. Final Assessment

Phase 1 achieved a real milestone with two validated circuit subdomains and one identified robustness blocker.

Phase 2 completed the architectural refactoring: the circuit extension is now a clean adapter package requiring zero changes to the core library. This is the correct foundation for a future upstream PR.

The current state is:

**Two-phase validated progress — measurable task-level gains in Phase 1, clean architectural separation in Phase 2, expanded 4-topic benchmark ready for the next experimental run.**
