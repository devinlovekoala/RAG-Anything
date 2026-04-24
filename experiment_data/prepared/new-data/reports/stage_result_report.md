# Electronic Circuit QA Benchmark Stage Report

Date: 2026-04-24

## 1. Objective

This stage evaluates whether the electronic-circuit enhanced version of RAG-Anything provides measurable and reproducible answer-quality improvements over the baseline retrieval setting on university electronics lecture materials.

The evaluation is treated as a frozen stage benchmark rather than an exploratory smoke test. During this run, the document set, question set, model configuration, and scoring protocol are fixed; only the retrieval/answering condition changes between the compared methods.

## 2. Benchmark Protocol

### Dataset

The benchmark contains 80 Chinese QA items across four electronic-circuit topics:

| Topic | Source document | Questions |
|---|---|---:|
| BJT | `experiment_data/2024-ch2-BJTs.pdf` | 20 |
| FET | `experiment_data/2024-ch3-FETs-Enhance.pdf` | 20 |
| Frequency domain | `experiment_data/2024-ch5-frequency.pdf` | 20 |
| Op-Amp | `experiment_data/2024-ch8-op amp.pdf` | 20 |

Each item includes a stable `qid`, topic, category, Chinese question, reference answer, and rule-based scoring rubric. The benchmark covers conceptual understanding, factoid recall, circuit topology, formula-to-circuit mapping, solution procedures, reasoning, cross-modal figure/text interpretation, and unanswerable controls.

### Compared Methods

This report uses the E1 setting as the main answer-quality comparison:

| Method label | Runtime condition | Interpretation |
|---|---|---|
| Baseline | `naive` | Text-oriented baseline retrieval/answering path |
| Enhanced | `hybrid` | Electronic-circuit enhanced multimodal/graph retrieval path |

The experiment also produced E2 results (`hybrid` vs `mix`) for dual-graph analysis. However, the headline answer-quality conclusion in this stage report is based on E1.

### Execution Artifacts

The full benchmark run produced the following artifacts:

| Artifact | Path |
|---|---|
| Benchmark manifest | `experiment_data/prepared/new-data/benchmark_manifest.json` |
| Master benchmark | `experiment_data/prepared/new-data/benchmark_master.jsonl` |
| Per-topic runtime files | `experiment_data/prepared/new-data/runtime/` |
| Baseline answers | `experiment_data/prepared/new-data/results/results_baseline.json` |
| Enhanced answers | `experiment_data/prepared/new-data/results/results_enhanced.json` |
| Scoring report | `experiment_data/prepared/new-data/results/benchmark_report.json` |
| Stage report | `experiment_data/prepared/new-data/reports/stage_result_report.md` |

All four topic stores were built successfully with `multimodal_processed=true`. For all four topics, E1 and E2 reached `20/20` query success rate, and no query-level `error` was found in the generated `ablation_results.json` files.

## 3. Results

### Overall Result

| Metric | Baseline | Enhanced | Delta |
|---|---:|---:|---:|
| Overall rule-based score | 0.413 | 0.443 | +0.030 |

Under the current rule-based scorer, the enhanced system achieves a modest but positive overall improvement.

### Topic-Level Results

| Topic | Baseline | Enhanced | Delta |
|---|---:|---:|---:|
| BJT | 0.347 | 0.381 | +0.034 |
| FET | 0.404 | 0.407 | +0.003 |
| Frequency domain | 0.417 | 0.492 | +0.075 |
| Op-Amp | 0.482 | 0.493 | +0.011 |

The largest topic-level gain appears in frequency-domain circuit analysis, followed by BJT. FET and Op-Amp show smaller but still positive deltas.

### Category-Level Results

| Category | Baseline | Enhanced | Delta |
|---|---:|---:|---:|
| Concept | 0.409 | 0.452 | +0.043 |
| Cross-modal | 0.378 | 0.409 | +0.031 |
| Factoid | 0.398 | 0.398 | +0.000 |
| Mapping | 0.304 | 0.304 | +0.000 |
| Procedure | 0.177 | 0.203 | +0.026 |
| Reasoning | 0.448 | 0.493 | +0.045 |
| Topology | 0.408 | 0.479 | +0.071 |
| Unanswerable | 1.000 | 1.000 | +0.000 |

The main gains are concentrated in topology, reasoning, concept, and cross-modal categories. This aligns with the intended value of the electronic-circuit enhancement: it should help the system better use circuit structure, figure/text correspondence, and analytical context, rather than merely improving direct keyword-style factual retrieval.

### Per-Question Outcome

| Outcome | Count |
|---|---:|
| Enhanced wins | 14 / 80 |
| Ties | 62 / 80 |
| Enhanced losses | 4 / 80 |

The per-question profile suggests that the enhanced method improves a targeted subset of structure-heavy and reasoning-heavy questions without causing broad regression on simpler items.

## 4. Interpretation

This stage benchmark supports the following conclusions:

1. The electronic-circuit enhanced version of RAG-Anything runs stably on the frozen 80-question benchmark. All four topics completed multimodal processing and QA evaluation without query-level failures.
2. Under the E1 comparison, the enhanced setting achieves a positive overall rule-based score delta of `+0.030`.
3. The strongest improvements appear in categories that are most aligned with circuit understanding: topology, reasoning, cross-modal interpretation, and frequency-domain analysis.
4. Factoid and mapping questions remain largely tied, which suggests that the enhanced method is not merely exploiting simple keyword matches. Its value is more visible in structurally richer and more analytical questions.

## 5. Data Limitations and Expected Scaling

The current benchmark should be interpreted as a stage-level validation rather than a final large-scale domain benchmark. The source materials are undergraduate-level university lecture slides. This has several implications:

1. The data scale is limited: the benchmark currently contains 80 QA items across four lecture documents.
2. The source quality is constrained by slide-style teaching materials, which often contain sparse explanations, incomplete derivations, and limited visual annotations.
3. The difficulty level is primarily undergraduate instructional content rather than professional engineering documentation, datasheets, design notes, SPICE reports, or advanced circuit design cases.
4. Many questions are answerable from local textual evidence alone, so the gap between baseline and enhanced retrieval can be naturally compressed.

Because of these limitations, the observed `+0.030` overall improvement should be read conservatively. It demonstrates that the enhanced system is stable and directionally better on this stage benchmark, but it may understate the potential benefit of the electronic-circuit enhancement. On larger, more specialized, and more visually/structurally demanding circuit datasets, the advantage of circuit-aware multimodal processing and graph retrieval is likely to become more pronounced.

## 6. Remaining Limitations

This stage result is based on an automatic rule-based scorer, not a fully human-adjudicated final benchmark.

Known limitations:

1. The scoring function relies on `must_include` and `must_not_claim` rubric checks, so it is intentionally conservative.
2. Long-form reasoning answers can be under-scored if they use correct but different wording.
3. The benchmark is Chinese-only by design and does not evaluate cross-lingual QA.
4. The raw lecture PDFs are local experiment inputs. The versioned artifacts preserve the benchmark questions, reference answers, scoring protocol, generated outputs, and report.

## 7. Reproduction Commands

Run the full benchmark:

```bash
cd /home/devin/Workspace/HKUDS-project/RAG-Anything
bash experiment_data/prepared/new-data/run_benchmark_experiments.sh
```

Run local answer-quality scoring:

```bash
cd /home/devin/Workspace/HKUDS-project/RAG-Anything
bash experiment_data/prepared/new-data/run_benchmark_quality_eval.sh
```

The second command only exports existing results and runs local rule-based scoring. It does not call the LLM again.

## 8. Stage Conclusion

This stage establishes a reproducible 80-question electronic-circuit QA benchmark and shows that the electronic-circuit enhanced version of RAG-Anything improves automatic answer-quality scores over the baseline retrieval setting. The overall improvement is modest, but the gains appear in the categories where circuit understanding matters most, especially topology analysis, reasoning, cross-modal interpretation, and frequency-domain analysis.

Given the undergraduate lecture-slide nature of the current data, these results should be viewed as a professional stage outcome and a lower-bound signal rather than the final ceiling of the approach. Future evaluation on larger, higher-quality, and more professionally demanding circuit datasets is expected to provide a stronger testbed and may reveal larger gains from the circuit-aware enhancement.
