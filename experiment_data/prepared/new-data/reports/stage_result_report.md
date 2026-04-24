# 电子电路 QA Benchmark 阶段成果报告

日期：2026-04-24

## 1. 目标

本阶段评测的目标是验证：在大学电子电路讲义问答场景下，加入电子电路支持后的 RAG-Anything 相比基线检索设置，是否能够带来可观测、可复现的答案质量提升。

本轮评测按照“阶段冻结版 benchmark”执行，而不是临时 smoke test。评测过程中固定文档集合、问题集合、模型配置和评分规则，只比较不同检索/回答条件下的输出差异。

## 2. Benchmark 协议

### 数据集

本轮 benchmark 共包含 80 道中文 QA，覆盖 4 个电子电路 topic：

| Topic | Source document | Questions |
|---|---|---:|
| BJT | `experiment_data/2024-ch2-BJTs.pdf` | 20 |
| FET | `experiment_data/2024-ch3-FETs-Enhance.pdf` | 20 |
| Frequency domain | `experiment_data/2024-ch5-frequency.pdf` | 20 |
| Op-Amp | `experiment_data/2024-ch8-op amp.pdf` | 20 |

每道题均包含稳定的 `qid`、topic、category、中文问题、参考答案和规则评分 rubric。题型覆盖概念理解、事实问答、电路拓扑、公式与电路映射、解题步骤、推理、跨模态图文理解，以及不可回答问题。

### 对比方法

本报告使用 E1 设置作为主结论口径：

| Method label | Runtime condition | Interpretation |
|---|---|---|
| Baseline | `naive` | 偏文本的基线检索/回答路径 |
| Enhanced | `hybrid` | 启用电子电路增强后的多模态/图检索路径 |

本轮实验也生成了 E2 结果，即 `hybrid` vs `mix`，用于分析双图策略相关表现；但本阶段答案质量主结论以 E1 为准。

### 产物文件

本轮完整运行生成了以下关键产物：

| Artifact | Path |
|---|---|
| Benchmark manifest | `experiment_data/prepared/new-data/benchmark_manifest.json` |
| Master benchmark | `experiment_data/prepared/new-data/benchmark_master.jsonl` |
| Per-topic runtime files | `experiment_data/prepared/new-data/runtime/` |
| Baseline answers | `experiment_data/prepared/new-data/results/results_baseline.json` |
| Enhanced answers | `experiment_data/prepared/new-data/results/results_enhanced.json` |
| Scoring report | `experiment_data/prepared/new-data/results/benchmark_report.json` |
| Stage report | `experiment_data/prepared/new-data/reports/stage_result_report.md` |

四个 topic 的 topic store 均成功完成，且 `multimodal_processed=true`。四个 topic 的 E1/E2 ablation 均达到 `20/20` query success rate，未发现 query 级别 `error`。

## 3. 结果

### 总体结果

| Metric | Baseline | Enhanced | Delta |
|---|---:|---:|---:|
| Overall rule-based score | 0.413 | 0.443 | +0.030 |

在当前规则评分器下，电子电路增强版取得了小幅但稳定的总体提升。

### Topic 维度结果

| Topic | Baseline | Enhanced | Delta |
|---|---:|---:|---:|
| BJT | 0.347 | 0.381 | +0.034 |
| FET | 0.404 | 0.407 | +0.003 |
| Frequency domain | 0.417 | 0.492 | +0.075 |
| Op-Amp | 0.482 | 0.493 | +0.011 |

提升最明显的是频域分析，其次是 BJT。FET 与 Op-Amp 上提升较小，但仍保持正向。

### 题型维度结果

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

增强版的主要收益集中在 topology、reasoning、concept 和 cross-modal 题型上。这与电子电路增强模块的目标一致：它更应该帮助模型理解电路结构、图文对应关系和分析过程，而不是单纯提升直接事实检索。

### 逐题胜负

| Outcome | Count |
|---|---:|
| Enhanced wins | 14 / 80 |
| Ties | 62 / 80 |
| Enhanced losses | 4 / 80 |

逐题结果显示，增强版没有造成大面积退化，主要是在一部分结构更强、推理更重的问题上取得提升。

## 4. 结论解读

本轮阶段 benchmark 支持以下结论：

1. 电子电路增强版 RAG-Anything 在 80 题冻结 benchmark 上具备稳定运行能力。四个 topic 均完成多模态处理和问答评测，未出现 query 级失败。
2. 在 E1 口径下，增强版相对基线取得 `+0.030` 的总体规则评分提升。
3. 提升主要集中在更符合电路理解价值的题型上，包括 topology、reasoning、cross-modal 和 frequency-domain analysis。
4. factoid 与 mapping 题型基本持平，说明增强版并不是依靠简单关键词题拉开差距；其优势更多体现在结构理解与综合分析问题上。

## 5. 局限性

本阶段结果应表述为“基于规则评分器的初步答案质量对比”，而不是完全人工裁决的最终质量评测。

当前局限包括：

1. 评分器基于 `must_include` 与 `must_not_claim` 做规则评分，较保守。
2. 长答案如果使用了等价但不同的表述，可能被低估。
3. 本轮 benchmark 固定为中文问答，不覆盖跨语言问答能力。
4. 原始讲义 PDF 作为本地实验输入存在；版本化产物主要保留问题集、参考答案、评分协议和生成结果。

## 6. 复现实验命令

运行完整 benchmark：

```bash
cd /home/devin/Workspace/HKUDS-project/RAG-Anything
bash experiment_data/prepared/new-data/run_benchmark_experiments.sh
```

运行本地答案质量评分：

```bash
cd /home/devin/Workspace/HKUDS-project/RAG-Anything
bash experiment_data/prepared/new-data/run_benchmark_quality_eval.sh
```

第二条命令只做本地结果导出和规则评分，不会再次调用 LLM。

## 7. 阶段成果结论

本阶段建立了一套可复现的 80 题电子电路 QA benchmark，并在该 benchmark 上验证了电子电路增强版 RAG-Anything 相比基线设置的答案质量提升。总体提升幅度不夸张，但集中出现在最能体现电路理解能力的题型上，尤其是拓扑分析、推理和频域分析。

因此，本轮结果可以作为阶段性专业成果输出。后续若进入更高规格的最终评测，可在该 benchmark 基础上增加人工复核或 LLM-as-judge，对 reasoning 与 cross-modal 题型进行抽样裁决，从而进一步增强结论的说服力。
