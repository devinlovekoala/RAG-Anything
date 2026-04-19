# 电路垂直领域 RAG 实验指导手册

> 适用对象：正在基于 `RAG-Anything` 扩展电子电路垂直能力，并希望验证 `CircuitModalProcessor` 对电路类文档 RAG 效果增强的开发者。

---

## 1. 文档目标

这份手册的目标不是只帮助你“跑通代码”，而是帮助你建立一套**可重复、可对比、可解释**的实验闭环，用来回答下面这个核心问题：

**在电路相关文档场景中，加入结构化电路理解能力后，RAG-Anything 是否比原始多模态 caption-only 流程表现更好？**

你最终需要拿到的，不是一句主观判断，而是一组证据：

- 哪些类型的问题提升明显
- 提升来自哪里：元件识别、连接关系、拓扑恢复，还是跨模态对齐
- 哪些问题仍然失败
- 下一轮该优化哪一层：检测、解析、索引、检索还是回答

---

## 2. 当前实现能力边界

截至当前版本，仓库中已经具备以下基础能力：

- 电路图会在图片模态中被启发式识别，并优先路由到 `CircuitModalProcessor`
- 电路图不再只生成 caption，还会尝试生成结构化 `CircuitDesign`
- 元件和连接关系会以结构化实体/关系形式写入知识图谱和向量索引
- 电路 chunk 会附带元件清单、连接关系、SPICE-like 网表摘要

但也要明确当前还未完全覆盖的部分：

- 还没有电路领域专属的 retrieval rerank
- 还没有实验结果自动打分脚本
- 还没有标准化 gold dataset
- 还没有完整的 batch 回归实验流水线文档

所以你接下来最重要的工作是：**准备实验数据、补齐环境、跑通流程、记录结果。**

---

## 3. 实验总流程

建议按下面顺序推进，不要跳步：

1. 安装并验证开发环境
2. 准备电路实验文档数据集
3. 为每份文档准备测试问题与参考答案
4. 跑 baseline：原始 RAG-Anything
5. 跑 enhanced：加入电路结构化处理的版本
6. 对比输出结果并记录指标
7. 复盘失败样本，决定下一轮优化方向

---

## 4. 环境准备

### 4.1 Python 环境建议

建议使用独立虚拟环境，避免和系统 Python 冲突。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
```

### 4.2 安装项目依赖

建议先走仓库提供的轻量安装脚本。默认只安装 RAG-Anything、LightRAG、OpenAI SDK、pytest 等核心依赖，不会立即拉取 `MinerU` 的重型 `torch/CUDA` 栈：

```bash
bash reproduce/00_setup_env.sh
```

如需完整 PDF 解析器，再按需安装：

```bash
# 轻量优先，可先尝试 docling
INSTALL_DOCLING=true bash reproduce/00_setup_env.sh

# 需要 MinerU 时再安装，注意该步骤可能下载 torch/CUDA 等大型依赖
INSTALL_MINERU=true bash reproduce/00_setup_env.sh
```

如果你的目标是一次性完整复现实验，也可以尝试：

```bash
pip install -e '.[all]'
```

如果失败，则退一步安装最小实验依赖，并确保以下能力可用：

- `RAG-Anything`
- `LightRAG`
- 至少一个可用的 LLM / VLM 接口
- 文档解析依赖

### 4.3 先做最小验证

先确认新增能力的定向测试通过：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/python -m pytest -q tests/test_circuit_processing.py
./.venv/bin/python -m py_compile raganything/*.py tests/test_circuit_processing.py
```

如果这里都过不了，不要急着跑大规模实验。

### 4.4 模型栈与电路处理器 Smoke Test

在安装完整 PDF 解析器前，可以先验证 `LLM/VLM/Embedding` 配置和 `CircuitModalProcessor` 是否可用：

```bash
./.venv/bin/python -u reproduce/06_smoke_test_model_stack.py \
  --pdf "experiment_data/Week 5 Lecture 1 - Operation Amplifier.pdf"
```

该脚本会：

- 用 `pdftoppm` 抽取 PDF 第一页图片
- 调用 `LLM_*` 测试文本模型
- 调用 `EMBEDDING_*` 测试向量模型与维度
- 调用 `VLM_*` 测试看图能力

然后可以抽取一页包含电路符号或电路图的页面，并直接验证电路处理器：

```bash
mkdir -p output_smoke/pages
pdftoppm -png -f 5 -l 15 -r 110 \
  "experiment_data/Week 5 Lecture 1 - Operation Amplifier.pdf" \
  output_smoke/pages/opamp

./.venv/bin/python -u reproduce/07_smoke_test_circuit_processor.py \
  --image "output_smoke/pages/opamp-14.png" \
  --caption "Simplified terminals of operational amplifier with non-inverting input, inverting input, output terminal voltage and current labels" \
  --entity-name "Op Amp Simplified Terminals"
```

期望看到：

- `circuit_detected=true`
- `chunk_type=circuit`
- `entity_type=circuit_design`
- 输出 `circuit_components`、`circuit_connections` 和 `circuit_netlist`

如果封面页或普通说明页被判定为 `circuit_detected=false`，这也是正确行为，说明检测器没有把非电路页面误判为电路图。

### 4.5 再做基线功能验证

在跑电路实验前，先确认原始 RAG-Anything 基本链路正常：

```bash
./.venv/bin/python reproduce/01_run_pipeline.py your_test.pdf
```

你需要确认：

- 文档能被成功解析
- text chunk 能写入索引
- multimodal 内容能被处理
- query 能返回结果

---

## 5. 数据集准备

建议你把实验数据按下面结构组织：

```text
experiment_data/
├── docs/
│   ├── A_textbook_circuits/
│   ├── B_lab_docs_mixed/
│   └── C_non_circuit_control/
├── queries/
│   ├── A_textbook_circuits.json
│   ├── B_lab_docs_mixed.json
│   └── C_non_circuit_control.json
├── gold/
│   ├── A_textbook_circuits_gold.json
│   ├── B_lab_docs_mixed_gold.json
│   └── C_non_circuit_control_gold.json
└── results/
    ├── baseline/
    └── enhanced/
```

### 5.1 A 类：纯电路教材页

目标：验证结构恢复对**元件级、拓扑级问答**的增强。

建议包含：

- 运放电路
- RC / RL / RLC 电路
- 二极管整流电路
- 三极管放大电路
- 含清晰标注值的基础教学图

每类最好 3-5 份。

### 5.2 B 类：图文混排实验文档

目标：验证结构恢复是否能提升**跨模态推理**。

建议包含：

- 电路图 + 实验步骤
- 电路图 + 参数表
- 电路图 + 波形图 + 结论说明
- 实验指导书 / 课件 / PDF 任务书

### 5.3 C 类：非电路对照文档

目标：验证新能力不会显著破坏通用能力。

建议包含：

- 普通论文
- 含图表但不含电路图的教学文档
- 含流程图、普通示意图的 PDF

这个对照组很重要，因为它能告诉你：

- circuit detector 是否误判过多
- 新增 processor 是否对非目标文档带来副作用

---

## 6. 查询集设计

每份文档建议准备 5-10 个问题，至少覆盖 4 类。

### 6.1 元件参数类

示例：

- 电路中 `R1` 的阻值是多少？
- 图中的运放型号是什么？
- 电容 `C1` 的容量是多少？

### 6.2 连接关系类

示例：

- `R1` 与哪个元件直接相连？
- 输出节点连接了哪些器件？
- 反馈支路由哪些元件构成？

### 6.3 拓扑理解类

示例：

- 这是一个什么类型的电路？
- 该电路的信号流向是什么？
- 这是反相放大器还是同相放大器？判断依据是什么？

### 6.4 跨模态一致性类

示例：

- 图中的电路与文中给出的增益公式是否一致？
- 图 2 的截止频率和表 1 参数能否对应上？
- 电路图中的测量点是否与实验步骤中的说明一致？

---

## 7. Gold 标注建议

实验如果没有参考答案，结果就只能靠感觉判断。

建议每个问题至少标注下面字段：

```json
{
  "doc_id": "lab_rc_01",
  "question_id": "q3",
  "question": "该电路的反馈路径是什么？",
  "answer_type": "topology",
  "gold_answer": "输出端通过反馈电阻 R2 回到运放反相输入端，构成负反馈路径。",
  "key_facts": ["R2", "输出端", "反相输入端", "负反馈"],
  "scoring_rule": "如果答案识别出 R2 且指出输出到反相输入的反馈关系，则判定为正确。"
}
```

建议把 `answer_type` 统一成以下几类：

- `component_value`
- `component_identity`
- `connection`
- `topology`
- `cross_modal_consistency`
- `comparison`

---

## 8. 实验分组设计

至少做两组：

### 8.1 Baseline 组

使用原始图像处理逻辑，电路图只作为普通 image caption 处理。

目标：作为对照。

### 8.2 Enhanced 组

启用 `CircuitModalProcessor`，让电路图走结构化理解流程。

目标：验证结构恢复带来的收益。

### 8.3 可选第三组

如果后续你继续优化，可增加：

- `Enhanced + circuit-aware retrieval rerank`
- `Enhanced + better circuit prompt`
- `Enhanced + drawsee transfer netlist`

这样你能把收益拆解到更细层级。

---

## 9. 重点观察指标

建议至少记录以下指标。

### 9.1 文档处理指标

- 文档总数
- 成功解析数
- 检测为 circuit 的图片数
- 结构恢复成功数
- 结构恢复失败数
- 平均每图抽取元件数
- 平均每图抽取连接数

### 9.2 问答效果指标

- `Top-1 命中率`
- `答案正确率`
- `关键事实召回率`
- `拓扑类问题正确率`
- `跨模态一致性问题正确率`

### 9.3 副作用指标

- 非电路文档误判率
- 非电路文档回答质量下降情况
- 检索噪声是否上升

---

## 10. 结果记录模板

你可以先用 Markdown 表格，后续再转成脚本产物。

### 10.1 文档级记录

| Doc ID | 类型 | 页数 | 电路图页数 | 检测命中 | 结构恢复成功 | 备注 |
|--------|------|------|------------|----------|--------------|------|
| lab_rc_01 | B | 12 | 3 | 3/3 | 2/3 | 第 7 页连接漏掉 |

### 10.2 问题级记录

| Doc ID | QID | 问题类型 | Baseline | Enhanced | 是否提升 | 备注 |
|--------|-----|----------|----------|----------|----------|------|
| lab_rc_01 | q1 | component_value | 错 | 对 | 是 | Enhanced 识别到 R1=10k |
| lab_rc_01 | q3 | topology | 模糊 | 对 | 是 | Enhanced 能指出负反馈路径 |

### 10.3 汇总指标

| 指标 | Baseline | Enhanced | 差值 |
|------|----------|----------|------|
| 元件参数问题正确率 | 42% | 71% | +29% |
| 连接关系问题正确率 | 35% | 68% | +33% |
| 拓扑问题正确率 | 38% | 73% | +35% |
| 跨模态一致性问题正确率 | 41% | 62% | +21% |

---

## 11. 推荐执行顺序

建议按这个顺序跑，成本最低，信息密度最高。

### 第一轮：最小样本验证

目标：确认链路可跑通。

- 2 份 A 类文档
- 2 份 B 类文档
- 1 份 C 类文档
- 每份 3 个问题

如果第一轮都不稳定，就不要急着扩数据量。

### 第二轮：小规模对比实验

目标：确认增强是否真实有效。

- 每类 3-5 份文档
- 每份 5-10 个问题
- 跑 baseline / enhanced 双组对比

### 第三轮：失败样本驱动优化

目标：找出结构恢复与检索的真实瓶颈。

重点看：

- 是否检测漏判
- 是否元件识别错误
- 是否连接关系错误
- 是否检索命中了错误 chunk
- 是否答案生成阶段没有用到结构化结果

---

## 12. 当前推荐入口

基于当前仓库实现与已验证环境，建议你优先采用下面这条执行路径：

1. 使用 `docling` 作为默认 PDF 解析器
2. 使用仓库内置的 `CircuitModalProcessor`
3. 先跑单页/单图验证，再跑整份 PDF，再做多文档对比实验

推荐命令：

```bash
# 1) 单页电路处理验证
./.venv/bin/python -u reproduce/03_circuit_processor.py \
  --image "output_smoke/pages/opamp-14.png" \
  --caption "Simplified terminals of operational amplifier with non-inverting input, inverting input, output terminal voltage and current labels" \
  --entity-name "Op Amp Simplified Terminals"

# 2) 单文档端到端索引
./.venv/bin/python reproduce/01_run_pipeline.py \
  "experiment_data/Week 5 Lecture 1 - Operation Amplifier.pdf" \
  --parser docling \
  --working-dir ./rag_storage_circuit_docling \
  --output ./output_circuit_docling

# 3) 小规模对比实验
./.venv/bin/python reproduce/04_ablation_experiments.py \
  --working-dir ./rag_storage_circuit_docling \
  --output ./output_ablation_docling \
  --experiments E1 E2
```

如果后续你真的需要 `MinerU`，建议把它视为“可选增强解析器”而不是当前默认入口。当前阶段先把 `docling` 路线跑稳，更有利于尽快完成二阶段实验与效果验证。

---

## 12. 当前版本最值得你关注的观察点

结合当前代码实现，建议重点观察下面几件事。

### 12.1 Circuit detector 的误判和漏判

这是整个电路增强入口的第一关。

如果漏判多：

- 电路图仍会退回普通 image caption 流程
- 后续结构化能力完全发挥不出来

如果误判多：

- 非电路图片会被不必要地拉进 circuit 分支
- 可能污染索引

### 12.2 结构化恢复质量

重点看这些字段是否稳定：

- `circuit_type`
- `components`
- `connections`
- `circuit_netlist`

如果这些字段质量不高，说明需要优先优化 prompt 或 parser。

### 12.3 检索是否真的利用了电路结构

不要只看“答案看起来更专业”，要看：

- 检索到的 chunk 是否真包含目标元件/连接
- KG 中是否出现了相关 component entity
- 相比 baseline，是否更早命中正确证据

---

## 13. 常见失败模式

### 13.1 结构恢复失败，但 caption 看起来正常

现象：

- 图像描述很像对了
- 但没有抽到 `components` / `connections`

说明：

- VLM 输出了自然语言，但没有遵守结构化 JSON 格式
- 或 parser 对返回格式兼容不够

### 13.2 能抽到元件，抽不到连接

现象：

- `R1`、`C1`、`U1` 都有
- 但问“反馈路径是什么”还是答不好

说明：

- 结构恢复只到了“识别器件”，没有到“恢复拓扑”
- 这通常是电路问答效果提升受限的根因

### 13.3 Enhanced 索引变大，但检索没提升

现象：

- chunk 更多了
- KG 节点更多了
- 但答案没明显变好

说明：

- 结构化信息进入索引了，但 retrieval/fusion 没有充分利用它

### 13.4 非电路文档表现变差

现象：

- 通用文档结果下降

说明：

- detector 误判
- circuit chunk 噪声过大
- multimodal 路由策略需要更保守

---

## 14. 本轮实验完成标准

建议你把“完成”定义得具体一些。至少满足：

- 环境可稳定运行
- 至少一轮 baseline / enhanced 对比已跑完
- 至少 10 份文档完成测试
- 至少 50 个问题有记录结果
- 能明确指出 3 个最有效提升场景
- 能明确指出 3 个主要失败模式

当你达到这个标准时，下一轮优化才有方向。

---

## 15. 下一步建议

在你完成第一轮实验后，优先做下面三类优化中的一类，不建议同时铺开。

### 路线 A：先补检测稳定性

如果漏判/误判很多，优先优化 circuit detector。

### 路线 B：先补结构恢复质量

如果“识别出是电路图，但 components / connections 质量差”，优先改 prompt 和 parser。

### 路线 C：先补 retrieval 利用率

如果结构化结果已经进索引，但回答质量提升有限，优先做 circuit-aware retrieval / rerank。

---

## 16. 建议你现在立刻做的事情

今天最推荐的顺序：

1. 补齐依赖环境，确保基础测试能通过
2. 从你的昭析项目资料里挑 5-10 份代表性 PDF
3. 按本手册建立 `experiment_data/` 目录
4. 为每份文档写 5 个问题和参考答案
5. 先跑一轮最小样本对比
6. 根据失败样本决定下一轮优化方向

---

## 17. 与现有计划文档的关系

本手册是执行层文档。

- [CircuitModalProcessor_Engineering_Plan.md](/home/devin/Workspace/HKUDS-project/RAG-Anything/docs/CircuitModalProcessor_Engineering_Plan.md) 更偏“研发蓝图与阶段规划”
- 本文更偏“你现在该怎么准备数据、怎么跑实验、怎么记录结果”

建议两个文档配合使用：

- 看大方向：工程计划文档
- 真正落地执行：本实验指导手册
