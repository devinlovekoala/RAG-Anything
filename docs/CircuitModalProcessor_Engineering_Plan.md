# CircuitModalProcessor 开发与对比实验工程计划

> 目标：在 RAG-Anything 的 "1+3+N" 框架中开发面向电路图的垂直领域模态处理器，并通过严谨的对比实验证明结构恢复策略相对于纯 caption 策略的价值。

---

## 全局架构概览

```
                         RAG-Anything Framework
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
   ImageModalProcessor   TableModalProcessor   EquationModalProcessor
          │                                           │
          │          ┌────────────────────┐            │
          │          │  CircuitModal      │            │
          │          │  Processor (NEW)   │            │
          │          │                    │            │
          │          │  VLM caption       │            │
          │          │       +            │            │
          │          │  Netlist parsing   │            │
          │          │       +            │            │
          │          │  Component→Entity  │            │
          │          │  Connection→Edge   │            │
          │          └────────────────────┘            │
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   Dual-Graph KG       │
                    │                       │
                    │  Cross-Modal KG:      │
                    │   R1 ──connects── C1  │
                    │   OpAmp ──feeds── R2  │
                    │   Figure ──shows── *  │
                    │                       │
                    │  Text KG:             │
                    │   (LightRAG native)   │
                    └───────────────────────┘
```

---

## 阶段一：前置准备（Day 1-2）

### 1.1 环境与代码准备

```bash
# 1. Clone 并安装 RAG-Anything
git clone https://github.com/HKUDS/RAG-Anything.git
cd RAG-Anything
pip install -e '.[all]'

# 2. 验证基线功能
python examples/raganything_example.py test_doc.pdf --api-key YOUR_KEY

# 3. 创建你的开发分支
git checkout -b feature/circuit-modal-processor
```

### 1.2 阅读源码（精确到行）

必须先通读以下文件，理解接口契约：

| 文件 | 重点理解 | 预计时间 |
|------|----------|----------|
| `raganything/modalprocessors/__init__.py` | 处理器注册机制 | 15min |
| `raganything/modalprocessors/image_processor.py` | `process_multimodal_content()` 的输入输出格式、`entity_info` 的结构 | 1h |
| `raganything/modalprocessors/context_extractor.py` | 上下文窗口如何工作 | 30min |
| `raganything/raganything.py` 中的 `_process_multimodal_content_batch_type_aware()` | 批处理如何按类型分发到处理器 | 1h |
| LightRAG 中的 `lightrag/kg/` 目录 | 实体和关系如何写入知识图谱 | 1h |

### 1.3 测试数据集准备

准备三类 PDF 文档（每类 3-5 份）：

**A 类 — 纯电路教材页**
- 含运放电路原理图、标注清晰的元件值
- 来源建议：模电/数电教材扫描件，或 ZhaoXi 项目已有课件

**B 类 — 图文混排实验文档**
- 含电路图 + 文字说明 + 表格参数
- 来源建议：你的实验任务书 PDF

**C 类 — 通用学术论文（对照组）**
- 不含电路图，但含常规图表
- 用于验证 CircuitModalProcessor 不会对非电路内容产生负面影响

为每份文档准备 **5-10 个测试查询**，涵盖：
- 元件值查询："电路中 R1 的阻值是多少？"
- 拓扑理解查询："该放大电路的反馈路径是什么？"
- 跨模态推理查询："图 3 中的电路与文中描述的增益公式是否一致？"
- 对比查询："两个滤波器电路的截止频率有何区别？"

---

## 阶段二：CircuitModalProcessor 核心开发（Day 3-7）

### 2.1 文件结构

```
raganything/modalprocessors/
├── __init__.py                    # 修改：注册新处理器
├── image_processor.py             # 参考：继承基类
├── table_processor.py
├── equation_processor.py
├── context_extractor.py
└── circuit_processor.py           # 新增：核心文件
```

### 2.2 核心类设计

```python
# raganything/modalprocessors/circuit_processor.py

"""
CircuitModalProcessor: 面向电路图的垂直领域模态处理器

与 ImageModalProcessor 的关键区别：
1. 不止生成 caption，还执行 netlist 结构恢复
2. 将电路元件（R/C/L/OpAmp/...）作为独立实体注入 KG
3. 将元件连接关系作为 KG 边，而非嵌入在描述文本中
4. 支持 SPICE netlist 格式的结构化输出
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from raganything.modalprocessors.image_processor import ImageModalProcessor

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class CircuitComponent:
    """单个电路元件"""
    component_id: str          # e.g., "R1", "C2", "U1"
    component_type: str        # e.g., "resistor", "capacitor", "op_amp"
    value: Optional[str] = None  # e.g., "10kΩ", "100nF"
    pins: List[str] = field(default_factory=list)
    description: Optional[str] = None

    def to_entity_dict(self) -> Dict[str, str]:
        """转换为 KG 实体格式"""
        name = f"{self.component_id}"
        desc_parts = [f"Type: {self.component_type}"]
        if self.value:
            desc_parts.append(f"Value: {self.value}")
        if self.pins:
            desc_parts.append(f"Pins: {', '.join(self.pins)}")
        if self.description:
            desc_parts.append(self.description)
        return {
            "entity_name": name,
            "entity_type": "circuit_component",
            "description": "; ".join(desc_parts),
        }


@dataclass
class CircuitConnection:
    """元件之间的电气连接"""
    source: str         # e.g., "R1"
    target: str         # e.g., "C1"
    net_name: str       # e.g., "net_vout", "GND"
    connection_type: str = "electrical"  # electrical / feedback / signal

    def to_relation_dict(self) -> Dict[str, str]:
        """转换为 KG 关系（边）格式"""
        return {
            "src_id": self.source,
            "tgt_id": self.target,
            "description": f"Connected via {self.net_name}",
            "relation_type": self.connection_type,
        }


@dataclass
class CircuitDesign:
    """完整的电路结构化表示"""
    circuit_name: str
    components: List[CircuitComponent] = field(default_factory=list)
    connections: List[CircuitConnection] = field(default_factory=list)
    netlist_raw: Optional[str] = None
    circuit_type: Optional[str] = None   # e.g., "inverting_amplifier"
    description: Optional[str] = None

    def to_entity_info(self) -> Dict[str, Any]:
        """
        将整个电路转换为 RAG-Anything 的 entity_info 格式。
        包含：
        - 电路本身作为顶层实体
        - 每个元件作为子实体
        - 连接关系作为边
        """
        entities = []
        relations = []

        # 电路本身作为顶层实体
        circuit_entity = {
            "entity_name": self.circuit_name,
            "entity_type": "circuit_design",
            "description": self._build_circuit_summary(),
        }
        entities.append(circuit_entity)

        # 每个元件作为独立实体
        for comp in self.components:
            entities.append(comp.to_entity_dict())
            # 元件→电路的从属关系
            relations.append({
                "src_id": comp.component_id,
                "tgt_id": self.circuit_name,
                "description": f"{comp.component_id} is part of {self.circuit_name}",
                "relation_type": "belongs_to",
            })

        # 元件间连接作为边
        for conn in self.connections:
            relations.append(conn.to_relation_dict())

        return {
            "entities": entities,
            "relations": relations,
            "metadata": {
                "circuit_type": self.circuit_type,
                "component_count": len(self.components),
                "connection_count": len(self.connections),
                "has_netlist": self.netlist_raw is not None,
            },
        }

    def _build_circuit_summary(self) -> str:
        parts = []
        if self.circuit_type:
            parts.append(f"Circuit type: {self.circuit_type}")
        if self.description:
            parts.append(self.description)
        comp_summary = ", ".join(
            f"{c.component_id}({c.component_type}={c.value})"
            for c in self.components if c.value
        )
        if comp_summary:
            parts.append(f"Components: {comp_summary}")
        if self.netlist_raw:
            # 只保留前 500 字符的 netlist 摘要
            parts.append(f"Netlist: {self.netlist_raw[:500]}")
        return "; ".join(parts)


# ============================================================
# Netlist 解析器（从 ZhaoXi 的 CircuitImageNetlistParser 迁移）
# ============================================================

class NetlistParser:
    """
    从 VLM 输出的文本中解析电路结构。
    
    支持两种输入格式：
    1. SPICE-like netlist（R1 net1 net2 10k）
    2. 结构化 JSON 描述（VLM 按 prompt 输出的 JSON）
    """

    # 常见元件类型正则
    COMPONENT_PATTERNS = {
        "resistor":   r'(R\d+)\s+(\w+)\s+(\w+)\s+([\d.]+[kKmMuU]?[Ωω]?)',
        "capacitor":  r'(C\d+)\s+(\w+)\s+(\w+)\s+([\d.]+[pnuUmM]?[fF]?)',
        "inductor":   r'(L\d+)\s+(\w+)\s+(\w+)\s+([\d.]+[munUM]?[hH]?)',
        "diode":      r'(D\d+)\s+(\w+)\s+(\w+)',
        "transistor":  r'(Q\d+|M\d+)\s+(\w+)\s+(\w+)\s+(\w+)',
        "op_amp":     r'(U\d+|OP\d+)\s+(\w+)\s+(\w+)\s+(\w+)',
        "voltage_src": r'(V\d+)\s+(\w+)\s+(\w+)\s+([\d.]+[mMkK]?[vV]?)',
    }

    @classmethod
    def parse_netlist_text(cls, text: str) -> CircuitDesign:
        """
        从 SPICE-like netlist 文本解析电路结构。
        """
        components = []
        connections = []
        nets_to_components: Dict[str, List[str]] = {}

        for comp_type, pattern in cls.COMPONENT_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                comp_id = groups[0]
                pins = list(groups[1:-1]) if len(groups) > 2 else list(groups[1:])
                value = groups[-1] if len(groups) > 2 else None

                # 如果最后一个 group 看起来是 net 名而非 value
                if value and re.match(r'^(net|node|gnd|vcc|vdd|vss)', 
                                       value, re.IGNORECASE):
                    pins.append(value)
                    value = None

                comp = CircuitComponent(
                    component_id=comp_id,
                    component_type=comp_type,
                    value=value,
                    pins=pins,
                )
                components.append(comp)

                # 记录 net → component 映射
                for pin in pins:
                    net_name = pin.lower()
                    if net_name not in nets_to_components:
                        nets_to_components[net_name] = []
                    nets_to_components[net_name].append(comp_id)

        # 从共享 net 推导连接关系
        for net_name, comp_ids in nets_to_components.items():
            if net_name in ("gnd", "0", "vcc", "vdd", "vss"):
                continue  # 电源和地线不建立连接边
            for i in range(len(comp_ids)):
                for j in range(i + 1, len(comp_ids)):
                    connections.append(CircuitConnection(
                        source=comp_ids[i],
                        target=comp_ids[j],
                        net_name=net_name,
                    ))

        return CircuitDesign(
            circuit_name="parsed_circuit",
            components=components,
            connections=connections,
            netlist_raw=text,
        )

    @classmethod
    def parse_json_description(cls, json_text: str) -> Optional[CircuitDesign]:
        """
        从 VLM 输出的 JSON 格式描述解析电路结构。
        期望格式：
        {
          "circuit_name": "Inverting Amplifier",
          "circuit_type": "inverting_amplifier",
          "components": [
            {"id": "R1", "type": "resistor", "value": "10kΩ", "pins": ["Vin", "inv_input"]},
            ...
          ],
          "connections": [
            {"from": "R1", "to": "R2", "net": "inv_input"},
            ...
          ]
        }
        """
        try:
            # 提取 JSON（VLM 输出可能包裹在 markdown code block 中）
            json_match = re.search(r'\{[\s\S]*\}', json_text)
            if not json_match:
                return None
            data = json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            return None

        components = []
        for c in data.get("components", []):
            components.append(CircuitComponent(
                component_id=c.get("id", ""),
                component_type=c.get("type", "unknown"),
                value=c.get("value"),
                pins=c.get("pins", []),
                description=c.get("description"),
            ))

        connections = []
        for conn in data.get("connections", []):
            connections.append(CircuitConnection(
                source=conn.get("from", ""),
                target=conn.get("to", ""),
                net_name=conn.get("net", "unknown"),
                connection_type=conn.get("type", "electrical"),
            ))

        return CircuitDesign(
            circuit_name=data.get("circuit_name", "unknown_circuit"),
            components=components,
            connections=connections,
            circuit_type=data.get("circuit_type"),
            description=data.get("description"),
        )


# ============================================================
# 电路图检测器
# ============================================================

class CircuitDetector:
    """判断一张图片是否为电路图"""

    CIRCUIT_KEYWORDS = [
        "circuit", "schematic", "netlist", "resistor", "capacitor",
        "inductor", "amplifier", "op-amp", "opamp", "transistor",
        "diode", "voltage", "current", "ground", "gnd", "vcc",
        "filter", "oscillator", "电路", "原理图", "运放", "电阻",
        "电容", "三极管", "二极管", "滤波", "放大器",
    ]

    @classmethod
    def is_likely_circuit(cls, caption: str, image_path: str = "") -> bool:
        """
        基于 caption 文本判断是否为电路图。
        后续可扩展为基于图像特征的判断。
        """
        caption_lower = caption.lower()
        keyword_hits = sum(
            1 for kw in cls.CIRCUIT_KEYWORDS if kw in caption_lower
        )
        return keyword_hits >= 2

    @classmethod
    def compute_complexity_score(cls, image_path: str) -> float:
        """
        计算图像复杂度分数（迁移自 ZhaoXi 的 PdfUtils 灰度方差方法）。
        用于 page-level visual pre-retrieval。
        """
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(image_path).convert("L")  # 转灰度
            arr = np.array(img, dtype=np.float32)
            variance = float(np.var(arr))
            return variance
        except Exception:
            return 0.0


# ============================================================
# 主处理器
# ============================================================

class CircuitModalProcessor(ImageModalProcessor):
    """
    面向电路图的垂直领域模态处理器。
    
    继承 ImageModalProcessor，在其基础上增加：
    1. 电路图检测（判断是否需要走结构恢复流程）
    2. 结构化 netlist 提取（通过特化的 VLM prompt）
    3. 元件→KG实体、连接→KG边 的映射
    4. 增强的 entity_info 输出（包含结构化电路信息）
    
    对于非电路图片，降级为 ImageModalProcessor 的标准行为。
    """

    # VLM prompt：要求同时输出 caption 和结构化电路信息
    CIRCUIT_ANALYSIS_PROMPT = """Analyze this circuit diagram image. Provide TWO outputs:

1. DESCRIPTION: A detailed natural language description of the circuit, including:
   - Circuit type and purpose
   - Key components and their values
   - Signal flow and topology
   - Any notable design characteristics

2. CIRCUIT_JSON: A structured JSON representation with this exact format:
{
  "circuit_name": "<descriptive name>",
  "circuit_type": "<e.g., inverting_amplifier, low_pass_filter, common_emitter>",
  "components": [
    {"id": "R1", "type": "resistor", "value": "10kΩ", "pins": ["node1", "node2"]},
    {"id": "C1", "type": "capacitor", "value": "100nF", "pins": ["node2", "GND"]},
    {"id": "U1", "type": "op_amp", "pins": ["inv_input", "non_inv_input", "output"]}
  ],
  "connections": [
    {"from": "R1", "to": "U1", "net": "inv_input", "type": "signal"},
    {"from": "R2", "to": "U1", "net": "feedback", "type": "feedback"}
  ],
  "description": "<brief functional description>"
}

If this is NOT a circuit diagram, respond with DESCRIPTION only and set CIRCUIT_JSON to null.

Surrounding context from the document:
{context}
"""

    def __init__(
        self,
        lightrag,
        modal_caption_func,
        context_extractor=None,
        circuit_confidence_threshold: float = 0.5,
        enable_structure_recovery: bool = True,
    ):
        super().__init__(lightrag, modal_caption_func, context_extractor)
        self.circuit_confidence_threshold = circuit_confidence_threshold
        self.enable_structure_recovery = enable_structure_recovery
        self._stats = {
            "total_processed": 0,
            "circuit_detected": 0,
            "structure_recovered": 0,
            "fallback_to_caption": 0,
        }

    async def process_multimodal_content(
        self,
        modal_content: Dict,
        content_type: str,
        file_path: str,
        entity_name: str,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        处理流程：
        
        1. 调用父类获取标准 caption + entity_info
        2. 判断是否为电路图
        3. 如果是 → 调用特化 prompt 获取结构化输出 → 解析 → 注入 KG
        4. 如果否 → 直接返回父类结果
        """
        self._stats["total_processed"] += 1

        # Step 1: 获取标准 caption（父类行为）
        base_description, base_entity_info = await super().process_multimodal_content(
            modal_content, content_type, file_path, entity_name, **kwargs
        )

        # Step 2: 判断是否为电路图
        caption_text = base_description or ""
        img_caption_list = modal_content.get("img_caption", [])
        combined_text = caption_text + " " + " ".join(img_caption_list)

        is_circuit = CircuitDetector.is_likely_circuit(combined_text)

        if not is_circuit or not self.enable_structure_recovery:
            self._stats["fallback_to_caption"] += 1
            return base_description, base_entity_info

        self._stats["circuit_detected"] += 1

        # Step 3: 调用特化 prompt 获取结构化电路信息
        circuit_design = await self._extract_circuit_structure(
            modal_content, combined_text
        )

        if circuit_design is None:
            # 结构恢复失败，降级为标准 caption
            logger.warning(
                f"Circuit structure recovery failed for {entity_name}, "
                f"falling back to caption-only mode"
            )
            self._stats["fallback_to_caption"] += 1
            return base_description, base_entity_info

        self._stats["structure_recovered"] += 1

        # Step 4: 将结构化信息融合到 entity_info 中
        enhanced_description = self._build_enhanced_description(
            base_description, circuit_design
        )
        enhanced_entity_info = self._build_enhanced_entity_info(
            base_entity_info, circuit_design, entity_name
        )

        logger.info(
            f"Circuit structure recovered: {circuit_design.circuit_name}, "
            f"{len(circuit_design.components)} components, "
            f"{len(circuit_design.connections)} connections"
        )

        return enhanced_description, enhanced_entity_info

    async def _extract_circuit_structure(
        self,
        modal_content: Dict,
        context: str,
    ) -> Optional[CircuitDesign]:
        """
        调用 VLM 提取电路结构化信息。
        尝试两种解析路径：JSON 格式 → SPICE netlist 格式。
        """
        prompt = self.CIRCUIT_ANALYSIS_PROMPT.format(context=context[:500])

        try:
            # 调用 VLM（复用父类的 modal_caption_func）
            img_path = modal_content.get("img_path", "")
            vlm_response = await self.modal_caption_func(
                prompt,
                image_path=img_path,
            )
        except Exception as e:
            logger.error(f"VLM call failed for circuit analysis: {e}")
            return None

        if not vlm_response:
            return None

        # 尝试 JSON 解析
        circuit = NetlistParser.parse_json_description(vlm_response)
        if circuit and len(circuit.components) > 0:
            return circuit

        # 降级：尝试 SPICE netlist 解析
        circuit = NetlistParser.parse_netlist_text(vlm_response)
        if circuit and len(circuit.components) > 0:
            return circuit

        return None

    def _build_enhanced_description(
        self,
        base_description: str,
        circuit: CircuitDesign,
    ) -> str:
        """
        构建增强描述：基础 caption + 结构化电路摘要。
        这段文本将参与文本 KG 的构建。
        """
        parts = [base_description]

        if circuit.circuit_type:
            parts.append(f"Circuit type: {circuit.circuit_type}.")

        if circuit.components:
            comp_list = ", ".join(
                f"{c.component_id} ({c.component_type}"
                + (f", {c.value}" if c.value else "")
                + ")"
                for c in circuit.components
            )
            parts.append(f"Components: {comp_list}.")

        if circuit.connections:
            conn_list = ", ".join(
                f"{c.source}→{c.target} via {c.net_name}"
                for c in circuit.connections[:10]  # 限制长度
            )
            parts.append(f"Connections: {conn_list}.")

        return " ".join(parts)

    def _build_enhanced_entity_info(
        self,
        base_entity_info: Dict,
        circuit: CircuitDesign,
        parent_entity_name: str,
    ) -> Dict[str, Any]:
        """
        构建增强的 entity_info。
        
        关键设计：将电路元件和连接关系直接映射为 KG 实体和边，
        而不是嵌入在描述文本中等待 LLM 做 entity extraction。
        
        这比 RAG-Anything 原生的"VLM caption → LLM extract entities"
        多了一层显式结构注入，跳过了 LLM 实体抽取可能引入的信息损失。
        """
        enhanced = dict(base_entity_info) if base_entity_info else {}

        circuit_info = circuit.to_entity_info()

        # 合并实体
        existing_entities = enhanced.get("entities", [])
        existing_entities.extend(circuit_info["entities"])
        enhanced["entities"] = existing_entities

        # 合并关系
        existing_relations = enhanced.get("relations", [])
        existing_relations.extend(circuit_info["relations"])
        enhanced["relations"] = existing_relations

        # 添加图像实体→电路实体的跨模态关系
        enhanced["relations"].append({
            "src_id": parent_entity_name,
            "tgt_id": circuit.circuit_name,
            "description": f"Image '{parent_entity_name}' depicts circuit '{circuit.circuit_name}'",
            "relation_type": "depicts",
        })

        # 附加元数据
        enhanced["circuit_metadata"] = circuit_info.get("metadata", {})

        return enhanced

    def get_stats(self) -> Dict[str, int]:
        """返回处理统计，用于实验评估"""
        return dict(self._stats)
```

### 2.3 处理器注册

修改 `raganything/modalprocessors/__init__.py`：

```python
from .circuit_processor import (
    CircuitModalProcessor,
    CircuitDesign,
    CircuitComponent,
    CircuitConnection,
    NetlistParser,
    CircuitDetector,
)
```

修改 `raganything/raganything.py` 中的处理器初始化逻辑，支持通过配置注册自定义处理器：

```python
# 在 RAGAnythingConfig 中添加
circuit_processing: bool = False  # 默认关闭，显式开启

# 在处理器初始化中添加
if config.circuit_processing:
    from raganything.modalprocessors import CircuitModalProcessor
    self.modal_processors["circuit_image"] = CircuitModalProcessor(
        lightrag=self.lightrag,
        modal_caption_func=self.modal_caption_func,
        context_extractor=self.context_extractor,
    )
```

### 2.4 图片路由逻辑修改

在 `_process_multimodal_content_batch_type_aware()` 中，对 image 类型增加电路检测分支：

```python
# 在 image 分发逻辑中
if config.circuit_processing and item.get("type") == "image":
    caption_hints = item.get("img_caption", [])
    combined_hints = " ".join(caption_hints)
    if CircuitDetector.is_likely_circuit(combined_hints):
        items_by_type["circuit_image"].append(item)
        continue
# 否则走原有 image pipeline
items_by_type["image"].append(item)
```

### 2.5 VLM Prompt 调优（关键工程细节）

VLM 对电路图的结构化输出质量是整个方案的瓶颈。需要迭代调优 prompt，具体策略：

**Prompt 版本管理**：

```python
# circuit_prompts.py — 维护多版本 prompt，方便 A/B 测试

PROMPT_V1_BASIC = """Describe this circuit diagram..."""

PROMPT_V2_JSON = """Analyze this circuit diagram. Output JSON..."""

PROMPT_V3_CONTEXTUAL = """
Given the following document context:
{context}

Analyze the circuit diagram and provide:
1. A natural language description
2. A structured JSON with components and connections
...
"""
```

**调优方向**：
- few-shot：在 prompt 中给 1-2 个电路 JSON 输出的示例
- 上下文注入：把周围文本（caption、章节标题）作为 hint
- 格式约束：严格要求 JSON schema，减少解析失败率

---

## 阶段三：集成测试与调试（Day 8-10）

### 3.1 单元测试

```python
# tests/test_circuit_processor.py

import pytest
from raganything.modalprocessors.circuit_processor import (
    NetlistParser, CircuitDetector, CircuitDesign
)


class TestNetlistParser:
    def test_parse_spice_resistor(self):
        text = "R1 net1 net2 10k\nR2 net2 net3 20k"
        circuit = NetlistParser.parse_netlist_text(text)
        assert len(circuit.components) == 2
        assert circuit.components[0].component_id == "R1"
        assert circuit.components[0].value == "10k"

    def test_parse_json_description(self):
        json_text = '''
        {
          "circuit_name": "Inverting Amplifier",
          "circuit_type": "inverting_amplifier",
          "components": [
            {"id": "R1", "type": "resistor", "value": "10kΩ", "pins": ["Vin", "inv"]},
            {"id": "R2", "type": "resistor", "value": "100kΩ", "pins": ["inv", "Vout"]},
            {"id": "U1", "type": "op_amp", "pins": ["inv", "GND", "Vout"]}
          ],
          "connections": [
            {"from": "R1", "to": "U1", "net": "inv_input"},
            {"from": "R2", "to": "U1", "net": "feedback"}
          ]
        }
        '''
        circuit = NetlistParser.parse_json_description(json_text)
        assert circuit is not None
        assert circuit.circuit_name == "Inverting Amplifier"
        assert len(circuit.components) == 3
        assert len(circuit.connections) == 2

    def test_parse_json_from_markdown_block(self):
        """VLM 输出经常包裹在 ```json ... ``` 中"""
        text = "Here is the analysis:\n```json\n{\"circuit_name\": \"RC Filter\", \"components\": [{\"id\": \"R1\", \"type\": \"resistor\", \"value\": \"1kΩ\", \"pins\": [\"in\", \"out\"]}], \"connections\": []}\n```"
        circuit = NetlistParser.parse_json_description(text)
        assert circuit is not None
        assert circuit.circuit_name == "RC Filter"

    def test_entity_info_structure(self):
        circuit = CircuitDesign(
            circuit_name="test_circuit",
            components=[
                CircuitComponent("R1", "resistor", "10k", ["n1", "n2"]),
                CircuitComponent("C1", "capacitor", "100n", ["n2", "GND"]),
            ],
            connections=[
                CircuitConnection("R1", "C1", "n2"),
            ],
        )
        info = circuit.to_entity_info()
        assert len(info["entities"]) == 3  # circuit + R1 + C1
        assert len(info["relations"]) == 3  # 2 belongs_to + 1 connection
        assert info["metadata"]["component_count"] == 2


class TestCircuitDetector:
    def test_detect_circuit_caption(self):
        assert CircuitDetector.is_likely_circuit(
            "Figure 3: Inverting amplifier circuit schematic with resistor feedback"
        )

    def test_reject_non_circuit(self):
        assert not CircuitDetector.is_likely_circuit(
            "Figure 1: Training loss over 100 epochs"
        )
```

### 3.2 端到端集成测试

```python
# tests/test_circuit_e2e.py

async def test_circuit_pipeline():
    """端到端测试：PDF → CircuitModalProcessor → KG → Query"""
    config = RAGAnythingConfig(
        working_dir="./test_circuit_rag",
        circuit_processing=True,
    )
    # ... 初始化 LightRAG 和 RAGAnything ...

    # 处理包含电路图的 PDF
    await rag.process_document_complete("test_circuit_doc.pdf")

    # 验证 KG 中是否包含电路元件实体
    # （需要根据 LightRAG 的 KG API 调整）
    kg_entities = await rag.lightrag.get_entities()
    circuit_entities = [
        e for e in kg_entities
        if e.get("entity_type") == "circuit_component"
    ]
    assert len(circuit_entities) > 0, "No circuit components found in KG"

    # 查询测试
    result = await rag.aquery(
        "What is the value of R1 in the amplifier circuit?",
        mode="hybrid",
    )
    assert "R1" in result or "10k" in result
```

### 3.3 调试检查清单

- [ ] MinerU 能否正确提取电路图图片文件
- [ ] `CircuitDetector` 对测试集的分类准确率 > 90%
- [ ] VLM 对电路图的 JSON 输出解析成功率 > 70%
- [ ] 解析失败时是否正确降级为标准 caption
- [ ] KG 中能查到 `circuit_component` 类型实体
- [ ] `hybrid` 模式查询能检索到电路元件信息
- [ ] 处理非电路文档时不引入副作用

---

## 阶段四：对比实验设计与执行（Day 11-17）

### 4.1 实验总体设计

**核心研究问题**：在电路类文档上，将电路图恢复为结构化 KG 实体（structure recovery）是否比纯 caption 化（caption-only）带来可测量的检索和生成质量提升？

### 4.2 实验矩阵

| 实验编号 | 方法 | 多模态策略 | 知识图谱内容 |
|----------|------|------------|--------------|
| **Baseline-A** | 纯文本 LightRAG | 忽略图片 | 仅文本实体 |
| **Baseline-B** | RAG-Anything (原生) | ImageModalProcessor caption | 文本实体 + 图片 caption 实体 |
| **Ours** | RAG-Anything + CircuitModalProcessor | 结构恢复 | 文本实体 + 元件实体 + 连接边 |

### 4.3 测试集规格

```
test_dataset/
├── docs/                        # 测试文档
│   ├── circuit_textbook_ch5.pdf  # A类：教材电路章节
│   ├── circuit_textbook_ch8.pdf
│   ├── lab_report_opamp.pdf      # B类：实验文档
│   ├── lab_report_filter.pdf
│   └── general_ml_paper.pdf      # C类：对照（非电路）
├── queries/
│   ├── component_queries.json    # 元件值查询（15题）
│   ├── topology_queries.json     # 拓扑理解查询（15题）
│   ├── crossmodal_queries.json   # 跨模态推理查询（10题）
│   └── general_queries.json      # 通用查询对照（10题）
└── ground_truth/
    ├── component_answers.json    # 人工标注的标准答案
    ├── topology_answers.json
    └── crossmodal_answers.json
```

**查询集示例**：

```json
{
  "queries": [
    {
      "id": "comp_01",
      "category": "component_value",
      "question": "在图5.3的反相放大电路中，反馈电阻R2的阻值是多少？",
      "ground_truth": "100kΩ",
      "source_doc": "circuit_textbook_ch5.pdf",
      "source_page": 12,
      "difficulty": "easy",
      "requires_visual": true
    },
    {
      "id": "topo_01",
      "category": "topology",
      "question": "该低通滤波器是一阶还是二阶？判断依据是什么？",
      "ground_truth": "二阶，因为包含两个RC级联网络",
      "source_doc": "lab_report_filter.pdf",
      "source_page": 3,
      "difficulty": "medium",
      "requires_visual": true
    },
    {
      "id": "cross_01",
      "category": "crossmodal",
      "question": "图3中放大器的实际增益与文中公式计算的理论增益相比，误差是多少？",
      "ground_truth": "理论增益 -10V/V，实际约 -9.5V/V，误差约 5%",
      "source_doc": "circuit_textbook_ch8.pdf",
      "source_page": 25,
      "difficulty": "hard",
      "requires_visual": true
    }
  ]
}
```

### 4.4 评估指标

#### 指标 1：KG 实体覆盖率（自动化）

```python
def compute_entity_coverage(kg_entities, ground_truth_components):
    """
    衡量 KG 中是否包含了文档中实际存在的电路元件。
    
    ground_truth_components: 人工标注的文档中所有电路元件列表
    kg_entities: 从 KG 中导出的实体列表
    """
    gt_set = set(c.lower() for c in ground_truth_components)
    kg_set = set(e["entity_name"].lower() for e in kg_entities)
    
    recall = len(gt_set & kg_set) / len(gt_set) if gt_set else 0
    precision = len(gt_set & kg_set) / len(kg_set) if kg_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```

#### 指标 2：检索召回率（自动化）

```python
def compute_retrieval_recall(retrieved_chunks, ground_truth_evidence):
    """
    衡量检索结果是否覆盖了回答所需的关键证据。
    
    ground_truth_evidence: 人工标注的"回答这个问题需要的关键信息片段"
    retrieved_chunks: 检索返回的 chunk 列表
    """
    retrieved_text = " ".join(c["content"] for c in retrieved_chunks)
    hits = sum(
        1 for evidence in ground_truth_evidence
        if evidence.lower() in retrieved_text.lower()
    )
    return hits / len(ground_truth_evidence) if ground_truth_evidence else 0
```

#### 指标 3：回答质量 — LLM-as-Judge（参考 RAG-Anything 论文 Appendix A.4）

```python
JUDGE_PROMPT = """You are evaluating the quality of a RAG system's answer about electronic circuits.

Question: {question}
Ground Truth Answer: {ground_truth}
System Answer: {system_answer}

Rate the system answer on these dimensions (1-5 each):

1. **Factual Accuracy**: Are component values, circuit types, and technical details correct?
2. **Completeness**: Does the answer cover all aspects of the ground truth?
3. **Specificity**: Does the answer reference specific components (R1, C2, etc.) rather than generic descriptions?
4. **Cross-Modal Integration**: Does the answer effectively combine information from both text and circuit diagrams?

Respond in JSON format:
{{"accuracy": <1-5>, "completeness": <1-5>, "specificity": <1-5>, "cross_modal": <1-5>, "explanation": "<brief>"}}
"""
```

#### 指标 4：效率指标（自动化）

```python
@dataclass
class EfficiencyMetrics:
    total_processing_time_s: float     # 文档处理总耗时
    vlm_calls_count: int               # VLM 调用次数
    vlm_total_tokens: int              # VLM 消耗 token 数
    kg_entity_count: int               # KG 实体总数
    kg_edge_count: int                 # KG 边总数
    query_latency_avg_ms: float        # 平均查询延迟
    query_latency_p95_ms: float        # P95 查询延迟
```

### 4.5 实验执行脚本

```python
# experiments/run_comparison.py

import asyncio
import json
import time
from pathlib import Path

async def run_experiment(
    method_name: str,
    config: RAGAnythingConfig,
    docs: List[str],
    queries: List[dict],
):
    """执行单个实验方法的完整流程"""
    results = {
        "method": method_name,
        "docs_processed": [],
        "query_results": [],
        "kg_stats": {},
        "efficiency": {},
    }

    # 初始化
    rag = RAGAnything(config=config, lightrag=create_lightrag(config))

    # ---- 入库阶段 ----
    t_start = time.time()
    for doc_path in docs:
        await rag.process_document_complete(doc_path)
        results["docs_processed"].append(doc_path)
    t_ingest = time.time() - t_start

    # ---- KG 统计 ----
    # 导出实体和边的统计
    results["kg_stats"] = await extract_kg_stats(rag.lightrag)

    # ---- 查询阶段 ----
    query_latencies = []
    for q in queries:
        t_q = time.time()
        answer = await rag.aquery(q["question"], mode="hybrid")
        latency = (time.time() - t_q) * 1000
        query_latencies.append(latency)

        results["query_results"].append({
            "query_id": q["id"],
            "question": q["question"],
            "answer": answer,
            "ground_truth": q["ground_truth"],
            "latency_ms": latency,
        })

    # ---- 效率统计 ----
    results["efficiency"] = {
        "ingest_time_s": t_ingest,
        "avg_query_latency_ms": sum(query_latencies) / len(query_latencies),
        "p95_query_latency_ms": sorted(query_latencies)[int(len(query_latencies) * 0.95)],
    }

    return results


async def main():
    docs = ["docs/circuit_textbook_ch5.pdf", "docs/lab_report_opamp.pdf"]
    queries = json.loads(Path("queries/all_queries.json").read_text())

    # ---- Baseline A: 纯文本 LightRAG ----
    config_a = RAGAnythingConfig(
        working_dir="./exp_baseline_a",
        enable_image_processing=False,
        enable_table_processing=False,
    )
    results_a = await run_experiment("Baseline_TextOnly", config_a, docs, queries)

    # ---- Baseline B: RAG-Anything 原生 ----
    config_b = RAGAnythingConfig(
        working_dir="./exp_baseline_b",
        enable_image_processing=True,
        enable_table_processing=True,
    )
    results_b = await run_experiment("RAGAnything_Native", config_b, docs, queries)

    # ---- Ours: CircuitModalProcessor ----
    config_c = RAGAnythingConfig(
        working_dir="./exp_ours",
        enable_image_processing=True,
        enable_table_processing=True,
        circuit_processing=True,
    )
    results_c = await run_experiment("Ours_CircuitProcessor", config_c, docs, queries)

    # ---- 评估 ----
    all_results = [results_a, results_b, results_c]
    evaluate_and_report(all_results)


if __name__ == "__main__":
    asyncio.run(main())
```

### 4.6 结果可视化

实验完成后，生成以下图表（用于个人主页）：

1. **柱状图**：三种方法在 component / topology / crossmodal 三类查询上的准确率
2. **表格**：KG 实体覆盖率对比（precision / recall / F1）
3. **雷达图**：四个质量维度（accuracy / completeness / specificity / cross_modal）
4. **成本分析表**：VLM token 消耗 vs 质量提升的 ROI

---

## 阶段五：整理产出与个人主页撰写（Day 18-21）

### 5.1 代码产出

```
你的 GitHub 仓库结构：
RAG-Anything/                          # Fork 自 HKUDS
├── raganything/modalprocessors/
│   └── circuit_processor.py           # 你的核心贡献
├── tests/
│   └── test_circuit_processor.py      # 测试
├── experiments/
│   ├── run_comparison.py              # 实验脚本
│   ├── evaluate.py                    # 评估脚本
│   └── results/                       # 实验结果
└── docs/
    └── circuit_modal_processor.md     # 技术文档
```

### 5.2 个人主页内容结构

```markdown
## RAG-Anything × Circuit Structure Recovery

### Motivation
RAG-Anything 将多模态内容 caption 化后注入知识图谱。但在电路类文档中，
caption 会丢失元件参数和拓扑连接等关键结构信息。
我基于 ZhaoXi 项目的 netlist 恢复经验，开发了 CircuitModalProcessor，
将电路图直接恢复为结构化实体和关系，注入 dual-graph KG。

### Technical Approach
[系统架构图]
[关键代码片段]

### Experimental Results
[对比实验图表]
- 元件值查询准确率：Ours xx% vs Baseline-B xx% (+xx%)
- 跨模态推理查询：Ours xx% vs Baseline-B xx% (+xx%)
- KG 实体覆盖率 F1：Ours xx vs Baseline-B xx

### Key Insight
结构恢复（structure recovery）相比 caption 化（caption-only）的核心优势在于：
将领域结构知识从 LLM 的隐式理解提升为 KG 的显式表示，
使检索可以直接在元件和连接层面进行精确匹配，
而不是依赖嵌入空间中 caption 文本的模糊相似度。
```

### 5.3 向 HKUDS 提 PR 的策略

**PR 1（小而确定，优先提交）**：
- 在 docs/ 中添加 `circuit_modal_processor.md` 文档
- 说明如何为垂直领域开发自定义模态处理器
- 这本身对 RAG-Anything 社区有价值

**PR 2（代码贡献，如果实验效果好）**：
- 提交 `circuit_processor.py` 作为示例/可选模态处理器
- 附带测试和文档

### 5.4 申请邮件中的技术 Pitch

> I have hands-on experience building a multimodal RAG system for electronics education (ZhaoXi, National Innovation Programme), where I designed a circuit diagram structure recovery pipeline that converts visual schematics into parseable netlists and structured circuit objects. Recently, I extended your RAG-Anything framework with a CircuitModalProcessor that injects circuit components and connections as first-class KG entities — going beyond caption-based multimodal processing. My comparative experiments on circuit-domain documents show [xx%] improvement in component-level retrieval accuracy over the native ImageModalProcessor baseline.

---

## 时间规划总览

| Day | 阶段 | 核心任务 | 交付物 |
|-----|------|----------|--------|
| 1-2 | 前置准备 | 环境搭建、源码精读、测试数据准备 | 可运行的开发环境 + 测试数据集 |
| 3-5 | 核心开发 | CircuitModalProcessor 主体代码 | circuit_processor.py 完成 |
| 6-7 | Prompt 调优 | VLM prompt 迭代、JSON 解析率优化 | prompt 版本确定，解析率 > 70% |
| 8-10 | 集成测试 | 单元测试 + 端到端测试 + 调试 | 全部测试通过 |
| 11-13 | 实验准备 | 查询集标注、Ground truth 准备 | 完整的 benchmark 数据集 |
| 14-16 | 实验执行 | 三组对比实验运行 + 评估 | 原始实验数据 |
| 17 | 结果分析 | 统计分析 + 可视化 | 实验图表 |
| 18-19 | 整理代码 | 代码清理 + 文档撰写 | 可提 PR 的代码仓库 |
| 20-21 | 主页撰写 | 个人主页项目展示页 | 完整展示内容 |
