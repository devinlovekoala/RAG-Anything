#!/usr/bin/env python3
"""
Step 6: 准备电路实验数据清单与 QA 模板

功能：
1. 扫描 experiment_data/ 下的 PDF 课件
2. 生成实验清单 manifest
3. 按启发式规则给出文档分组建议
4. 为后续人工补题生成 QA 模板

用法:
    python reproduce/05_prepare_circuit_experiment_data.py
    python reproduce/05_prepare_circuit_experiment_data.py --data-dir ./experiment_data --output-dir ./experiment_data/prepared
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


PDF_SUFFIXES = {".pdf", ".PDF"}

GROUP_RULES = {
    "frequency_domain": [
        "frequency domain",
        "phasor",
        "frequency",
    ],
    "analog": [
        "op amp",
        "operation amplifier",
        "bjt",
        "fets",
        "feedback",
        "power amp",
        "multistage",
    ],
    "circuit_analysis": [
        "kcl",
        "kvl",
        "thevenin",
        "norton",
        "superposition",
        "node-voltage",
        "mesh",
        "rlc",
        "response",
        "source transformation",
    ],
    "answer_sheet": [
        "答案",
        "answer",
    ],
}

BASE_QUESTION_TEMPLATES = [
    {
        "category": "lecture_scope",
        "question": "这份讲义主要讲解了哪些核心概念、定律或电路类型？",
        "answer": "",
    },
    {
        "category": "worked_example",
        "question": "讲义中最典型的例题或代表性电路是什么，它要解决什么问题？",
        "answer": "",
    },
    {
        "category": "study_focus",
        "question": "如果我要快速复习这份讲义，最应该优先关注哪些公式、图示、结论或分析步骤？",
        "answer": "",
    },
]

ANALOG_QUESTION_TEMPLATES = [
    {
        "category": "component_value",
        "question": "该讲义代表性模拟电路中的关键器件参数分别是什么？",
        "answer": "",
    },
    {
        "category": "circuit_function",
        "question": "讲义中的代表性模拟电路输入是什么、输出是什么，它希望实现怎样的功能？",
        "answer": "",
    },
    {
        "category": "bias_or_operation_region",
        "question": "文档中的关键模拟器件工作在什么偏置或工作区间，依据是什么？",
        "answer": "",
    },
    {
        "category": "topology",
        "question": "讲义展示的核心电路属于什么拓扑，输入、输出和反馈路径如何识别？",
        "answer": "",
    },
    {
        "category": "formula_mapping",
        "question": "文档中的主要公式与电路图中的器件、节点或参数分别对应什么物理意义？",
        "answer": "",
    },
    {
        "category": "assumptions_or_approximations",
        "question": "讲义在分析该模拟电路时采用了哪些理想条件、近似假设或默认前提？",
        "answer": "",
    },
]

CIRCUIT_ANALYSIS_TEMPLATES = [
    {
        "category": "known_unknown",
        "question": "该讲义例题中已知条件、待求量和使用的方法分别是什么？",
        "answer": "",
    },
    {
        "category": "method_selection",
        "question": "如果把这道题当成一次作业题，应该优先选择哪种分析方法，为什么？",
        "answer": "",
    },
    {
        "category": "equation_setup",
        "question": "文档中的关键方程是如何根据 KCL、KVL、元件关系或等效变换一步步建立起来的？",
        "answer": "",
    },
    {
        "category": "equivalent_transformation",
        "question": "如果文档涉及等效变换，原电路与等效电路之间是如何对应的？",
        "answer": "",
    },
    {
        "category": "solution_steps",
        "question": "该讲义中的典型电路题目是按什么步骤完成求解的？",
        "answer": "",
    },
    {
        "category": "result_interpretation",
        "question": "最终求得的电压、电流、功率或响应结果说明了什么电路特性？",
        "answer": "",
    },
]

FREQUENCY_DOMAIN_TEMPLATES = [
    {
        "category": "component_value",
        "question": "该讲义代表性频域电路中的关键器件参数分别是什么？",
        "answer": "",
    },
    {
        "category": "topology",
        "question": "讲义展示的核心电路属于什么拓扑，信号路径和关键节点如何识别？",
        "answer": "",
    },
    {
        "category": "formula_mapping",
        "question": "文档中的主要频域公式与电路图中的阻抗、节点、电压或电流分别对应什么物理意义？",
        "answer": "",
    },
    {
        "category": "phasor_analysis",
        "question": "讲义中的相量分析方法如何应用于该电路，阻抗与相位应如何表示？",
        "answer": "",
    },
    {
        "category": "node_mesh_freq",
        "question": "频域中节点电压法或网孔电流法如何建立方程，与时域分析相比最关键的区别是什么？",
        "answer": "",
    },
    {
        "category": "transfer_function",
        "question": "该电路的传递函数、频率响应、截止频率或谐振点是如何得到的？",
        "answer": "",
    },
]

UNCLASSIFIED_TEMPLATES = [
    {
        "category": "main_topic",
        "question": "这份讲义的主要主题是什么，文中重点围绕哪些电路或分析对象展开？",
        "answer": "",
    },
    {
        "category": "diagram_text_alignment",
        "question": "讲义中的图示与正文解释之间最重要的对应关系是什么？",
        "answer": "",
    },
    {
        "category": "study_focus",
        "question": "如果把这份讲义当作复习材料，最值得先读懂的图、公式或结论分别是什么？",
        "answer": "",
    },
]

TITLE_KEYWORD_TEMPLATES = {
    "op amp": [
        {
            "category": "opamp_terminal_gain",
            "question": "讲义中的运算放大器电路属于反相、同相还是其他结构，其增益关系如何确定？",
            "answer": "",
        },
        {
            "category": "opamp_feedback",
            "question": "运放电路中的反馈网络由哪些元件构成，它如何影响输出？",
            "answer": "",
        },
        {
            "category": "opamp_assumptions",
            "question": "讲义在推导运放电路结论时用了哪些理想运放假设，这些假设会影响哪些结果？",
            "answer": "",
        },
    ],
    "operation amplifier": [
        {
            "category": "opamp_terminal_gain",
            "question": "讲义中的运算放大器电路属于反相、同相还是其他结构，其增益关系如何确定？",
            "answer": "",
        },
        {
            "category": "opamp_feedback",
            "question": "运放电路中的反馈网络由哪些元件构成，它如何影响输出？",
            "answer": "",
        },
        {
            "category": "opamp_assumptions",
            "question": "讲义在推导运放电路结论时用了哪些理想运放假设，这些假设会影响哪些结果？",
            "answer": "",
        },
    ],
    "bjt": [
        {
            "category": "bjt_region",
            "question": "文档中的 BJT 电路工作在截止区、放大区还是饱和区，依据是什么？",
            "answer": "",
        },
        {
            "category": "bjt_bias_path",
            "question": "BJT 偏置网络由哪些元件构成，基极、集电极和发射极分别如何连接？",
            "answer": "",
        },
        {
            "category": "bjt_parameter_dependency",
            "question": "在文档讨论的 BJT 电路里，哪些参数最直接影响静态工作点、增益或输出摆幅？",
            "answer": "",
        },
    ],
    "fets": [
        {
            "category": "fet_region",
            "question": "讲义中的 FET 工作在什么工作区，栅极、漏极和源极电压关系如何判断？",
            "answer": "",
        },
        {
            "category": "fet_topology",
            "question": "该 FET 电路是共源、共漏还是共栅结构，输入输出分别位于哪里？",
            "answer": "",
        },
        {
            "category": "fet_bias_path",
            "question": "MOSFET 或 FET 的偏置网络由哪些元件构成，栅极、漏极和源极分别如何连接？",
            "answer": "",
        },
    ],
    "thevenin": [
        {
            "category": "thevenin_equivalent",
            "question": "该讲义中的原电路如何化简为戴维宁等效电路，等效电压和等效电阻分别是什么？",
            "answer": "",
        },
    ],
    "norton": [
        {
            "category": "norton_equivalent",
            "question": "该讲义中的原电路如何化简为诺顿等效电路，等效电流和等效电阻分别是什么？",
            "answer": "",
        },
    ],
    "superposition": [
        {
            "category": "superposition_sources",
            "question": "若使用叠加定理，文档中的各独立源需要如何分别处理，最终结果如何叠加？",
            "answer": "",
        },
    ],
    "rlc": [
        {
            "category": "rlc_response",
            "question": "该 RLC 电路对应的是自然响应还是受迫响应，其阻尼状态如何判断？",
            "answer": "",
        },
    ],
    "frequency": [
        {
            "category": "frequency_response",
            "question": "讲义中的频域分析重点关注哪些量，它们如何随频率变化？",
            "answer": "",
        },
        {
            "category": "impedance_interpretation",
            "question": "文档中的复阻抗表达式分别对应哪些元件，它们如何决定幅值和相位的变化？",
            "answer": "",
        },
    ],
    "feedback": [
        {
            "category": "feedback_effect",
            "question": "该反馈电路采用了什么反馈方式，它对增益、稳定性或带宽有什么影响？",
            "answer": "",
        },
    ],
    "power amp": [
        {
            "category": "power_amplifier_mode",
            "question": "该功率放大电路属于哪一类工作方式，效率和失真特性如何体现？",
            "answer": "",
        },
    ],
}


def detect_group(name: str) -> str:
    lowered = name.lower()
    for group, keywords in GROUP_RULES.items():
        if any(keyword in lowered for keyword in keywords):
            return group
    return "unclassified"


def infer_tags(name: str) -> List[str]:
    lowered = name.lower()
    tags = []
    for token in [
        "op amp",
        "bjt",
        "fets",
        "feedback",
        "power amp",
        "frequency",
        "rlc",
        "thevenin",
        "norton",
        "superposition",
        "phasor",
        "answer",
        "答案",
    ]:
        if token in lowered:
            tags.append(token)
    return tags


def slugify(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^\w\u4e00-\u9fff]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "document"


def build_manifest(pdf_files: List[Path], data_dir: Path) -> List[Dict]:
    manifest = []
    for index, path in enumerate(sorted(pdf_files), start=1):
        relative_path = path.relative_to(data_dir)
        stem = path.stem
        manifest.append(
            {
                "doc_id": f"circuit_doc_{index:03d}",
                "file_name": path.name,
                "relative_path": str(relative_path),
                "group": detect_group(stem),
                "tags": infer_tags(stem),
                "title_hint": stem,
                "qa_file": f"queries/{slugify(stem)}.json",
                "gold_file": f"gold/{slugify(stem)}.json",
                "status": "pending_qa",
            }
        )
    return manifest


def build_qa_template(manifest_entry: Dict) -> Dict:
    questions = list(BASE_QUESTION_TEMPLATES)
    group = manifest_entry["group"]
    title_hint = manifest_entry["title_hint"].lower()

    if group == "analog":
        questions.extend(ANALOG_QUESTION_TEMPLATES)
    elif group == "frequency_domain":
        questions.extend(FREQUENCY_DOMAIN_TEMPLATES)
    elif group == "circuit_analysis":
        questions.extend(CIRCUIT_ANALYSIS_TEMPLATES)
    else:
        questions.extend(UNCLASSIFIED_TEMPLATES)

    for keyword, keyword_templates in TITLE_KEYWORD_TEMPLATES.items():
        if keyword in title_hint:
            questions.extend(keyword_templates)

    seen_questions = set()
    deduped_questions = []
    for item in questions:
        question_text = item["question"]
        if question_text in seen_questions:
            continue
        seen_questions.add(question_text)
        deduped_questions.append(item)

    return {
        "doc_id": manifest_entry["doc_id"],
        "file_name": manifest_entry["file_name"],
        "group": manifest_entry["group"],
        "questions": deduped_questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare circuit experiment dataset metadata")
    parser.add_argument("--data-dir", default="experiment_data")
    parser.add_argument("--output-dir", default="experiment_data/prepared")
    parser.add_argument(
        "--force-refresh-queries",
        action="store_true",
        help="覆盖已存在的 queries 模板，按新的课程主题问题模板重新生成",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    queries_dir = output_dir / "queries"
    gold_dir = output_dir / "gold"

    output_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = [
        path for path in data_dir.iterdir() if path.is_file() and path.suffix in PDF_SUFFIXES
    ]
    manifest = build_manifest(pdf_files, data_dir)

    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for entry in manifest:
        qa_template = build_qa_template(entry)
        qa_path = queries_dir / Path(entry["qa_file"]).name
        gold_path = gold_dir / Path(entry["gold_file"]).name
        if args.force_refresh_queries or not qa_path.exists():
            qa_path.write_text(json.dumps(qa_template, ensure_ascii=False, indent=2), encoding="utf-8")
        if not gold_path.exists():
            gold_template = {
                "doc_id": entry["doc_id"],
                "file_name": entry["file_name"],
                "answers": [],
            }
            gold_path.write_text(
                json.dumps(gold_template, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    group_stats: Dict[str, int] = {}
    for entry in manifest:
        group_stats[entry["group"]] = group_stats.get(entry["group"], 0) + 1

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "document_count": len(manifest),
        "group_stats": group_stats,
        "next_step": "Fill prepared/queries/*.json and prepared/gold/*.json, then run pipeline and ablation scripts.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[prepared] manifest: {manifest_path}")
    print(f"[prepared] summary: {summary_path}")
    print(f"[prepared] documents: {len(manifest)}")
    for group, count in sorted(group_stats.items()):
        print(f"  - {group}: {count}")


if __name__ == "__main__":
    main()
