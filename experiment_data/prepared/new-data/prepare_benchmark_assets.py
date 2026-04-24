#!/usr/bin/env python3
"""
从 benchmark_master.jsonl 生成：
1. 按 topic 拆分的 runtime JSONL
2. baseline / enhanced 结果模板
3. topic manifest，方便后续跑 topic-store 与 ablation
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


PLACEHOLDER_ANSWER = "__TO_BE_FILLED_BY_EXPERIMENT__"
TOPIC_WORKSPACE_MAP = {
    "bjt": {
        "doc_path": "experiment_data/2024-ch2-BJTs.pdf",
        "working_dir": "./rag_storage_benchmark_bjt_v1",
        "topic_output_dir": "./output_benchmark_bjt_v1",
        "ablation_output_dir": "./output_ablation_benchmark_bjt_v1",
    },
    "fet": {
        "doc_path": "experiment_data/2024-ch3-FETs-Enhance.pdf",
        "working_dir": "./rag_storage_benchmark_fet_v1",
        "topic_output_dir": "./output_benchmark_fet_v1",
        "ablation_output_dir": "./output_ablation_benchmark_fet_v1",
    },
    "freq_domain": {
        "doc_path": "experiment_data/2024-ch5-frequency.pdf",
        "working_dir": "./rag_storage_benchmark_freq_domain_v1",
        "topic_output_dir": "./output_benchmark_freq_domain_v1",
        "ablation_output_dir": "./output_ablation_benchmark_freq_domain_v1",
    },
    "opamp": {
        "doc_path": "experiment_data/2024-ch8-op amp.pdf",
        "working_dir": "./rag_storage_benchmark_opamp_v1",
        "topic_output_dir": "./output_benchmark_opamp_v1",
        "ablation_output_dir": "./output_ablation_benchmark_opamp_v1",
    },
}


def load_master(master_path: Path) -> list[dict]:
    records = []
    with open(master_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_runtime_record(record: dict) -> dict:
    return {
        "qid": record["qid"],
        "doc_id": record["doc_id"],
        "file_name": record["file_name"],
        "group": record["group"],
        "category": record["category"],
        "question": record["question"],
        "answer": record["gold_answer"],
    }


def build_result_template_record(record: dict, method: str) -> dict:
    return {
        "qid": record["qid"],
        "doc_id": record["doc_id"],
        "question": record["question"],
        "answer": PLACEHOLDER_ANSWER,
        "correct_answer": record["gold_answer"],
        "category": record["category"],
        "topic": record["topic"],
        "method": method,
        "rubric": record.get("rubric", {}),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    root = Path(__file__).resolve().parent
    master_path = root / "benchmark_master.jsonl"
    runtime_all_path = root / "benchmark_runtime_all.jsonl"
    baseline_template_path = root / "results_baseline_template.json"
    enhanced_template_path = root / "results_enhanced_template.json"
    manifest_path = root / "benchmark_manifest.json"
    runtime_dir = root / "runtime"
    templates_dir = root / "templates"

    runtime_dir.mkdir(exist_ok=True)
    templates_dir.mkdir(exist_ok=True)

    records = load_master(master_path)
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_topic[record["topic"]].append(record)

    runtime_all = [build_runtime_record(record) for record in records]
    baseline_template = [build_result_template_record(record, "baseline") for record in records]
    enhanced_template = [build_result_template_record(record, "enhanced") for record in records]

    write_jsonl(runtime_all_path, runtime_all)
    write_json(baseline_template_path, baseline_template)
    write_json(enhanced_template_path, enhanced_template)
    write_json(templates_dir / "results_baseline_template.json", baseline_template)
    write_json(templates_dir / "results_enhanced_template.json", enhanced_template)

    manifest = {"topics": {}}
    for topic, topic_records in sorted(by_topic.items()):
        runtime_rows = [build_runtime_record(record) for record in topic_records]
        write_jsonl(runtime_dir / f"benchmark_runtime_{topic}.jsonl", runtime_rows)
        manifest["topics"][topic] = {
            "question_count": len(topic_records),
            "file_name": topic_records[0]["file_name"],
            **TOPIC_WORKSPACE_MAP.get(topic, {}),
            "runtime_file": f"runtime/benchmark_runtime_{topic}.jsonl",
        }

    write_json(manifest_path, manifest)
    print(f"[OK] 写入 runtime files -> {runtime_dir}")
    print(f"[OK] 写入 templates -> {templates_dir}")
    print(f"[OK] 写入 manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
