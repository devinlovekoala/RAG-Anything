#!/usr/bin/env python3
"""
将各 topic 的 ablation_results.json 导出为统一的 baseline / enhanced 结果文件。

示例:
  python export_ablation_results.py \
    --benchmark-master benchmark_master.jsonl \
    --experiment E1 \
    --baseline-condition naive \
    --enhanced-condition hybrid \
    --topic-result bjt=output_ablation_benchmark_bjt_v1/ablation_results.json \
    --topic-result fet=output_ablation_benchmark_fet_v1/ablation_results.json \
    --topic-result freq_domain=output_ablation_benchmark_freq_domain_v1/ablation_results.json \
    --topic-result opamp=output_ablation_benchmark_opamp_v1/ablation_results.json \
    --baseline-out results/results_baseline.json \
    --enhanced-out results/results_enhanced.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_master(master_path: Path) -> dict[str, dict]:
    records = {}
    with open(master_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records[f"{record['topic']}::{record['question']}"] = record
    return records


def load_ablation(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_result_record(record: dict, method: str, answer: str) -> dict:
    return {
        "qid": record["qid"],
        "doc_id": record["doc_id"],
        "question": record["question"],
        "answer": answer or "",
        "correct_answer": record["gold_answer"],
        "category": record["category"],
        "topic": record["topic"],
        "method": method,
        "rubric": record.get("rubric", {}),
    }


def parse_topic_result(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"无效的 --topic-result 参数: {spec}")
    topic, path = spec.split("=", 1)
    return topic.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-master", required=True)
    parser.add_argument("--experiment", default="E1")
    parser.add_argument("--baseline-condition", required=True)
    parser.add_argument("--enhanced-condition", required=True)
    parser.add_argument("--topic-result", action="append", required=True,
                        help="格式: topic=path/to/ablation_results.json")
    parser.add_argument("--baseline-out", required=True)
    parser.add_argument("--enhanced-out", required=True)
    args = parser.parse_args()

    master = load_master(Path(args.benchmark_master))
    baseline_rows = []
    enhanced_rows = []

    for spec in args.topic_result:
        topic, result_path = parse_topic_result(spec)
        payload = load_ablation(result_path)
        experiment_data = payload.get(args.experiment, {})
        conditions = experiment_data.get("conditions", {})
        baseline_items = conditions.get(args.baseline_condition, [])
        enhanced_items = conditions.get(args.enhanced_condition, [])

        if not baseline_items:
            raise ValueError(f"{result_path} 中未找到 baseline condition={args.baseline_condition}")
        if not enhanced_items:
            raise ValueError(f"{result_path} 中未找到 enhanced condition={args.enhanced_condition}")

        baseline_by_question = {item["question"]: item for item in baseline_items}
        enhanced_by_question = {item["question"]: item for item in enhanced_items}
        topic_master_items = [
            record for record in master.values()
            if record["topic"] == topic
        ]

        for record in topic_master_items:
            question = record["question"]
            baseline_answer = baseline_by_question.get(question, {}).get("answer", "")
            enhanced_answer = enhanced_by_question.get(question, {}).get("answer", "")
            baseline_rows.append(build_result_record(record, "baseline", baseline_answer))
            enhanced_rows.append(build_result_record(record, "enhanced", enhanced_answer))

        missing_baseline = sorted(
            question for question in (record["question"] for record in topic_master_items)
            if question not in baseline_by_question
        )
        missing_enhanced = sorted(
            question for question in (record["question"] for record in topic_master_items)
            if question not in enhanced_by_question
        )
        if missing_baseline or missing_enhanced:
            raise ValueError(
                f"{topic} 结果与 benchmark 题集不匹配: "
                f"missing_baseline={len(missing_baseline)}, "
                f"missing_enhanced={len(missing_enhanced)}"
            )

    for output_path, rows in [
        (Path(args.baseline_out), baseline_rows),
        (Path(args.enhanced_out), enhanced_rows),
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"[OK] 写入 {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
