#!/usr/bin/env python3
"""
Step 9: 从 prepared/queries 生成实验用 QA JSONL

功能：
1. 读取 experiment_data/prepared/manifest.json
2. 根据文档名或 doc_id 选择目标讲义
3. 将对应 queries/*.json 转成 ablation 脚本可直接消费的 JSONL

用法:
    ./.venv/bin/python reproduce/08_build_eval_questions.py \
      --doc "Week 5 Lecture 1 - Operation Amplifier.pdf" \
      --output experiment_data/prepared/eval/week5_opamp.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_manifest(manifest_path: Path) -> list[dict]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_manifest_entry(manifest: list[dict], doc: str) -> dict:
    lowered = doc.strip().lower()
    for entry in manifest:
        if entry["file_name"].lower() == lowered or entry["doc_id"].lower() == lowered:
            return entry
    for entry in manifest:
        if lowered in entry["file_name"].lower() or lowered in entry["title_hint"].lower():
            return entry
    raise FileNotFoundError(f"No manifest entry matched: {doc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation QA JSONL from prepared queries")
    parser.add_argument("--doc", required=True, help="Document file name, title hint, or doc_id")
    parser.add_argument("--manifest", default="experiment_data/prepared/manifest.json")
    parser.add_argument("--queries-dir", default="experiment_data/prepared/queries")
    parser.add_argument("--output", required=True, help="Target JSONL path")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    queries_dir = Path(args.queries_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    entry = resolve_manifest_entry(manifest, args.doc)
    query_path = queries_dir / Path(entry["qa_file"]).name
    query_payload = json.loads(query_path.read_text(encoding="utf-8"))

    lines = []
    for item in query_payload["questions"]:
        lines.append(
            json.dumps(
                {
                    "doc_id": entry["doc_id"],
                    "file_name": entry["file_name"],
                    "group": entry["group"],
                    "category": item["category"],
                    "question": item["question"],
                    "answer": item.get("answer", ""),
                },
                ensure_ascii=False,
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[doc] {entry['file_name']}")
    print(f"[questions] {len(lines)}")
    print(f"[output] {output_path}")


if __name__ == "__main__":
    main()
