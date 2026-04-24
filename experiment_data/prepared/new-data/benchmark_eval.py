#!/usr/bin/env python3
"""
benchmark_eval.py - 对比实验评分脚本
用法:
  python benchmark_eval.py \
    --baseline results_baseline.json \
    --enhanced results_enhanced.json \
    --output report.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


PLACEHOLDER_ANSWERS = {
    "",
    "__TO_BE_FILLED__",
    "__TO_BE_FILLED_BY_EXPERIMENT__",
}

def load_results(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_placeholder_answer(answer: str) -> bool:
    return (answer or "").strip() in PLACEHOLDER_ANSWERS


def rule_based_score(answer: str, correct: str, rubric: Dict) -> float:
    """基于 rubric 的规则评分，适用于 factoid/topology/mapping 类题目"""
    if is_placeholder_answer(answer):
        return 0.0

    answer_lower = answer.lower()
    score = 1.0

    must_include = rubric.get("must_include", [])
    must_not_claim = rubric.get("must_not_claim", [])

    # 检查必须包含的关键点
    if must_include:
        hits = sum(1 for kw in must_include if kw.lower() in answer_lower)
        score = hits / len(must_include)

    # 检查不应声称的内容（任一命中则扣分）
    for bad_claim in must_not_claim:
        if bad_claim.lower() in answer_lower:
            score *= 0.2  # 严重扣分
            break

    return round(score, 3)

def unanswerable_score(answer: str, rubric: Dict) -> float:
    """专门评估 unanswerable 题目"""
    if is_placeholder_answer(answer):
        return 0.0
    answer_lower = answer.lower()
    must_include = rubric.get("must_include", [])
    # 对 unanswerable 题，模型应说"没有/不在文档中"
    refusal_signals = [
        "未提供", "未找到", "未包含", "未提及", "不在", "无此", "没有",
        "无法基于", "无法从", "不能从", "需查", "数据手册",
        "not in", "not found", "not provided", "not mentioned", "not included",
    ]
    has_refusal = any(s in answer_lower for s in refusal_signals)

    hallucination_signals = rubric.get("must_not_claim", [])
    has_hallucination = any(h.lower() in answer_lower for h in hallucination_signals)

    if has_hallucination:
        return 0.0
    if has_refusal:
        return 1.0
    return 0.3  # 部分分

def evaluate(results: List[Dict]) -> Dict:
    scores_by_topic = defaultdict(list)
    scores_by_category = defaultdict(list)
    all_scores = []

    for item in results:
        cat = item.get("category", "unknown")
        topic = item.get("topic", "unknown")
        rubric = item.get("rubric", {})
        answer = item.get("answer", "")
        correct = item.get("correct_answer", "")

        if cat == "unanswerable":
            score = unanswerable_score(answer, rubric)
        else:
            score = rule_based_score(answer, correct, rubric)

        item["auto_score"] = score
        scores_by_topic[topic].append(score)
        scores_by_category[cat].append(score)
        all_scores.append(score)

    report = {
        "overall": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
        "by_topic": {t: round(sum(s)/len(s), 3) for t, s in scores_by_topic.items()},
        "by_category": {c: round(sum(s)/len(s), 3) for c, s in scores_by_category.items()},
        "total_questions": len(all_scores),
    }
    return report


def check_qid_alignment(baseline: List[Dict], enhanced: List[Dict]) -> Dict:
    baseline_qids = {item["qid"] for item in baseline}
    enhanced_qids = {item["qid"] for item in enhanced}
    missing_in_enhanced = sorted(baseline_qids - enhanced_qids)
    missing_in_baseline = sorted(enhanced_qids - baseline_qids)
    shared = sorted(baseline_qids & enhanced_qids)
    return {
        "shared_qids": len(shared),
        "missing_in_enhanced": missing_in_enhanced,
        "missing_in_baseline": missing_in_baseline,
    }


def compare_and_report(baseline: List[Dict], enhanced: List[Dict]) -> Dict:
    b_report = evaluate(baseline)
    e_report = evaluate(enhanced)
    alignment = check_qid_alignment(baseline, enhanced)

    print("=" * 60)
    print("BENCHMARK EVALUATION REPORT")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>10} {'Enhanced':>10} {'Delta':>10}")
    print("-" * 60)
    print(f"{'Overall':<30} {b_report['overall']:>10.3f} {e_report['overall']:>10.3f} {e_report['overall']-b_report['overall']:>+10.3f}")
    print()
    print("--- By Topic ---")
    for topic in sorted(b_report['by_topic'].keys()):
        b = b_report['by_topic'].get(topic, 0)
        e = e_report['by_topic'].get(topic, 0)
        print(f"  {topic:<28} {b:>10.3f} {e:>10.3f} {e-b:>+10.3f}")
    print()
    print("--- By Category ---")
    for cat in sorted(b_report['by_category'].keys()):
        b = b_report['by_category'].get(cat, 0)
        e = e_report['by_category'].get(cat, 0)
        print(f"  {cat:<28} {b:>10.3f} {e:>10.3f} {e-b:>+10.3f}")
    print("=" * 60)
    if alignment["missing_in_enhanced"] or alignment["missing_in_baseline"]:
        print("\n[WARN] baseline / enhanced 的 qid 不完全对齐")
        print(f"  shared_qids: {alignment['shared_qids']}")
        if alignment["missing_in_enhanced"]:
            print(f"  missing_in_enhanced: {alignment['missing_in_enhanced'][:5]}")
        if alignment["missing_in_baseline"]:
            print(f"  missing_in_baseline: {alignment['missing_in_baseline'][:5]}")

    # Win/Tie/Loss per question
    wins, ties, losses = 0, 0, 0
    b_by_qid = {item["qid"]: item for item in baseline}
    e_by_qid = {item["qid"]: item for item in enhanced}
    for qid in b_by_qid:
        bs = b_by_qid[qid].get("auto_score", 0)
        es = e_by_qid.get(qid, {}).get("auto_score", 0)
        if es > bs + 0.1: wins += 1
        elif bs > es + 0.1: losses += 1
        else: ties += 1
    total = wins + ties + losses
    print(f"\nEnhanced vs Baseline per-question: Win {wins}/{total} | Tie {ties}/{total} | Loss {losses}/{total}")
    return {
        "baseline": b_report,
        "enhanced": e_report,
        "alignment": alignment,
        "win_tie_loss": {
            "win": wins,
            "tie": ties,
            "loss": losses,
            "total": total,
        },
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--enhanced", required=True)
    parser.add_argument("--output", help="可选：将评分报告保存为 JSON")
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    enhanced = load_results(args.enhanced)
    report = compare_and_report(baseline, enhanced)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
