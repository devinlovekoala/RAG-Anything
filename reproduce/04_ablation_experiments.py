#!/usr/bin/env python
"""
Step 5: 对比消融实验脚本（E1-E4）

实验矩阵（与学习计划 4.1 节对应）：
  E1: RAG-Anything(hybrid) vs 纯文本 LightRAG(naive)   — 评估多模态信息的检索增益
  E2: Dual-Graph(hybrid) vs Single-Graph(mix)           — 验证双图策略的必要性
  E3: VLM开/关 对 caption 质量的影响                    — 通过 entity_info 丰富度评估
  E4: context_window=0/1/2 对结果的影响                 — 上下文窗口大小消融

前提：
  - 已用 01_run_pipeline.py 建好索引（提供 --working-dir）
  - 查询问题来自 --qa-file（JSONL，每行 {"question": "...", "answer": "..."}）
    若不提供则用内置默认问题集

用法:
    python reproduce/04_ablation_experiments.py \
        --working-dir ./rag_storage_step1 \
        --qa-file path/to/qa.jsonl \
        --api-key sk-xxx \
        --experiments E1 E2 E4
"""

import os
import sys
import json
import asyncio
import argparse
import time
import re
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

import json_repair
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger
from raganything import RAGAnything, RAGAnythingConfig


# ─────────────────────────────────────────────
# 默认测试问题集（无 QA 文件时使用）
# ─────────────────────────────────────────────
DEFAULT_QUESTIONS = [
    "What are the main contributions of this work?",
    "What experimental results are shown in the figures?",
    "What values are reported in the performance comparison table?",
    "Describe the system architecture.",
    "What datasets were used for evaluation?",
]


def get_env_or_fallback(primary_key: str, fallback_key: str, default: str = "") -> str:
    return os.getenv(primary_key) or os.getenv(fallback_key) or default


def _extract_keyword_list(payload: object, canonical_key: str, fallback_patterns: List[str]) -> List[str]:
    if not isinstance(payload, dict):
        return []

    candidate_keys = [canonical_key]
    for key in payload.keys():
        lowered = key.lower()
        if any(pattern in lowered for pattern in fallback_patterns):
            candidate_keys.append(key)

    seen = set()
    values: List[str] = []
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        raw_value = payload.get(key, [])
        if isinstance(raw_value, list):
            values.extend(str(item).strip() for item in raw_value if str(item).strip())
        elif isinstance(raw_value, str):
            values.extend(
                token.strip()
                for token in re.split(r"[,;\n]", raw_value)
                if token.strip()
            )
    return values


def _normalize_keyword_extraction_response(raw_text: str) -> str:
    try:
        payload = json_repair.loads(raw_text)
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    normalized = {
        "high_level_keywords": _extract_keyword_list(
            payload, "high_level_keywords", ["high_level_keyword"]
        ),
        "low_level_keywords": _extract_keyword_list(
            payload, "low_level_keywords", ["low_level_keyword"]
        ),
    }
    return json.dumps(normalized, ensure_ascii=False)


def build_funcs(api_key: str, base_url: str):
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vlm_model = get_env_or_fallback("VLM_MODEL", "LLM_MODEL", llm_model)
    vlm_base_url = get_env_or_fallback("VLM_BINDING_HOST", "LLM_BINDING_HOST", base_url)
    vlm_api_key = get_env_or_fallback("VLM_BINDING_API_KEY", "LLM_BINDING_API_KEY", api_key)
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_base_url = os.getenv("EMBEDDING_BINDING_HOST", base_url)
    embedding_api_key = os.getenv("EMBEDDING_BINDING_API_KEY", api_key)
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        try:
            return await openai_complete_if_cache(
                llm_model, prompt,
                system_prompt=system_prompt, history_messages=history_messages,
                api_key=api_key, base_url=base_url, **kwargs,
            )
        except Exception:
            if not kwargs.get("keyword_extraction"):
                raise

            logger.warning(
                "Structured keyword extraction failed for model %s; retrying with tolerant JSON fallback.",
                llm_model,
            )
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("keyword_extraction", None)
            fallback_kwargs.pop("response_format", None)
            fallback_prompt = (
                f"{prompt}\n\n"
                "Return valid JSON only with this exact schema:\n"
                "{\"high_level_keywords\": [\"...\"], \"low_level_keywords\": [\"...\"]}"
            )
            raw_result = await openai_complete_if_cache(
                llm_model,
                fallback_prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **fallback_kwargs,
            )
            return _normalize_keyword_extraction_response(raw_result)

    def vision_func(prompt, system_prompt=None, history_messages=[],
                    image_data=None, messages=None, **kwargs):
        if messages:
            return openai_complete_if_cache(
                vlm_model, "",
                messages=messages, api_key=vlm_api_key, base_url=vlm_base_url, **kwargs,
            )
        elif image_data:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ],
            })
            return openai_complete_if_cache(
                vlm_model, "",
                messages=msgs, api_key=vlm_api_key, base_url=vlm_base_url, **kwargs,
            )
        else:
            return llm_func(prompt, system_prompt, history_messages, **kwargs)

    embed_func = EmbeddingFunc(
        embedding_dim=embedding_dim, max_token_size=8192,
        model_name=embedding_model,
        func=lambda texts: openai_embed.func(
            texts, model=embedding_model,
            api_key=embedding_api_key, base_url=embedding_base_url,
        ),
    )
    return llm_func, vision_func, embed_func


async def query_with_timing(rag, question: str, mode: str) -> Dict:
    t0 = time.perf_counter()
    try:
        answer = await rag.aquery(question, mode=mode)
        latency = time.perf_counter() - t0
        return {"answer": answer, "latency_s": round(latency, 3), "error": None}
    except Exception as e:
        latency = time.perf_counter() - t0
        return {"answer": None, "latency_s": round(latency, 3), "error": str(e)}


def add_condition_stats(results: Dict) -> Dict:
    """Append lightweight latency/length stats for each experiment condition."""
    for cond_name, items in list(results.get("conditions", {}).items()):
        if cond_name.endswith("_stats"):
            continue
        valid = [i for i in items if i["error"] is None]
        avg_latency = sum(i["latency_s"] for i in valid) / len(valid) if valid else 0
        avg_ans_len = (
            sum(len(i["answer"] or "") for i in valid) / len(valid) if valid else 0
        )
        results["conditions"][cond_name + "_stats"] = {
            "avg_latency_s": round(avg_latency, 3),
            "avg_answer_length": round(avg_ans_len, 1),
            "success_rate": f"{len(valid)}/{len(items)}",
        }
    return results


async def run_e1_multimodal_vs_textonly(rag, questions: List[str]) -> Dict:
    """E1: hybrid(多模态) vs naive(纯文本向量)"""
    print("\n[E1] RAG-Anything hybrid vs naive (text-only vector search)")
    results = {"experiment": "E1", "conditions": {}}

    for mode in ["hybrid", "naive"]:
        print(f"  Running mode={mode}...")
        cond_results = []
        for q in questions:
            r = await query_with_timing(rag, q, mode)
            cond_results.append({"question": q, **r})
            print(f"    Q: {q[:50]}... | latency={r['latency_s']}s | ans_len={len(r['answer'] or '')}")
        results["conditions"][mode] = cond_results

    return add_condition_stats(results)


async def run_e2_dual_vs_single(rag, questions: List[str]) -> Dict:
    """E2: Dual-Graph (hybrid) vs 近似 Single-Graph (mix)"""
    print("\n[E2] Dual-Graph(hybrid) vs Single-Graph approx(mix)")
    results = {"experiment": "E2", "conditions": {}}

    for mode in ["hybrid", "mix"]:
        print(f"  Running mode={mode}...")
        cond_results = []
        for q in questions:
            r = await query_with_timing(rag, q, mode)
            cond_results.append({"question": q, **r})
            print(f"    Q: {q[:50]}... | latency={r['latency_s']}s")
        results["conditions"][mode] = cond_results

    return add_condition_stats(results)


async def run_e4_context_window(
    api_key: str, base_url: str,
    working_dir: str, questions: List[str],
    file_path: str = None,
) -> Dict:
    """E4: context_window=0/1/2 对结果的影响"""
    print("\n[E4] Context window size ablation (0 / 1 / 2)")
    results = {"experiment": "E4", "conditions": {}}

    for win in [0, 1, 2]:
        print(f"  context_window={win}...")
        cfg = RAGAnythingConfig(
            working_dir=f"{working_dir}_ctx{win}",
            parser=os.getenv("PARSER", "docling"),
            context_window=win,
            enable_circuit_processing=True,
        )
        llm_func, vision_func, embed_func = build_funcs(api_key, base_url)
        rag_local = RAGAnything(
            config=cfg,
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=embed_func,
        )
        await rag_local.initialize_storages()
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()

        if file_path:
            print(f"    Re-indexing with context_window={win}...")
            await rag_local.process_document_complete(
                file_path=file_path,
                output_dir=f"./output_ctx{win}",
            )

        cond_results = []
        for q in questions[:3]:  # E4 只跑前3条节省费用
            r = await query_with_timing(rag_local, q, "hybrid")
            cond_results.append({"question": q, **r})
        results["conditions"][f"window_{win}"] = cond_results
        await rag_local.finalize_storages()

    return add_condition_stats(results)


async def main_async(args):
    questions = DEFAULT_QUESTIONS
    if args.qa_file and os.path.exists(args.qa_file):
        questions = []
        with open(args.qa_file) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    questions.append(d["question"])
        print(f"[QA] 从文件加载 {len(questions)} 条问题")
    else:
        print(f"[QA] 使用默认 {len(questions)} 条问题")

    # 构造主 RAG 实例（用于 E1, E2）
    llm_func, vision_func, embed_func = build_funcs(args.api_key, args.base_url)
    config = RAGAnythingConfig(
        working_dir=args.working_dir,
        parser=os.getenv("PARSER", "docling"),
        enable_circuit_processing=True,
    )
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_func,
        vision_model_func=vision_func,
        embedding_func=embed_func,
    )
    await rag.initialize_storages()

    all_results = {}

    for exp in args.experiments:
        if exp == "E1":
            r = await run_e1_multimodal_vs_textonly(rag, questions)
            all_results["E1"] = r
        elif exp == "E2":
            r = await run_e2_dual_vs_single(rag, questions)
            all_results["E2"] = r
        elif exp == "E4":
            r = await run_e4_context_window(
                args.api_key, args.base_url,
                args.working_dir, questions,
                file_path=args.file_path,
            )
            all_results["E4"] = r
        else:
            print(f"[跳过] 实验 {exp} 尚未实现（E3/E5 需要额外 VLM 配置）")

    await rag.finalize_storages()

    # ─── 打印摘要 ───
    print("\n" + "=" * 60)
    print("实验结果摘要")
    print("=" * 60)
    for exp_name, exp_data in all_results.items():
        print(f"\n[{exp_name}]")
        conds = exp_data.get("conditions", {})
        for cond_name, cond_data in conds.items():
            if cond_name.endswith("_stats"):
                print(f"  {cond_name}: {cond_data}")

    # ─── 保存完整结果 ───
    os.makedirs(args.output, exist_ok=True)
    out_file = os.path.join(args.output, "ablation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 完整结果已保存到: {out_file}")


# args 需要在 build_funcs 中访问，临时放全局
args = None


def main():
    global args
    ap = argparse.ArgumentParser(description="RAG-Anything 消融对比实验")
    ap.add_argument("--working-dir", "-w", default="./rag_storage_step1",
                    help="已建好索引的 working_dir")
    ap.add_argument("--output", "-o", default="./output_ablation")
    ap.add_argument("--api-key", default=os.getenv("LLM_BINDING_API_KEY"))
    ap.add_argument("--base-url", default=os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"))
    ap.add_argument("--qa-file", default=None, help="JSONL 格式查询文件（每行 {question, answer}）")
    ap.add_argument("--file-path", default=None, help="E4 实验需要重建索引时的 PDF 路径")
    ap.add_argument("--experiments", nargs="+", default=["E1", "E2"],
                    choices=["E1", "E2", "E4"],
                    help="要运行的实验（默认: E1 E2）")
    args = ap.parse_args()

    if not args.api_key:
        print("[ERROR] 请设置 API Key")
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    print("=" * 50)
    print("RAG-Anything Step5: 消融对比实验")
    print("=" * 50)
    main()
