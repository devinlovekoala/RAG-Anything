#!/usr/bin/env python
"""
Step 1: 端到端 Pipeline 复现脚本
用法:
    python reproduce/01_run_pipeline.py path/to/paper.pdf [--working-dir ./rag_storage] [--output ./output]

验证点:
    - MinerU/docling 正确解析出文本、图像、表格
    - 知识图谱包含多模态实体节点
    - hybrid 查询能检索到图表中的信息
"""

import os
import sys
import asyncio
import argparse
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path

# 确保能 import 项目根目录的包
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.base import DocStatus
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig


def get_env_or_fallback(primary_key: str, fallback_key: str, default: str = "") -> str:
    return os.getenv(primary_key) or os.getenv(fallback_key) or default


def status_to_dict(status_obj):
    """Normalize LightRAG doc status values that may be dicts or dataclass objects."""
    if isinstance(status_obj, dict):
        return status_obj
    if is_dataclass(status_obj):
        return asdict(status_obj)
    if hasattr(status_obj, "__dict__"):
        return dict(status_obj.__dict__)
    return {"raw_status": repr(status_obj)}


def build_llm_func(api_key: str, base_url: str, model: str = "gpt-4o-mini"):
    """构造 LLM 调用函数"""
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            model, prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    return llm_model_func


def build_vision_func(api_key: str, base_url: str, model: str = "gpt-4o-mini"):
    """构造 VLM 调用函数（支持图像输入）"""
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[],
        image_data=None, messages=None, **kwargs,
    ):
        if messages:
            # 已经格式化好的多模态消息列表（aquery_with_multimodal 使用）
            return openai_complete_if_cache(
                model, "",
                messages=messages,
                api_key=api_key, base_url=base_url, **kwargs,
            )
        elif image_data:
            # 单张图片 base64
            msg_list = []
            if system_prompt:
                msg_list.append({"role": "system", "content": system_prompt})
            msg_list.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ],
            })
            return openai_complete_if_cache(
                model, "",
                messages=msg_list,
                api_key=api_key, base_url=base_url, **kwargs,
            )
        else:
            return openai_complete_if_cache(
                model, prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key, base_url=base_url, **kwargs,
            )
    return vision_model_func


def build_embedding_func(
    api_key: str,
    base_url: str,
    model: str = "text-embedding-3-large",
    dim: int = 3072,
):
    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=8192,
        model_name=model,
        func=lambda texts: openai_embed.func(
            texts,
            model=model,
            api_key=api_key, base_url=base_url,
        ),
    )


async def run_pipeline(
    file_path: str,
    working_dir: str,
    output_dir: str,
    api_key: str,
    base_url: str,
    parser: str = "docling",
    queries: list[str] = None,
    llm_model: str = "gpt-4o-mini",
    vlm_model: str = "gpt-4o-mini",
    vlm_api_key: str = "",
    vlm_base_url: str = "",
    embedding_model: str = "text-embedding-3-large",
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    embedding_dim: int = 3072,
):
    print(f"\n[文档] {file_path}")
    print(f"[存储] working_dir={working_dir}, output={output_dir}")
    print(f"[Parser] {parser}")

    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        enable_circuit_processing=True,
        context_window=1,
        context_mode="page",
    )

    llm_func = build_llm_func(api_key, base_url, llm_model)
    vision_func = build_vision_func(vlm_api_key, vlm_base_url, vlm_model)
    embed_func = build_embedding_func(
        embedding_api_key,
        embedding_base_url,
        embedding_model,
        embedding_dim,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_func,
        vision_model_func=vision_func,
        embedding_func=embed_func,
    )

    # ---- 1. 初始化存储 ----
    await rag.initialize_storages()
    from lightrag.kg.shared_storage import initialize_pipeline_status
    await initialize_pipeline_status()

    # ---- 1.5 先做 embedding 维度自检，避免进到建库阶段才失败 ----
    print("\n[Step 0/3] 验证 Embedding 维度配置...")
    sample_embedding = await embed_func(["RAG-Anything embedding dimension probe"])
    actual_dim = int(sample_embedding.shape[-1])
    if actual_dim != embedding_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch before indexing: env/config expects {embedding_dim}, "
            f"but model '{embedding_model}' returned {actual_dim}. "
            "请检查 reproduce/.env、当前 shell 环境变量，以及 working_dir 是否复用了旧实验目录。"
        )
    print(f"  -> Embedding dim OK: {actual_dim}")

    # ---- 2. 解析并索引文档 ----
    print("\n[Step 1/3] 解析文档并构建知识图谱...")
    await rag.process_document_complete(
        file_path=file_path,
        output_dir=output_dir,
        parse_method="auto",
    )
    print("  -> 文档处理流程结束")

    file_ref = rag._get_file_reference(file_path)
    failed_docs = await rag.lightrag.doc_status.get_docs_by_status(DocStatus.FAILED)
    processed_docs = await rag.lightrag.doc_status.get_docs_by_status(DocStatus.PROCESSED)

    matched_failure = None
    for doc_id, doc_info in failed_docs.items():
        doc_payload = status_to_dict(doc_info)
        if doc_payload.get("file_path") == file_ref:
            matched_failure = {"doc_id": doc_id, **doc_payload}
            break

    matched_processed = next(
        (
            {"doc_id": doc_id, **status_to_dict(doc_info)}
            for doc_id, doc_info in processed_docs.items()
            if status_to_dict(doc_info).get("file_path") == file_ref
        ),
        None,
    )

    duplicate_failure = (
        matched_failure is not None
        and "already exists" in str(matched_failure.get("error_msg", "")).lower()
    )

    if matched_failure is not None and not (duplicate_failure and matched_processed is not None):
        raise RuntimeError(
            "Document processing failed before query stage. "
            f"Matched failed doc status: {matched_failure}"
        )
    if duplicate_failure and matched_processed is not None:
        print(
            "  -> 检测到同文档重复导入记录；当前 working_dir 中已有已处理版本，"
            "将继续使用已存在的索引结果。"
        )

    # ---- 3. 验证 KG 中的多模态实体 ----
    print("\n[Step 2/3] 检查知识图谱中的多模态实体...")
    try:
        kg_data = await rag.lightrag.chunk_entity_relation_graph.get_all_labels()
        print(f"  KG 实体总数: {len(kg_data) if kg_data else 'unknown'}")
    except Exception as e:
        print(f"  KG 实体统计跳过: {e}")

    # ---- 4. 查询测试 ----
    if queries is None:
        queries = [
            "Summarize the main contributions of this paper.",
            "What experimental results are shown in the figures or tables?",
            "What is the architecture of the proposed system?",
        ]

    print(f"\n[Step 3/3] 执行 {len(queries)} 条测试查询...")
    results = {}
    for mode in ["hybrid", "naive"]:
        results[mode] = []
        for q in queries:
            print(f"\n  [{mode}] Q: {q[:60]}...")
            try:
                ans = await rag.aquery(q, mode=mode)
                short = ans[:200].replace("\n", " ") if ans else "(empty)"
                print(f"  A: {short}...")
                results[mode].append({"question": q, "answer": ans})
            except Exception as e:
                print(f"  ERROR: {e}")
                results[mode].append({"question": q, "error": str(e)})

    # ---- 5. 保存结果 ----
    import json
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "query_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 查询结果已保存到: {out_file}")

    await rag.finalize_storages()


def main():
    ap = argparse.ArgumentParser(description="RAG-Anything 端到端复现")
    ap.add_argument("file_path", help="待处理的 PDF/图片文件路径")
    ap.add_argument("--working-dir", "-w", default="./rag_storage_step1")
    ap.add_argument("--output", "-o", default="./output_step1")
    ap.add_argument("--api-key", default=os.getenv("LLM_BINDING_API_KEY"))
    ap.add_argument("--base-url", default=os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"))
    ap.add_argument("--parser", default=os.getenv("PARSER", "docling"),
                    choices=["mineru", "docling", "paddleocr"],
                    help="文档解析器选择")
    ap.add_argument("--query", "-q", action="append", dest="queries",
                    help="自定义查询（可多次使用）")

    args = ap.parse_args()

    if not args.api_key:
        print("[ERROR] 请设置 API Key：--api-key 参数 或在 .env 中设置 LLM_BINDING_API_KEY")
        sys.exit(1)

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vlm_model = get_env_or_fallback("VLM_MODEL", "LLM_MODEL", llm_model)
    vlm_base_url = get_env_or_fallback(
        "VLM_BINDING_HOST", "LLM_BINDING_HOST", args.base_url
    )
    vlm_api_key = get_env_or_fallback(
        "VLM_BINDING_API_KEY", "LLM_BINDING_API_KEY", args.api_key
    )
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_base_url = os.getenv("EMBEDDING_BINDING_HOST", args.base_url)
    embedding_api_key = os.getenv("EMBEDDING_BINDING_API_KEY", args.api_key)
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))

    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")
    logger.setLevel(logging.INFO)

    asyncio.run(run_pipeline(
        file_path=args.file_path,
        working_dir=args.working_dir,
        output_dir=args.output,
        api_key=args.api_key,
        base_url=args.base_url,
        parser=args.parser,
        queries=args.queries,
        llm_model=llm_model,
        vlm_model=vlm_model,
        vlm_api_key=vlm_api_key,
        vlm_base_url=vlm_base_url,
        embedding_model=embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_dim=embedding_dim,
    ))


if __name__ == "__main__":
    print("=" * 50)
    print("RAG-Anything Step1: 端到端 Pipeline 复现")
    print("=" * 50)
    main()
