#!/usr/bin/env python
"""
Step 2: ModalProcessor 独立测试脚本
分别测试 Image / Table / Equation 三个处理器

用法:
    python reproduce/02_test_modal_processors.py \
        --image path/to/figure.png \
        --api-key sk-xxx

观察点:
    1. VLM 生成的 caption 质量
    2. entity_info 的结构（哪些字段被抽取为 KG 实体）
    3. context_window 开关对 caption 质量的影响（--no-context）
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger
from raganything import RAGAnythingConfig
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    ContextExtractor,
    ContextConfig,
)


def get_env_or_fallback(primary_key: str, fallback_key: str, default: str = "") -> str:
    return os.getenv(primary_key) or os.getenv(fallback_key) or default


def build_rag(
    api_key: str,
    base_url: str,
    working_dir: str,
    llm_model: str,
    vlm_model: str,
    vlm_api_key: str,
    vlm_base_url: str,
    embedding_model: str,
    embedding_api_key: str,
    embedding_base_url: str,
    embedding_dim: int,
):
    """构造轻量 LightRAG 实例，仅用于 ModalProcessor 初始化"""
    def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            llm_model, prompt,
            system_prompt=system_prompt, history_messages=history_messages,
            api_key=api_key, base_url=base_url, **kwargs,
        )

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

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=embed_func,
    )
    return rag, vision_func


# ─────────────────────────────────────────────
# 测试 1: ImageModalProcessor
# ─────────────────────────────────────────────
async def test_image_processor(image_path: str, rag, vision_func, use_context: bool):
    print(f"\n{'='*50}")
    print(f"[ImageModalProcessor] image={image_path}, context={use_context}")
    print('='*50)

    ctx_config = ContextConfig(context_window=1 if use_context else 0)
    ctx_extractor = ContextExtractor(config=ctx_config)

    processor = ImageModalProcessor(
        lightrag=rag,
        modal_caption_func=vision_func,
        context_extractor=ctx_extractor,
    )

    # 模拟 MinerU 输出的 image content
    image_content = {
        "img_path": image_path,
        "img_caption": ["Figure: Example circuit diagram"],
        "img_footnote": [],
    }

    # 模拟 content_list（上下文窗口数据）
    mock_content_list = [
        {"type": "text", "page_idx": 0, "text": "This paper proposes a novel circuit."},
        {"type": "image", "page_idx": 0, "img_path": image_path},
        {"type": "text", "page_idx": 1, "text": "The experimental results show improvement."},
    ]

    description, entity_info = await processor.process_multimodal_content(
        modal_content=image_content,
        content_type="image",
        file_path="test_paper.pdf",
        entity_name="Test Figure",
        content_source=mock_content_list if use_context else None,
        item_info={"page_idx": 0, "index": 1},
    )

    print(f"\n[Description 前 300 字]:\n{description[:300] if description else '(empty)'}")
    print(f"\n[Entity Info]:")
    print(json.dumps(entity_info, ensure_ascii=False, indent=2) if entity_info else "  (empty)")
    return description, entity_info


# ─────────────────────────────────────────────
# 测试 2: TableModalProcessor
# ─────────────────────────────────────────────
async def test_table_processor(rag, vision_func):
    print(f"\n{'='*50}")
    print("[TableModalProcessor]")
    print('='*50)

    processor = TableModalProcessor(
        lightrag=rag,
        modal_caption_func=vision_func,
    )

    # 模拟一个 HTML 表格（MinerU 输出格式）
    table_content = {
        "table_body": """<table>
<tr><th>Method</th><th>Accuracy</th><th>F1</th></tr>
<tr><td>Baseline (LightRAG text-only)</td><td>72.3%</td><td>0.71</td></tr>
<tr><td>RAG-Anything (single graph)</td><td>78.9%</td><td>0.77</td></tr>
<tr><td>RAG-Anything (dual graph)</td><td>85.2%</td><td>0.84</td></tr>
</table>""",
        "table_caption": ["Table 1: Performance comparison on DocBench"],
        "table_footnote": [],
    }

    description, entity_info = await processor.process_multimodal_content(
        modal_content=table_content,
        content_type="table",
        file_path="test_paper.pdf",
        entity_name="Performance Comparison Table",
    )

    print(f"\n[Description]:\n{description[:300] if description else '(empty)'}")
    print(f"\n[Entity Info]:")
    print(json.dumps(entity_info, ensure_ascii=False, indent=2) if entity_info else "  (empty)")
    return description, entity_info


# ─────────────────────────────────────────────
# 测试 3: EquationModalProcessor
# ─────────────────────────────────────────────
async def test_equation_processor(rag, vision_func):
    print(f"\n{'='*50}")
    print("[EquationModalProcessor]")
    print('='*50)

    processor = EquationModalProcessor(
        lightrag=rag,
        modal_caption_func=vision_func,
    )

    equation_content = {
        "latex": r"S(q, K) = \text{softmax}\left(\frac{qK^T}{\sqrt{d}}\right)",
        "equation_caption": ["Equation 3: Cross-modal similarity score"],
    }

    description, entity_info = await processor.process_multimodal_content(
        modal_content=equation_content,
        content_type="equation",
        file_path="test_paper.pdf",
        entity_name="Cross-modal Similarity Equation",
    )

    print(f"\n[Description]:\n{description[:300] if description else '(empty)'}")
    print(f"\n[Entity Info]:")
    print(json.dumps(entity_info, ensure_ascii=False, indent=2) if entity_info else "  (empty)")
    return description, entity_info


async def main_async(args):
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vlm_model = get_env_or_fallback("VLM_MODEL", "LLM_MODEL", llm_model)
    vlm_base_url = get_env_or_fallback("VLM_BINDING_HOST", "LLM_BINDING_HOST", args.base_url)
    vlm_api_key = get_env_or_fallback("VLM_BINDING_API_KEY", "LLM_BINDING_API_KEY", args.api_key)
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_base_url = os.getenv("EMBEDDING_BINDING_HOST", args.base_url)
    embedding_api_key = os.getenv("EMBEDDING_BINDING_API_KEY", args.api_key)
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))

    rag, vision_func = build_rag(
        args.api_key,
        args.base_url,
        args.working_dir,
        llm_model,
        vlm_model,
        vlm_api_key,
        vlm_base_url,
        embedding_model,
        embedding_api_key,
        embedding_base_url,
        embedding_dim,
    )
    await rag.initialize_storages()

    results = {}

    # Image 测试（有上下文 vs 无上下文）
    if args.image:
        desc_ctx, info_ctx = await test_image_processor(
            args.image, rag, vision_func, use_context=True
        )
        desc_no, info_no = await test_image_processor(
            args.image, rag, vision_func, use_context=False
        )
        results["image_with_context"] = {"description": desc_ctx, "entity_info": info_ctx}
        results["image_no_context"] = {"description": desc_no, "entity_info": info_no}

        # 简单对比
        print("\n[context_window 对比]")
        print(f"  有上下文 description 长度: {len(desc_ctx) if desc_ctx else 0}")
        print(f"  无上下文 description 长度: {len(desc_no) if desc_no else 0}")
    else:
        print("\n[跳过 Image 测试] 未提供 --image 参数（可以传任意图片路径）")

    # Table 测试
    if not args.skip_table:
        desc_t, info_t = await test_table_processor(rag, vision_func)
        results["table"] = {"description": desc_t, "entity_info": info_t}

    # Equation 测试
    if not args.skip_equation:
        desc_e, info_e = await test_equation_processor(rag, vision_func)
        results["equation"] = {"description": desc_e, "entity_info": info_e}

    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    out_file = os.path.join(args.output, "modal_processor_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 结果已保存到: {out_file}")

    await rag.finalize_storages()


def main():
    ap = argparse.ArgumentParser(description="ModalProcessor 独立测试")
    ap.add_argument("--image", default=None, help="测试图片路径（可选）")
    ap.add_argument("--working-dir", default="./rag_storage_step2")
    ap.add_argument("--output", "-o", default="./output_step2")
    ap.add_argument("--api-key", default=os.getenv("LLM_BINDING_API_KEY"))
    ap.add_argument("--base-url", default=os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"))
    ap.add_argument("--skip-table", action="store_true")
    ap.add_argument("--skip-equation", action="store_true")
    args = ap.parse_args()

    if not args.api_key:
        print("[ERROR] 请设置 API Key")
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    print("=" * 50)
    print("RAG-Anything Step2: ModalProcessor 独立测试")
    print("=" * 50)
    main()
