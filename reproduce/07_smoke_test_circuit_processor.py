#!/usr/bin/env python3
"""
Step 8: CircuitModalProcessor 轻量验证

不依赖 MinerU/Docling，直接把一张图片送入 CircuitModalProcessor，
验证电路检测、VLM 结构化输出解析、SPICE-like netlist 生成。

用法:
    ./.venv/bin/python -u reproduce/07_smoke_test_circuit_processor.py \
      --image "output_smoke/Week 5 Lecture 1 - Operation Amplifier-1.png"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache
from raganything.modalprocessors import CircuitModalProcessor, ContextConfig, ContextExtractor


load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)


def env_or_fallback(primary_key: str, fallback_key: str, default: str = "") -> str:
    return os.getenv(primary_key) or os.getenv(fallback_key) or default


def build_vision_func():
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vlm_model = env_or_fallback("VLM_MODEL", "LLM_MODEL", llm_model)
    vlm_base_url = env_or_fallback(
        "VLM_BINDING_HOST", "LLM_BINDING_HOST", "https://api.openai.com/v1"
    )
    vlm_api_key = env_or_fallback("VLM_BINDING_API_KEY", "LLM_BINDING_API_KEY", "")

    async def vision_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        if messages:
            return await openai_complete_if_cache(
                vlm_model,
                "",
                messages=messages,
                api_key=vlm_api_key,
                base_url=vlm_base_url,
                **kwargs,
            )
        if image_data:
            msg_list = []
            if system_prompt:
                msg_list.append({"role": "system", "content": system_prompt})
            msg_list.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            },
                        },
                    ],
                }
            )
            return await openai_complete_if_cache(
                vlm_model,
                "",
                messages=msg_list,
                api_key=vlm_api_key,
                base_url=vlm_base_url,
                **kwargs,
            )
        return await openai_complete_if_cache(
            vlm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=vlm_api_key,
            base_url=vlm_base_url,
            **kwargs,
        )

    return vision_func


@dataclass
class DummyLightRAG:
    """Minimal object needed by BaseModalProcessor initialization."""

    text_chunks: Any = None
    chunks_vdb: Any = None
    entities_vdb: Any = None
    relationships_vdb: Any = None
    chunk_entity_relation_graph: Any = None
    embedding_func: Any = None
    llm_model_func: Any = None
    llm_response_cache: Any = None
    tokenizer: Any = field(default=None)

    def __post_init__(self):
        class Tokenizer:
            @staticmethod
            def encode(text):
                return str(text).split()

        if self.tokenizer is None:
            self.tokenizer = Tokenizer()


async def main_async(args: argparse.Namespace) -> None:
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    processor = CircuitModalProcessor(
        lightrag=DummyLightRAG(),
        modal_caption_func=build_vision_func(),
        context_extractor=ContextExtractor(
            config=ContextConfig(context_window=0, max_context_tokens=1200)
        ),
    )

    modal_content = {
        "type": "image",
        "img_path": str(image_path),
        "img_caption": [
            args.caption
            or "Electronic circuits course page, possible operational amplifier diagram"
        ],
        "img_footnote": [],
    }

    description, entity_info = await processor.generate_description_only(
        modal_content,
        content_type="image",
        item_info={"page_idx": 0, "index": 0, "type": "image"},
        entity_name=args.entity_name,
    )

    print("[description]")
    print(description[:1200])
    print("\n[entity_info]")
    selected = {
        "entity_name": entity_info.get("entity_name"),
        "entity_type": entity_info.get("entity_type"),
        "chunk_type": entity_info.get("chunk_type"),
        "circuit_detected": entity_info.get("circuit_detected"),
        "summary": entity_info.get("summary"),
        "component_count": len(entity_info.get("circuit_components") or []),
        "connection_count": len(entity_info.get("circuit_connections") or []),
        "circuit_components": entity_info.get("circuit_components") or [],
        "circuit_connections": entity_info.get("circuit_connections") or [],
        "circuit_netlist": entity_info.get("circuit_netlist", "")[:2000],
    }
    print(json.dumps(selected, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test CircuitModalProcessor")
    parser.add_argument("--image", required=True)
    parser.add_argument("--caption", default="")
    parser.add_argument("--entity-name", default="Smoke Test Circuit Page")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
