#!/usr/bin/env python3
"""
Step 4: CircuitModalProcessor 定向验证入口

直接复用仓库当前内置的 CircuitModalProcessor，验证：
1. 电路页/电路图片是否被正确识别为 circuit image
2. VLM 输出是否被解析为结构化 CircuitDesign
3. 元件、连接关系和 SPICE-like netlist 是否生成成功

用法:
    ./.venv/bin/python -u reproduce/03_circuit_processor.py \
      --image output_smoke/pages/opamp-14.png \
      --caption "Simplified terminals of operational amplifier..." \
      --entity-name "Op Amp Simplified Terminals"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache
from raganything.modalprocessors import CircuitModalProcessor, ContextConfig, ContextExtractor


load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)


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
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
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
    """Minimal object required by BaseModalProcessor initialization."""

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

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
            args.caption or "Electronic circuits course page with potential schematic."
        ],
        "img_footnote": [],
    }

    description, entity_info = await processor.generate_description_only(
        modal_content=modal_content,
        content_type="image",
        entity_name=args.entity_name,
        item_info={"page_idx": args.page_idx, "index": 0, "type": "image"},
    )

    output_payload = {
        "description": description,
        "entity_info": entity_info,
        "summary": {
            "entity_name": entity_info.get("entity_name"),
            "entity_type": entity_info.get("entity_type"),
            "chunk_type": entity_info.get("chunk_type"),
            "circuit_detected": entity_info.get("circuit_detected"),
            "circuit_component_count": len(entity_info.get("circuit_components") or []),
            "circuit_connection_count": len(entity_info.get("circuit_connections") or []),
            "circuit_components": entity_info.get("circuit_components") or [],
            "circuit_connections": entity_info.get("circuit_connections") or [],
            "circuit_netlist": entity_info.get("circuit_netlist", "")[:2000],
        },
    }

    print("[description]")
    print((description or "")[:1200])
    print("\n[summary]")
    print(json.dumps(output_payload["summary"], ensure_ascii=False, indent=2))

    result_path = output_dir / "circuit_processor_result.json"
    result_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[result] {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the built-in CircuitModalProcessor")
    parser.add_argument("--image", required=True, help="Path to a circuit image/page PNG")
    parser.add_argument("--caption", default="", help="Optional image caption hint")
    parser.add_argument("--entity-name", default="Circuit Page")
    parser.add_argument("--source-file", default="manual_circuit_input.pdf")
    parser.add_argument("--page-idx", type=int, default=0)
    parser.add_argument("--output", "-o", default="./output_step4")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    print("=" * 50)
    print("RAG-Anything Step4: Built-in CircuitModalProcessor")
    print("=" * 50)
    main()
