#!/usr/bin/env python3
"""
Step 7: 轻量模型栈 Smoke Test

不依赖 MinerU/Docling，直接验证：
1. LLM 文本接口是否可用
2. Embedding 接口是否可用且维度符合 EMBEDDING_DIM
3. VLM 是否能读取从 PDF 第一页提取出的图片

用法:
    ./.venv/bin/python reproduce/06_smoke_test_model_stack.py \
      --pdf "experiment_data/Week 5 Lecture 1 - Operation Amplifier.pdf"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache, openai_embed


load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)


def env_or_fallback(primary_key: str, fallback_key: str, default: str = "") -> str:
    return os.getenv(primary_key) or os.getenv(fallback_key) or default


def read_config() -> dict[str, str | int]:
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    return {
        "llm_model": llm_model,
        "llm_api_key": os.getenv("LLM_BINDING_API_KEY", ""),
        "llm_base_url": os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        "vlm_model": env_or_fallback("VLM_MODEL", "LLM_MODEL", llm_model),
        "vlm_api_key": env_or_fallback("VLM_BINDING_API_KEY", "LLM_BINDING_API_KEY", ""),
        "vlm_base_url": env_or_fallback(
            "VLM_BINDING_HOST", "LLM_BINDING_HOST", "https://api.openai.com/v1"
        ),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        "embedding_api_key": os.getenv("EMBEDDING_BINDING_API_KEY", ""),
        "embedding_base_url": os.getenv(
            "EMBEDDING_BINDING_HOST", "https://api.openai.com/v1"
        ),
        "embedding_dim": int(os.getenv("EMBEDDING_DIM", "3072")),
    }


def extract_first_page(pdf_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / pdf_path.stem
    output_image = Path(f"{output_prefix}-1.png")
    if output_image.exists():
        return output_image

    command = [
        "pdftoppm",
        "-png",
        "-f",
        "1",
        "-singlefile",
        "-r",
        "120",
        str(pdf_path),
        str(output_prefix),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    generated = Path(f"{output_prefix}.png")
    if generated.exists():
        generated.rename(output_image)
    if not output_image.exists():
        raise FileNotFoundError(f"pdftoppm did not create expected image: {output_image}")
    return output_image


def image_to_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


async def test_llm(config: dict[str, str | int]) -> str:
    answer = await openai_complete_if_cache(
        str(config["llm_model"]),
        "Reply with exactly: OK",
        api_key=str(config["llm_api_key"]),
        base_url=str(config["llm_base_url"]),
        timeout=60,
    )
    return answer.strip()


async def test_embedding(config: dict[str, str | int]) -> tuple[int, str]:
    vectors = await openai_embed.func(
        ["RAG-Anything circuit retrieval smoke test"],
        model=str(config["embedding_model"]),
        api_key=str(config["embedding_api_key"]),
        base_url=str(config["embedding_base_url"]),
    )
    vector_dim = int(vectors.shape[-1])
    expected_dim = int(config["embedding_dim"])
    status = "OK" if vector_dim == expected_dim else f"DIM_MISMATCH expected={expected_dim}"
    return vector_dim, status


async def test_vlm(config: dict[str, str | int], image_path: Path) -> str:
    image_data = image_to_base64(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "This is the first page of an electronics course PDF. "
                        "Reply in one short sentence whether it appears related to circuits."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
            ],
        }
    ]
    answer = await openai_complete_if_cache(
        str(config["vlm_model"]),
        "",
        messages=messages,
        api_key=str(config["vlm_api_key"]),
        base_url=str(config["vlm_base_url"]),
        timeout=120,
    )
    return answer.strip()


async def main_async(args: argparse.Namespace) -> None:
    config = read_config()
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    print("[config]")
    print(f"  LLM_MODEL={config['llm_model']}")
    print(f"  VLM_MODEL={config['vlm_model'] or '(fallback to LLM_MODEL)'}")
    print(f"  EMBEDDING_MODEL={config['embedding_model']}")
    print(f"  EMBEDDING_DIM={config['embedding_dim']}")

    print("\n[1/4] Extract first PDF page...")
    image_path = extract_first_page(pdf_path, Path(args.output_dir))
    print(f"  image={image_path}")

    print("\n[2/4] Test LLM...")
    llm_answer = await test_llm(config)
    print(f"  LLM answer={llm_answer}")

    print("\n[3/4] Test Embedding...")
    vector_dim, embedding_status = await test_embedding(config)
    print(f"  embedding_dim={vector_dim} status={embedding_status}")

    print("\n[4/4] Test VLM...")
    vlm_answer = await test_vlm(config, image_path)
    print(f"  VLM answer={vlm_answer}")

    print("\n[done] Model stack smoke test completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test LLM/VLM/Embedding config")
    parser.add_argument(
        "--pdf",
        default="experiment_data/Week 5 Lecture 1 - Operation Amplifier.pdf",
    )
    parser.add_argument("--output-dir", default="output_smoke")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
