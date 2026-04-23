"""
CircuitParser: PDF document → CircuitIR

Drives document parsing (MinerU / Docling via RAG-Anything's existing parser
infrastructure), then calls the VLM on every extracted image to decide whether
it is a circuit diagram and, if so, to extract structured circuit information.

The result is a fully-populated CircuitIR object ready for CircuitEnhancer
and CircuitAdapter downstream.

Usage
-----
    parser = CircuitParser(modal_caption_func=my_vlm)
    ir = await parser.parse("lecture.pdf")
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from raganything.circuit import (
    CircuitDesign,
    CircuitDetector,
    CircuitNetlistParser,
    CircuitSpiceConverter,
)

from .ir import CircuitFigure, CircuitIR
from .prompts import (
    CIRCUIT_ANALYSIS_SYSTEM,
    CIRCUIT_VISION_PROMPT,
    CIRCUIT_VISION_PROMPT_WITH_CONTEXT,
)

logger = logging.getLogger(__name__)


class CircuitParser:
    """Parse a PDF/DOCX document into a CircuitIR.

    Parameters
    ----------
    modal_caption_func:
        Async callable ``(prompt, image_data=..., system_prompt=...) -> str``.
        Typically ``rag.vision_model_func or rag.llm_model_func``.
    detector_threshold:
        Heuristic score above which an image block is sent to the VLM for
        circuit extraction.  Default 2.0 matches CircuitDetector.is_likely_circuit.
    """

    def __init__(
        self,
        modal_caption_func: Callable,
        detector_threshold: float = 2.0,
    ) -> None:
        self.modal_caption_func = modal_caption_func
        self.detector_threshold = detector_threshold
        self._netlist_parser = CircuitNetlistParser()
        self._spice_converter = CircuitSpiceConverter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse(
        self,
        file_path: str,
        *,
        parse_method: str = "auto",
        lang: str = "en",
        output_dir: Optional[str] = None,
    ) -> CircuitIR:
        """Parse *file_path* and return a populated CircuitIR.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the PDF / DOCX to process.
        parse_method:
            Forwarded to the MinerU/Docling parser backend.  "auto" lets
            RAG-Anything pick the best method for the file type.
        lang:
            Document language hint for the OCR engine.
        output_dir:
            Where to write extracted images.  Defaults to a sibling
            ``<file_stem>_images/`` directory next to the source file.
        """
        file_path = str(Path(file_path).resolve())
        ir = CircuitIR(source_path=file_path)

        content_list = await self._parse_document(
            file_path, parse_method=parse_method, lang=lang, output_dir=output_dir
        )
        if not content_list:
            logger.warning("Document parser returned empty content_list for %s", file_path)
            return ir

        self._distribute_blocks(ir, content_list)
        await self._extract_circuit_figures(ir, content_list)
        ir.circuit_figure_count = len(ir.circuit_figures)
        ir.non_circuit_image_count = len(ir.image_blocks)
        return ir

    # ------------------------------------------------------------------
    # Document parsing (delegates to RAG-Anything's existing infrastructure)
    # ------------------------------------------------------------------

    async def _parse_document(
        self,
        file_path: str,
        parse_method: str,
        lang: str,
        output_dir: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Run MinerU / Docling and return the raw content_list."""
        try:
            from raganything.parser import get_parser
        except ImportError:
            logger.error("raganything.parser not importable — is RAG-Anything installed?")
            return []

        parser_kwargs: Dict[str, Any] = {"lang": lang}
        if parse_method != "auto":
            parser_kwargs["parse_method"] = parse_method
        if output_dir:
            parser_kwargs["output_dir"] = output_dir

        try:
            parser = get_parser("mineru")
            content_list, _ = await parser.parse(file_path, **parser_kwargs)
            if content_list:
                return content_list
        except Exception as exc:
            logger.debug("MinerU failed (%s), trying docling", exc)

        try:
            parser = get_parser("docling")
            content_list, _ = await parser.parse(file_path, **parser_kwargs)
            return content_list or []
        except Exception as exc:
            logger.error("Both parsers failed for %s: %s", file_path, exc)
            return []

    # ------------------------------------------------------------------
    # Block distribution
    # ------------------------------------------------------------------

    def _distribute_blocks(
        self, ir: CircuitIR, content_list: List[Dict[str, Any]]
    ) -> None:
        """Sort raw content blocks into the appropriate CircuitIR bucket."""
        page_set: set = set()
        for item in content_list:
            ctype = item.get("type", "")
            page_idx = item.get("page_idx", 0)
            page_set.add(page_idx)

            if ctype == "text":
                ir.text_blocks.append(item)
            elif ctype == "table":
                ir.table_blocks.append(item)
            elif ctype in ("equation", "interline_equation"):
                ir.equation_blocks.append(item)
            elif ctype == "image":
                # Initially staged as non-circuit; _extract_circuit_figures
                # will move confirmed circuits to ir.circuit_figures.
                ir.image_blocks.append(item)

        ir.total_pages = len(page_set)

    # ------------------------------------------------------------------
    # Circuit figure extraction
    # ------------------------------------------------------------------

    async def _extract_circuit_figures(
        self, ir: CircuitIR, content_list: List[Dict[str, Any]]
    ) -> None:
        """Iterate over image blocks; run VLM on likely circuits."""
        remaining_images: List[Dict[str, Any]] = []

        for item in ir.image_blocks:
            context = self._build_context(item, content_list)
            score = CircuitDetector.score_item(item, context)

            if score < self.detector_threshold:
                remaining_images.append(item)
                continue

            img_path = item.get("img_path", "")
            if not img_path or not Path(img_path).exists():
                logger.debug("Image path missing or not found: %s", img_path)
                remaining_images.append(item)
                continue

            figure = await self._extract_one_figure(item, context)
            if figure is not None and figure.is_circuit:
                ir.circuit_figures.append(figure)
            else:
                remaining_images.append(item)

        ir.image_blocks = remaining_images

    async def _extract_one_figure(
        self, item: Dict[str, Any], context: str
    ) -> Optional[CircuitFigure]:
        """Call the VLM for a single image and parse the response."""
        img_path: str = item.get("img_path", "")
        captions: List[str] = item.get("image_caption", item.get("img_caption", []))
        footnotes: List[str] = item.get("image_footnote", item.get("img_footnote", []))
        page_idx: int = item.get("page_idx", 0)
        entity_name = self._derive_entity_name(img_path, captions)

        prompt_template = (
            CIRCUIT_VISION_PROMPT_WITH_CONTEXT if context else CIRCUIT_VISION_PROMPT
        )
        vision_prompt = prompt_template.format(
            context=context,
            entity_name=entity_name,
            image_path=img_path,
            captions=captions if captions else "None",
            footnotes=footnotes if footnotes else "None",
        )

        image_base64 = _encode_image(img_path)
        if not image_base64:
            logger.warning("Cannot encode image: %s", img_path)
            return None

        try:
            response = await self.modal_caption_func(
                vision_prompt,
                image_data=image_base64,
                system_prompt=CIRCUIT_ANALYSIS_SYSTEM,
            )
        except Exception as exc:
            logger.error("VLM call failed for %s: %s", img_path, exc)
            return None

        return self._parse_vlm_response(
            response,
            img_path=img_path,
            page_idx=page_idx,
            captions=captions,
            footnotes=footnotes,
            entity_name=entity_name,
        )

    def _parse_vlm_response(
        self,
        response: str,
        *,
        img_path: str,
        page_idx: int,
        captions: List[str],
        footnotes: List[str],
        entity_name: str,
    ) -> CircuitFigure:
        """Parse a VLM JSON response into a CircuitFigure."""
        cleaned = _strip_thinking_tags(response)
        data = _robust_json_parse(cleaned)

        description: str = (
            data.get("detailed_description") or data.get("description") or cleaned
        )
        entity_info: Dict[str, Any] = data.get("entity_info", {})
        circuit_payload = (
            data.get("circuit_design")
            or data.get("circuit_json")
            or data.get("circuit")
        )

        circuit_design: Optional[CircuitDesign] = (
            self._netlist_parser.parse_structured_payload(circuit_payload)
        )
        if circuit_design is None:
            netlist_text = data.get("netlist") or data.get("spice_netlist")
            if isinstance(netlist_text, str) and netlist_text.strip():
                circuit_design = self._netlist_parser.parse(netlist_text)
        if circuit_design is None:
            circuit_design = self._netlist_parser.parse(cleaned)

        figure = CircuitFigure(
            page_idx=page_idx,
            img_path=img_path,
            caption=captions,
            footnote=footnotes,
        )

        if circuit_design is None:
            figure.is_circuit = False
            figure.summary = description[:300]
            return figure

        resolved_name = (
            entity_info.get("entity_name")
            or circuit_design.metadata.title
            or entity_name
        )
        if not circuit_design.metadata.title:
            circuit_design.metadata.title = resolved_name
        if not circuit_design.metadata.description:
            circuit_design.metadata.description = description

        circuit_type = circuit_design.metadata.circuit_type or ""
        if not circuit_type and isinstance(circuit_payload, dict):
            circuit_type = circuit_payload.get("circuit_type", "")

        netlist = self._spice_converter.generate_netlist(circuit_design)

        figure.is_circuit = True
        figure.detection_score = CircuitDetector.score_item(
            {
                "img_path": img_path,
                "image_caption": captions,
                "image_footnote": footnotes,
            }
        )
        figure.circuit_design = circuit_design.to_dict()
        figure.circuit_type = circuit_type
        figure.netlist = netlist
        figure.components = circuit_design.component_lines()
        figure.connections = circuit_design.connection_lines()
        figure.summary = (
            entity_info.get("summary")
            or circuit_design.summarize()
            or description[:200]
        )
        return figure

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_context(
        self, image_item: Dict[str, Any], content_list: List[Dict[str, Any]]
    ) -> str:
        """Collect nearby text blocks as context for the VLM prompt."""
        page = image_item.get("page_idx", 0)
        parts: List[str] = []
        for item in content_list:
            if item.get("type") != "text":
                continue
            if abs(item.get("page_idx", 0) - page) <= 1:
                text = item.get("text", "").strip()
                if text:
                    parts.append(text)
        return " ".join(parts)[:2000]  # cap to avoid bloating the prompt

    @staticmethod
    def _derive_entity_name(img_path: str, captions: List[str]) -> str:
        if captions:
            return captions[0][:80].strip()
        return Path(img_path).stem if img_path else "circuit"


# ---------------------------------------------------------------------------
# Module-level helpers (no instance state needed)
# ---------------------------------------------------------------------------

def _encode_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    except Exception as exc:
        logger.error("Failed to encode image %s: %s", image_path, exc)
        return ""


def _strip_thinking_tags(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(
        r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    return cleaned.strip()


def _robust_json_parse(response: str) -> dict:
    """Try several strategies to extract a JSON object from the response."""
    # Strategy 1: find all {...} blocks and try each (shortest to longest)
    candidates = list(re.finditer(r"\{.*?\}", response, re.DOTALL))
    for match in sorted(candidates, key=lambda m: len(m.group())):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 2: greedy outermost { ... }
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: regex field extraction fallback
    result: dict = {}
    for key in ("detailed_description", "description", "circuit_type"):
        m = re.search(rf'"{key}"\s*:\s*"(.*?)"', response, re.DOTALL)
        if m:
            result[key] = m.group(1)
    return result
