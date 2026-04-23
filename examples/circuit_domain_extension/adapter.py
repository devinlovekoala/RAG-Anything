"""
CircuitAdapter: CircuitIR → RAG-Anything content_list

Converts a fully-populated CircuitIR (produced by CircuitParser +
CircuitEnhancer) into the content_list format that
``RAGAnything.insert_content_list()`` understands.

The produced list contains:
  - All text / table / equation / non-circuit-image blocks verbatim
  - Each CircuitFigure as a "circuit" typed item with the structured fields
    that raganything/processor.py's circuit_chunk formatter expects

Callers: pipeline.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .ir import CircuitFigure, CircuitIR

logger = logging.getLogger(__name__)


class CircuitAdapter:
    """Build a RAG-Anything content_list from a CircuitIR.

    The ``include_raw_images`` flag controls whether non-circuit image blocks
    from the IR are passed through as plain "image" items (default True).
    Set to False when you want to suppress generic image processing entirely.
    """

    def __init__(self, include_raw_images: bool = True) -> None:
        self.include_raw_images = include_raw_images

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_content_list(self, ir: CircuitIR) -> List[Dict[str, Any]]:
        """Return the content_list ready for ``insert_content_list()``."""
        items: List[Dict[str, Any]] = []

        items.extend(ir.text_blocks)
        items.extend(ir.table_blocks)
        items.extend(ir.equation_blocks)

        if self.include_raw_images:
            items.extend(ir.image_blocks)

        for figure in ir.circuit_figures:
            items.append(self._figure_to_item(figure))

        # Sort by page_idx so items arrive in document order
        items.sort(key=lambda x: x.get("page_idx", 0))
        return items

    # ------------------------------------------------------------------
    # Internal conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _figure_to_item(figure: CircuitFigure) -> Dict[str, Any]:
        """Convert one CircuitFigure to a circuit-typed content_list item.

        Field names match what raganything/processor.py reads when it
        encounters ``content_type == "circuit"``.
        """
        return {
            "type": "circuit",
            "img_path": figure.img_path,
            "image_caption": figure.caption,
            "image_footnote": figure.footnote,
            "page_idx": figure.page_idx,
            # Structured circuit fields consumed by circuit_chunk formatter
            "circuit_design": figure.circuit_design or {},
            "circuit_type": figure.circuit_type,
            "circuit_summary": figure.summary,
            "circuit_components": figure.components,
            "circuit_connections": figure.connections,
            "circuit_netlist": figure.netlist,
            # Extra metadata for downstream retrieval / filtering
            "circuit_detected": figure.is_circuit,
            "detection_score": figure.detection_score,
        }
