"""
Circuit Intermediate Representation (IR)

Decouples the circuit-domain data model from RAG-Anything's internal schema.
All other extension modules produce or consume CircuitIR; the adapter converts
it to the content_list format that RAG-Anything's insert_content_list() expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CircuitFigure:
    """One extracted circuit figure from a document page."""

    page_idx: int
    img_path: str                          # absolute path to the extracted image file
    caption: List[str] = field(default_factory=list)
    footnote: List[str] = field(default_factory=list)

    # Populated by CircuitParser after VLM extraction
    is_circuit: bool = False
    detection_score: float = 0.0
    circuit_design: Optional[Dict[str, Any]] = None  # serialised CircuitDesign dict
    circuit_type: str = ""
    netlist: str = ""
    components: List[str] = field(default_factory=list)   # human-readable component lines
    connections: List[str] = field(default_factory=list)  # human-readable connection lines
    summary: str = ""                                     # one-paragraph description


@dataclass
class CircuitIR:
    """
    Complete intermediate representation of a circuit-domain document.

    Produced by CircuitParser, consumed by CircuitEnhancer and CircuitAdapter.
    Contains both the original multimodal content blocks (text, tables, equations)
    and the circuit-specific parsed figures.
    """

    source_path: str
    doc_id: str = ""

    # Raw content blocks from the document parser (MinerU / Docling output format)
    text_blocks: List[Dict[str, Any]] = field(default_factory=list)
    table_blocks: List[Dict[str, Any]] = field(default_factory=list)
    equation_blocks: List[Dict[str, Any]] = field(default_factory=list)
    image_blocks: List[Dict[str, Any]] = field(default_factory=list)  # non-circuit images

    # Circuit-specific parsed figures
    circuit_figures: List[CircuitFigure] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    # Stats set by CircuitParser
    total_pages: int = 0
    circuit_figure_count: int = 0
    non_circuit_image_count: int = 0

    def summary_stats(self) -> str:
        return (
            f"source={self.source_path}, pages={self.total_pages}, "
            f"text_blocks={len(self.text_blocks)}, tables={len(self.table_blocks)}, "
            f"equations={len(self.equation_blocks)}, "
            f"circuit_figures={self.circuit_figure_count}, "
            f"other_images={self.non_circuit_image_count}"
        )
