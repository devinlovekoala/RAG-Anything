"""
circuit_domain_extension — circuit-aware RAG ingestion for RAG-Anything.

Public API
----------
CircuitPipeline          End-to-end pipeline (parse → enhance → insert)
ingest_circuit_document  Convenience one-shot function
CircuitIR                Intermediate representation of a parsed document
CircuitFigure            Single extracted circuit figure
CircuitParser            PDF → CircuitIR (VLM-backed)
CircuitEnhancer          Heuristic post-processor (normalisation, topology inference)
CircuitAdapter           CircuitIR → RAG-Anything content_list
"""

from .adapter import CircuitAdapter
from .enhancer import CircuitEnhancer
from .ir import CircuitFigure, CircuitIR
from .parser import CircuitParser
from .pipeline import CircuitPipeline, ingest_circuit_document

__all__ = [
    "CircuitAdapter",
    "CircuitEnhancer",
    "CircuitFigure",
    "CircuitIR",
    "CircuitParser",
    "CircuitPipeline",
    "ingest_circuit_document",
]
