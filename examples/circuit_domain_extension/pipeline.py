"""
CircuitPipeline: end-to-end circuit-domain document ingestion.

Ties together CircuitParser → CircuitEnhancer → CircuitAdapter →
RAGAnything.insert_content_list() in a single convenience wrapper.

Usage
-----
    from raganything import RAGAnything
    from examples.circuit_domain_extension.pipeline import CircuitPipeline

    rag = RAGAnything(...)
    await rag.initialize()

    pipeline = CircuitPipeline(rag)
    await pipeline.ingest("lecture.pdf")

Or via the convenience function:

    from examples.circuit_domain_extension.pipeline import ingest_circuit_document
    await ingest_circuit_document(rag, "lecture.pdf")

Callers: __init__.py (re-export), run_experiment.py (entry point)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional

from .adapter import CircuitAdapter
from .enhancer import CircuitEnhancer
from .ir import CircuitIR
from .parser import CircuitParser

logger = logging.getLogger(__name__)


class CircuitPipeline:
    """Orchestrate full circuit-domain ingestion for one RAGAnything instance.

    Parameters
    ----------
    rag:
        An *initialized* ``RAGAnything`` instance.  The pipeline reads
        ``rag.vision_model_func`` (falling back to ``rag.llm_model_func``)
        for VLM calls.
    detector_threshold:
        Passed to ``CircuitParser``.  Images whose heuristic score falls
        below this value skip VLM extraction.
    include_raw_images:
        Passed to ``CircuitAdapter``.  When False, non-circuit images are
        dropped from the inserted content_list.
    """

    def __init__(
        self,
        rag: Any,
        detector_threshold: float = 2.0,
        include_raw_images: bool = True,
    ) -> None:
        self.rag = rag
        modal_fn: Optional[Callable] = getattr(rag, "vision_model_func", None) or getattr(
            rag, "llm_model_func", None
        )
        if modal_fn is None:
            raise ValueError(
                "RAGAnything instance must have vision_model_func or llm_model_func set "
                "before building a CircuitPipeline."
            )
        self._parser = CircuitParser(
            modal_caption_func=modal_fn,
            detector_threshold=detector_threshold,
        )
        self._enhancer = CircuitEnhancer()
        self._adapter = CircuitAdapter(include_raw_images=include_raw_images)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest(
        self,
        file_path: str,
        *,
        parse_method: str = "auto",
        lang: str = "en",
        output_dir: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> CircuitIR:
        """Parse, enhance, and insert *file_path* into the RAG knowledge base.

        Returns the intermediate ``CircuitIR`` so callers can inspect
        extraction results without re-parsing.
        """
        file_path = str(Path(file_path).resolve())
        logger.info("CircuitPipeline: ingesting %s", file_path)

        ir = await self._parser.parse(
            file_path,
            parse_method=parse_method,
            lang=lang,
            output_dir=output_dir,
        )
        logger.info("Parsed: %s", ir.summary_stats())

        self._enhancer.enhance(ir)

        content_list = self._adapter.build_content_list(ir)
        logger.info(
            "Inserting %d content items (%d circuit figures)",
            len(content_list),
            ir.circuit_figure_count,
        )

        await self.rag.insert_content_list(
            content_list,
            file_path=file_path,
            doc_id=doc_id,
        )
        logger.info("CircuitPipeline: ingestion complete for %s", file_path)
        return ir

    async def ingest_many(
        self,
        file_paths: List[str],
        **kwargs: Any,
    ) -> List[CircuitIR]:
        """Ingest multiple documents sequentially. Keyword args forwarded to ``ingest``."""
        results: List[CircuitIR] = []
        for path in file_paths:
            try:
                ir = await self.ingest(path, **kwargs)
                results.append(ir)
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", path, exc)
        return results


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

async def ingest_circuit_document(
    rag: Any,
    file_path: str,
    *,
    detector_threshold: float = 2.0,
    include_raw_images: bool = True,
    parse_method: str = "auto",
    lang: str = "en",
    output_dir: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> CircuitIR:
    """One-shot helper: build a pipeline and ingest a single document.

    Equivalent to::

        pipeline = CircuitPipeline(rag, detector_threshold, include_raw_images)
        return await pipeline.ingest(file_path, ...)
    """
    pipeline = CircuitPipeline(
        rag,
        detector_threshold=detector_threshold,
        include_raw_images=include_raw_images,
    )
    return await pipeline.ingest(
        file_path,
        parse_method=parse_method,
        lang=lang,
        output_dir=output_dir,
        doc_id=doc_id,
    )
