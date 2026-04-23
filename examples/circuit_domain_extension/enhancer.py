"""
CircuitEnhancer: post-process a CircuitIR in-place.

Responsibilities
----------------
- Normalize component reference designators (R1, C2, Q3 …)
- Infer missing circuit_type from component mix
- Fill empty summary fields from component/connection data
- Flag low-confidence figures (detection_score < threshold)

This module is intentionally simple: it runs heuristics on already-parsed
CircuitFigure objects and never calls the VLM again.

Callers: pipeline.py
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

from .ir import CircuitFigure, CircuitIR

logger = logging.getLogger(__name__)

# Mapping from component type strings → canonical reference-designator prefix
_TYPE_TO_PREFIX: Dict[str, str] = {
    "resistor": "R",
    "capacitor": "C",
    "inductor": "L",
    "diode": "D",
    "zener": "D",
    "led": "D",
    "transistor": "Q",
    "bjt": "Q",
    "bjt_pnp": "Q",
    "transistor_npn": "Q",
    "transistor_pnp": "Q",
    "mosfet": "M",
    "opamp": "U",
    "ic": "U",
    "voltage_source": "V",
    "dc_source": "V",
    "ac_source": "V",
    "current_source": "I",
    "ground": "GND",
    "wire": "",
    "junction": "",
}

# Topology heuristics: if these component types are present, guess a topology
_TOPOLOGY_HINTS: List[tuple] = [
    ({"opamp"}, "operational_amplifier"),
    ({"bjt", "transistor_npn", "transistor_pnp", "bjt_pnp"}, "transistor_amplifier"),
    ({"mosfet"}, "mosfet_circuit"),
    ({"capacitor", "inductor"}, "lc_filter"),
    ({"capacitor", "resistor"}, "rc_circuit"),
    ({"diode", "capacitor"}, "rectifier"),
    ({"diode"}, "diode_circuit"),
]


class CircuitEnhancer:
    """Post-process CircuitIR figures with normalization heuristics.

    Parameters
    ----------
    min_confidence:
        Figures with detection_score below this value are marked as
        low-confidence in their summary (but are NOT removed).
    """

    def __init__(self, min_confidence: float = 2.0) -> None:
        self.min_confidence = min_confidence

    def enhance(self, ir: CircuitIR) -> CircuitIR:
        """Enhance all circuit figures in *ir* in-place and return *ir*."""
        for figure in ir.circuit_figures:
            self._enhance_figure(figure)
        return ir

    # ------------------------------------------------------------------
    # Per-figure logic
    # ------------------------------------------------------------------

    def _enhance_figure(self, figure: CircuitFigure) -> None:
        if not figure.is_circuit:
            return

        self._normalize_component_refs(figure)
        self._infer_circuit_type(figure)
        self._fill_summary(figure)
        self._flag_low_confidence(figure)

    def _normalize_component_refs(self, figure: CircuitFigure) -> None:
        """Rewrite component lines to use canonical reference designators."""
        if not figure.circuit_design:
            return

        elements: List[Dict] = figure.circuit_design.get("elements", [])
        counters: Dict[str, int] = {}
        for element in elements:
            comp_type = element.get("type", "").lower()
            prefix = _TYPE_TO_PREFIX.get(comp_type)
            if not prefix:
                continue
            counters[prefix] = counters.get(prefix, 0) + 1
            existing_id: str = element.get("id", "")
            if not re.match(rf"^{re.escape(prefix)}\d+$", existing_id, re.IGNORECASE):
                element["id"] = f"{prefix}{counters[prefix]}"
                if not element.get("label"):
                    element["label"] = element["id"]

        # Rebuild human-readable component lines from updated elements
        if elements:
            lines = []
            for el in elements:
                line = f"- {el.get('label') or el.get('id')} [{el.get('id')}] type={el.get('type', '')}"
                if el.get("value"):
                    line += f", value={el['value']}"
                lines.append(line)
            figure.components = lines

    def _infer_circuit_type(self, figure: CircuitFigure) -> None:
        """Guess circuit topology from component types when missing."""
        if figure.circuit_type:
            return
        if not figure.circuit_design:
            return

        present_types = {
            el.get("type", "").lower()
            for el in figure.circuit_design.get("elements", [])
        }
        for required_types, topology in _TOPOLOGY_HINTS:
            if required_types & present_types:
                figure.circuit_type = topology
                return

    def _fill_summary(self, figure: CircuitFigure) -> None:
        """Build a minimal summary when none was produced by the VLM."""
        if figure.summary:
            return
        parts: List[str] = []
        if figure.circuit_type:
            parts.append(f"Circuit type: {figure.circuit_type}")
        if figure.components:
            n = len(figure.components)
            parts.append(f"{n} component{'s' if n != 1 else ''}")
        if figure.connections:
            n = len(figure.connections)
            parts.append(f"{n} connection{'s' if n != 1 else ''}")
        if figure.caption:
            parts.append(figure.caption[0][:80])
        figure.summary = "; ".join(parts) if parts else "Circuit diagram"

    def _flag_low_confidence(self, figure: CircuitFigure) -> None:
        if figure.detection_score < self.min_confidence and figure.summary:
            if "[low-confidence]" not in figure.summary:
                figure.summary = f"[low-confidence] {figure.summary}"
