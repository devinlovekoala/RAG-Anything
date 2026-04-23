"""
Circuit-domain prompt constants.

Migrated from raganything/prompt.py (CIRCUIT_ANALYSIS_SYSTEM, circuit_vision_prompt,
circuit_vision_prompt_with_context, circuit_chunk) so the core library stays domain-agnostic.
Extended with CIRCUIT_QA_SYSTEM for benchmark evaluation.

Callers:
  parser.py   — CIRCUIT_ANALYSIS_SYSTEM, CIRCUIT_VISION_PROMPT, CIRCUIT_VISION_PROMPT_WITH_CONTEXT
  adapter.py  — CIRCUIT_CHUNK
  benchmark.py — CIRCUIT_QA_SYSTEM
"""

from __future__ import annotations

CIRCUIT_ANALYSIS_SYSTEM: str = (
    "You are an expert electrical engineer. Recover circuit structure, "
    "component semantics, and signal topology accurately."
)

CIRCUIT_VISION_PROMPT: str = """Analyze this image as a potential electrical circuit diagram and return JSON only.

{{
    "detailed_description": "A precise technical description of the circuit image. Include circuit purpose, major components, visible values, signal flow, topology, and any ambiguities that remain.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "circuit_design",
        "summary": "concise summary of the circuit and its function (max 100 words)"
    }},
    "circuit_design": {{
        "circuit_name": "{entity_name}",
        "circuit_type": "best-effort topology label such as inverting_amplifier or rc_filter",
        "description": "brief functional description",
        "components": [
            {{"id": "R1", "type": "resistor", "label": "R1", "value": "10k", "pins": ["net_in", "net_mid"]}}
        ],
        "connections": [
            {{"from": "R1", "from_port": "port2", "to": "U1", "to_port": "input2", "net": "net_mid", "type": "signal"}}
        ]
    }}
}}

If this is not a circuit diagram, still return valid JSON but set `"circuit_design"` to null and describe the image normally.

Image details:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}"""

CIRCUIT_VISION_PROMPT_WITH_CONTEXT: str = """Analyze this image as a potential electrical circuit diagram using the surrounding document context. Return JSON only.

{{
    "detailed_description": "A precise technical description of the circuit image. Include circuit purpose, major components, visible values, signal flow, topology, and how it relates to the surrounding text.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "circuit_design",
        "summary": "concise summary of the circuit, its function, and its relation to the surrounding content (max 100 words)"
    }},
    "circuit_design": {{
        "circuit_name": "{entity_name}",
        "circuit_type": "best-effort topology label such as inverting_amplifier or rc_filter",
        "description": "brief functional description",
        "components": [
            {{"id": "R1", "type": "resistor", "label": "R1", "value": "10k", "pins": ["net_in", "net_mid"]}}
        ],
        "connections": [
            {{"from": "R1", "from_port": "port2", "to": "U1", "to_port": "input2", "net": "net_mid", "type": "signal"}}
        ]
    }}
}}

If this is not a circuit diagram, still return valid JSON but set `"circuit_design"` to null and describe the image normally.

Context from surrounding content:
{context}

Image details:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}"""

# chunk_type field added vs. raganything/prompt.py original to support typed retrieval
CIRCUIT_CHUNK: str = """Circuit Diagram Analysis:
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}
Circuit Type: {circuit_type}
Circuit Summary: {circuit_summary}
Circuit Components:
{circuit_components}
Circuit Connections:
{circuit_connections}
SPICE Netlist:
{circuit_netlist}

Technical Analysis: {enhanced_caption}"""

CIRCUIT_QA_SYSTEM: str = (
    "You are an expert electrical engineering assistant. "
    "Answer questions about circuits precisely, referencing component values, "
    "topologies, and signal paths from the provided context."
)
