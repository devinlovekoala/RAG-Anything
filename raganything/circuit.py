"""
Circuit-domain helpers for RAG-Anything.

This module provides a light-weight, Python-native adaptation of the circuit
design and netlist abstractions already used in the DrawSee backend so that
RAG-Anything can ingest circuit images as structured knowledge instead of only
storing free-form captions.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_PORT_POSITIONS = {
    "left": {"x": 0.0, "y": 50.0},
    "right": {"x": 100.0, "y": 50.0},
    "top": {"x": 50.0, "y": 0.0},
    "bottom": {"x": 50.0, "y": 100.0},
    "upper_right": {"x": 100.0, "y": 25.0},
    "lower_right": {"x": 100.0, "y": 75.0},
}


@dataclass
class CircuitPosition:
    x: float = 0.0
    y: float = 0.0


@dataclass
class CircuitPortPosition:
    side: str = "left"
    x: float = 0.0
    y: float = 0.0
    align: str = "center"


@dataclass
class CircuitPort:
    id: str
    name: str
    type: str = "bidirectional"
    position: CircuitPortPosition = field(default_factory=CircuitPortPosition)


@dataclass
class CircuitPortReference:
    element_id: str
    port_id: str


@dataclass
class CircuitElement:
    id: str
    type: str
    label: str = ""
    value: str = ""
    position: CircuitPosition = field(default_factory=CircuitPosition)
    rotation: int = 0
    properties: Dict[str, Any] = field(default_factory=dict)
    ports: List[CircuitPort] = field(default_factory=list)

    def display_name(self) -> str:
        return self.label or self.id

    def summary(self) -> str:
        parts = [f"type={self.type}"]
        if self.value:
            parts.append(f"value={self.value}")
        if self.properties:
            extras = []
            for key, value in self.properties.items():
                if key in {"label", "value"}:
                    continue
                if value in (None, "", []):
                    continue
                extras.append(f"{key}={value}")
            if extras:
                parts.append(", ".join(extras[:6]))
        if self.ports:
            parts.append(
                "ports=" + ", ".join(f"{port.id}:{port.type}" for port in self.ports[:6])
            )
        return "; ".join(parts)


@dataclass
class CircuitConnection:
    id: str
    source: CircuitPortReference
    target: CircuitPortReference
    net_name: str = ""
    connection_type: str = "electrical"


@dataclass
class CircuitMetadata:
    title: str = ""
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    circuit_type: str = ""


@dataclass
class CircuitDesign:
    elements: List[CircuitElement] = field(default_factory=list)
    connections: List[CircuitConnection] = field(default_factory=list)
    metadata: CircuitMetadata = field(default_factory=CircuitMetadata)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summarize(self, max_components: int = 8, max_connections: int = 6) -> str:
        parts: List[str] = []
        if self.metadata.title:
            parts.append(f"title={self.metadata.title}")
        if self.metadata.circuit_type:
            parts.append(f"topology={self.metadata.circuit_type}")
        if self.metadata.description:
            parts.append(self.metadata.description)
        if self.elements:
            component_text = ", ".join(
                f"{element.display_name()}[{element.type}]"
                + (f"={element.value}" if element.value else "")
                for element in self.elements[:max_components]
            )
            parts.append(f"components({len(self.elements)}): {component_text}")
        if self.connections:
            connection_text = ", ".join(
                self.describe_connection(connection)
                for connection in self.connections[:max_connections]
            )
            parts.append(f"connections({len(self.connections)}): {connection_text}")
        return "; ".join(part for part in parts if part)

    def component_lines(self) -> List[str]:
        lines = []
        for element in self.elements:
            line = f"- {element.display_name()} [{element.id}] type={element.type}"
            if element.value:
                line += f", value={element.value}"
            if element.properties:
                extra_pairs = []
                for key, value in element.properties.items():
                    if key in {"label", "value"} or value in (None, "", []):
                        continue
                    extra_pairs.append(f"{key}={value}")
                if extra_pairs:
                    line += ", " + ", ".join(extra_pairs[:6])
            lines.append(line)
        return lines

    def describe_connection(self, connection: CircuitConnection) -> str:
        net = f" via {connection.net_name}" if connection.net_name else ""
        return (
            f"{connection.source.element_id}.{connection.source.port_id} -> "
            f"{connection.target.element_id}.{connection.target.port_id}{net}"
        )

    def connection_lines(self) -> List[str]:
        return [f"- {self.describe_connection(connection)}" for connection in self.connections]

    def _entity_name_for_element(
        self, root_entity_name: str, element: CircuitElement
    ) -> str:
        return f"{root_entity_name}::{element.id}"

    def build_structured_kg_payload(
        self, root_entity_name: str
    ) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        entities: List[Dict[str, str]] = []
        relations: List[Dict[str, str]] = []
        element_name_map = {
            element.id: self._entity_name_for_element(root_entity_name, element)
            for element in self.elements
        }

        for element in self.elements:
            element_entity_name = element_name_map[element.id]
            entities.append(
                {
                    "entity_name": element_entity_name,
                    "entity_type": "circuit_component",
                    "description": (
                        f"Component {element.display_name()} in {root_entity_name}; "
                        f"{element.summary()}"
                    ),
                }
            )
            relations.append(
                {
                    "src_id": element_entity_name,
                    "tgt_id": root_entity_name,
                    "description": (
                        f"{element.display_name()} [{element.id}] belongs to "
                        f"{root_entity_name}"
                    ),
                    "relation_type": "belongs_to",
                    "keywords": "belongs_to,part_of,circuit_component",
                }
            )

        for connection in self.connections:
            source_name = element_name_map.get(connection.source.element_id)
            target_name = element_name_map.get(connection.target.element_id)
            if not source_name or not target_name or source_name == target_name:
                continue
            keywords = [
                connection.connection_type or "electrical",
                "connected_to",
                connection.net_name or "",
            ]
            relations.append(
                {
                    "src_id": source_name,
                    "tgt_id": target_name,
                    "description": (
                        f"{connection.source.element_id}.{connection.source.port_id} "
                        f"connects to {connection.target.element_id}.{connection.target.port_id}"
                        + (
                            f" via net {connection.net_name}"
                            if connection.net_name
                            else ""
                        )
                    ),
                    "relation_type": connection.connection_type or "electrical_connection",
                    "keywords": ",".join(keyword for keyword in keywords if keyword),
                }
            )

        return entities, relations


class CircuitNetlistParser:
    """Parse structured circuit outputs from a VLM into CircuitDesign."""

    KEY_VALUE_PATTERN = re.compile(r'(\w+)="([^"]*)"|(\w+)=([^\s]+)')
    JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
    CIRCUIT_JSON_KEYS = {"components", "connections", "elements", "metadata", "circuit_name"}
    SPICE_COMPONENT_PREFIXES = {
        "R": "resistor",
        "C": "capacitor",
        "L": "inductor",
        "D": "diode",
        "Q": "bjt",
        "M": "mosfet",
        "V": "dc_source",
        "I": "current_source",
        "X": "opamp",
        "U": "opamp",
    }
    PORT_TEMPLATES: Dict[str, List[tuple[str, str, str, str]]] = {
        "digital_input": [("out", "out", "output", "right")],
        "digital_output": [("in", "in", "input", "left")],
        "digital_clock": [("out", "out", "output", "right")],
        "digital_not": [
            ("in", "in", "input", "left"),
            ("out", "out", "output", "right"),
        ],
        "digital_and": [
            ("in1", "in1", "input", "left"),
            ("in2", "in2", "input", "left"),
            ("out", "out", "output", "right"),
        ],
        "digital_or": [
            ("in1", "in1", "input", "left"),
            ("in2", "in2", "input", "left"),
            ("out", "out", "output", "right"),
        ],
        "digital_nand": [
            ("in1", "in1", "input", "left"),
            ("in2", "in2", "input", "left"),
            ("out", "out", "output", "right"),
        ],
        "digital_nor": [
            ("in1", "in1", "input", "left"),
            ("in2", "in2", "input", "left"),
            ("out", "out", "output", "right"),
        ],
        "digital_xor": [
            ("in1", "in1", "input", "left"),
            ("in2", "in2", "input", "left"),
            ("out", "out", "output", "right"),
        ],
        "digital_xnor": [
            ("in1", "in1", "input", "left"),
            ("in2", "in2", "input", "left"),
            ("out", "out", "output", "right"),
        ],
        "digital_dff": [
            ("d", "d", "input", "left"),
            ("clk", "clk", "input", "bottom"),
            ("q", "q", "output", "right"),
        ],
        "resistor": [
            ("port1", "port1", "bidirectional", "left"),
            ("port2", "port2", "bidirectional", "right"),
        ],
        "capacitor": [
            ("port1", "port1", "bidirectional", "left"),
            ("port2", "port2", "bidirectional", "right"),
        ],
        "inductor": [
            ("port1", "port1", "bidirectional", "left"),
            ("port2", "port2", "bidirectional", "right"),
        ],
        "diode": [
            ("anode", "anode", "input", "left"),
            ("cathode", "cathode", "output", "right"),
        ],
        "diode_zener": [
            ("anode", "anode", "input", "left"),
            ("cathode", "cathode", "output", "right"),
        ],
        "diode_led": [
            ("anode", "anode", "input", "left"),
            ("cathode", "cathode", "output", "right"),
        ],
        "diode_schottky": [
            ("anode", "anode", "input", "left"),
            ("cathode", "cathode", "output", "right"),
        ],
        "bjt": [
            ("base", "base", "input", "left"),
            ("collector", "collector", "output", "upper_right"),
            ("emitter", "emitter", "output", "lower_right"),
        ],
        "bjt_pnp": [
            ("base", "base", "input", "left"),
            ("collector", "collector", "output", "upper_right"),
            ("emitter", "emitter", "output", "lower_right"),
        ],
        "transistor_npn": [
            ("base", "base", "input", "left"),
            ("collector", "collector", "output", "upper_right"),
            ("emitter", "emitter", "output", "lower_right"),
        ],
        "transistor_pnp": [
            ("base", "base", "input", "left"),
            ("collector", "collector", "output", "upper_right"),
            ("emitter", "emitter", "output", "lower_right"),
        ],
        "opamp": [
            ("input1", "input1", "input", "left"),
            ("input2", "input2", "input", "left"),
            ("output", "output", "output", "right"),
        ],
        "dc_source": [
            ("positive", "positive", "output", "right"),
            ("negative", "negative", "input", "left"),
        ],
        "ac_source": [
            ("positive", "positive", "output", "right"),
            ("negative", "negative", "input", "left"),
        ],
        "current_source": [
            ("positive", "positive", "output", "right"),
            ("negative", "negative", "input", "left"),
        ],
        "pulse_source": [
            ("positive", "positive", "output", "right"),
            ("negative", "negative", "input", "left"),
        ],
        "pwm_source": [
            ("positive", "positive", "output", "right"),
            ("negative", "negative", "input", "left"),
        ],
        "sine_source": [
            ("positive", "positive", "output", "right"),
            ("negative", "negative", "input", "left"),
        ],
        "ammeter": [
            ("in", "in", "bidirectional", "left"),
            ("out", "out", "bidirectional", "right"),
        ],
        "voltmeter": [
            ("positive", "positive", "bidirectional", "left"),
            ("negative", "negative", "bidirectional", "right"),
        ],
        "oscilloscope": [
            ("channel1", "channel1", "input", "left"),
            ("channel2", "channel2", "input", "left"),
            ("ground", "ground", "bidirectional", "bottom"),
        ],
        "wire": [
            ("port1", "port1", "bidirectional", "left"),
            ("port2", "port2", "bidirectional", "right"),
        ],
        "junction": [("junction", "junction", "bidirectional", "top")],
        "ground": [("ground", "ground", "input", "top")],
    }
    TYPE_ALIASES = {
        "op_amp": "opamp",
        "operational_amplifier": "opamp",
        "voltage_src": "dc_source",
        "voltage_source": "dc_source",
        "transistor": "bjt",
        "npn": "transistor_npn",
        "pnp": "transistor_pnp",
    }

    @classmethod
    def parse(cls, raw_text: str) -> Optional[CircuitDesign]:
        if not raw_text or not raw_text.strip():
            return None

        json_candidate = cls._extract_json_candidate(raw_text)
        if json_candidate:
            try:
                payload = json.loads(json_candidate)
                design = cls.parse_structured_payload(payload)
                if design is not None:
                    return design
            except json.JSONDecodeError:
                pass

        if "COMP" in raw_text and "WIRE" in raw_text:
            design = cls._parse_drawsee_netlist(raw_text)
            if design is not None:
                return design

        return cls._parse_spice_netlist(raw_text)

    @classmethod
    def parse_structured_payload(cls, payload: Any) -> Optional[CircuitDesign]:
        if payload is None:
            return None
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return cls.parse(payload)

        if not isinstance(payload, dict):
            return None
        if not cls.CIRCUIT_JSON_KEYS.intersection(payload.keys()):
            return None

        metadata = CircuitMetadata(
            title=payload.get("circuit_name")
            or payload.get("title")
            or payload.get("name")
            or cls._value_from_map(payload.get("metadata"), "title", "AI识别电路"),
            description=payload.get("description")
            or cls._value_from_map(payload.get("metadata"), "description", ""),
            circuit_type=payload.get("circuit_type")
            or cls._value_from_map(payload.get("metadata"), "circuit_type", ""),
        )

        elements_payload = payload.get("elements") or payload.get("components") or []
        elements: List[CircuitElement] = []
        nets_to_ports: Dict[str, List[CircuitPortReference]] = {}
        for index, raw_element in enumerate(elements_payload, start=1):
            if not isinstance(raw_element, dict):
                continue
            element_id = (
                raw_element.get("id")
                or raw_element.get("component_id")
                or raw_element.get("ref")
                or raw_element.get("name")
                or f"E{index}"
            )
            element_type = cls._normalize_type(
                raw_element.get("type") or raw_element.get("component_type") or "unknown"
            )
            label = raw_element.get("label") or raw_element.get("name") or element_id
            value = str(raw_element.get("value") or raw_element.get("labelValue") or "")
            properties = dict(raw_element.get("properties") or {})
            if value and "value" not in properties:
                properties["value"] = value
            if label and "label" not in properties:
                properties["label"] = label
            position = raw_element.get("position") or {}
            ports = cls._parse_ports(raw_element, element_type)
            pins = raw_element.get("pins") or raw_element.get("nets") or []
            if pins and ports:
                for port, pin in zip(ports, pins):
                    if isinstance(pin, str) and pin.strip():
                        nets_to_ports.setdefault(pin.strip(), []).append(
                            CircuitPortReference(element_id=element_id, port_id=port.id)
                        )
            element = CircuitElement(
                id=str(element_id),
                type=element_type,
                label=str(label),
                value=value,
                position=CircuitPosition(
                    x=float(position.get("x", 0.0)),
                    y=float(position.get("y", 0.0)),
                ),
                rotation=int(raw_element.get("rotation", 0) or 0),
                properties=properties,
                ports=ports,
            )
            elements.append(element)

        connections_payload = payload.get("connections") or []
        connections: List[CircuitConnection] = []
        for index, raw_connection in enumerate(connections_payload, start=1):
            connection = cls._parse_connection_payload(raw_connection, index)
            if connection is not None:
                connections.append(connection)

        if not connections and nets_to_ports:
            connections.extend(cls._build_connections_from_nets(nets_to_ports))

        if not elements:
            return None

        return CircuitDesign(elements=elements, connections=connections, metadata=metadata)

    @classmethod
    def _extract_json_candidate(cls, raw_text: str) -> Optional[str]:
        block_match = cls.JSON_BLOCK_PATTERN.search(raw_text)
        if block_match:
            return block_match.group(1)
        brace_match = re.search(r"\{[\s\S]*\}", raw_text)
        return brace_match.group(0) if brace_match else None

    @classmethod
    def _parse_drawsee_netlist(cls, raw_text: str) -> Optional[CircuitDesign]:
        elements: List[CircuitElement] = []
        connections: List[CircuitConnection] = []
        title = "AI识别电路"
        description = ""

        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("TITLE"):
                title = line[5:].strip() or title
                continue
            if line.startswith("DESCRIPTION"):
                description = line[11:].strip()
                continue
            if line.startswith("COMP"):
                kv = cls._parse_key_value_pairs(line[4:])
                element_id = kv.get("id", f"E{len(elements) + 1}")
                element_type = cls._normalize_type(kv.get("type", "unknown"))
                label = kv.get("label", element_id)
                properties: Dict[str, Any] = {**kv}
                properties.pop("id", None)
                properties.pop("type", None)
                properties.pop("x", None)
                properties.pop("y", None)
                properties.pop("rotation", None)
                element = CircuitElement(
                    id=element_id,
                    type=element_type,
                    label=label,
                    value=kv.get("value", ""),
                    position=CircuitPosition(
                        x=float(kv.get("x", 0.0) or 0.0),
                        y=float(kv.get("y", 0.0) or 0.0),
                    ),
                    rotation=int(float(kv.get("rotation", 0) or 0)),
                    properties=properties,
                    ports=cls.build_ports_for_type(element_type),
                )
                elements.append(element)
                continue
            if line.startswith("WIRE"):
                kv = cls._parse_key_value_pairs(line[4:])
                source = cls._parse_port_reference(kv.get("source"))
                target = cls._parse_port_reference(kv.get("target"))
                if source and target:
                    connections.append(
                        CircuitConnection(
                            id=kv.get("id", f"W{len(connections) + 1}"),
                            source=source,
                            target=target,
                            net_name=kv.get("net", ""),
                            connection_type=kv.get("type", "electrical"),
                        )
                    )

        if not elements:
            return None
        return CircuitDesign(
            elements=elements,
            connections=connections,
            metadata=CircuitMetadata(title=title, description=description),
        )

    @classmethod
    def _parse_spice_netlist(cls, raw_text: str) -> Optional[CircuitDesign]:
        elements: List[CircuitElement] = []
        nets_to_ports: Dict[str, List[CircuitPortReference]] = {}
        title = "SPICE Circuit"

        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("*"):
                if title == "SPICE Circuit":
                    title = stripped.lstrip("* ").strip() or title
                continue
            if stripped.startswith("."):
                continue
            tokens = stripped.split()
            if len(tokens) < 3:
                continue
            ref = tokens[0]
            component_type = cls._normalize_type(cls._infer_type_from_reference(ref))
            if component_type == "unknown":
                continue

            port_ids, value = cls._extract_spice_ports_and_value(component_type, tokens)
            if not port_ids:
                continue
            ports = cls.build_ports_for_type(component_type, len(port_ids))
            if len(ports) < len(port_ids):
                ports = cls._build_generic_ports(port_ids)
            for port, net in zip(ports, port_ids):
                if net and net.strip():
                    nets_to_ports.setdefault(net.strip(), []).append(
                        CircuitPortReference(element_id=ref, port_id=port.id)
                    )

            elements.append(
                CircuitElement(
                    id=ref,
                    type=component_type,
                    label=ref,
                    value=value,
                    properties={"value": value} if value else {},
                    ports=ports,
                )
            )

        if not elements:
            return None

        return CircuitDesign(
            elements=elements,
            connections=cls._build_connections_from_nets(nets_to_ports),
            metadata=CircuitMetadata(title=title),
        )

    @classmethod
    def _parse_ports(cls, raw_element: Dict[str, Any], element_type: str) -> List[CircuitPort]:
        ports_payload = raw_element.get("ports") or []
        if ports_payload:
            ports: List[CircuitPort] = []
            for index, raw_port in enumerate(ports_payload, start=1):
                if isinstance(raw_port, str):
                    raw_port = {"id": raw_port, "name": raw_port}
                if not isinstance(raw_port, dict):
                    continue
                position = raw_port.get("position") or {}
                default_pos = DEFAULT_PORT_POSITIONS.get(position.get("side", "left"), {})
                ports.append(
                    CircuitPort(
                        id=str(raw_port.get("id") or f"port{index}"),
                        name=str(raw_port.get("name") or raw_port.get("id") or f"port{index}"),
                        type=str(raw_port.get("type") or "bidirectional"),
                        position=CircuitPortPosition(
                            side=str(position.get("side", "left")),
                            x=float(position.get("x", default_pos.get("x", 0.0))),
                            y=float(position.get("y", default_pos.get("y", 50.0))),
                            align=str(position.get("align", "center")),
                        ),
                    )
                )
            if ports:
                return ports

        pins = raw_element.get("pins") or raw_element.get("nets") or []
        if pins:
            return cls._build_generic_ports(pins)
        return cls.build_ports_for_type(element_type)

    @classmethod
    def build_ports_for_type(
        cls, element_type: str, required_count: Optional[int] = None
    ) -> List[CircuitPort]:
        normalized_type = cls._normalize_type(element_type)
        templates = cls.PORT_TEMPLATES.get(normalized_type, [])
        ports: List[CircuitPort] = []
        for port_id, name, port_type, side in templates:
            default_pos = DEFAULT_PORT_POSITIONS.get(side, DEFAULT_PORT_POSITIONS["left"])
            ports.append(
                CircuitPort(
                    id=port_id,
                    name=name,
                    type=port_type,
                    position=CircuitPortPosition(
                        side=side,
                        x=default_pos["x"],
                        y=default_pos["y"],
                        align="center",
                    ),
                )
            )
        if required_count is not None and len(ports) < required_count:
            missing_ids = [f"port{i}" for i in range(len(ports) + 1, required_count + 1)]
            ports.extend(cls._build_generic_ports(missing_ids))
        return ports

    @classmethod
    def _build_generic_ports(cls, pin_names: Iterable[str]) -> List[CircuitPort]:
        ports: List[CircuitPort] = []
        pin_names = [str(name) for name in pin_names]
        last_index = max(len(pin_names) - 1, 1)
        for index, name in enumerate(pin_names):
            side = "left" if index == 0 else "right"
            if len(pin_names) > 2 and index not in (0, len(pin_names) - 1):
                side = "bottom"
            default_pos = DEFAULT_PORT_POSITIONS.get(side, DEFAULT_PORT_POSITIONS["left"])
            ports.append(
                CircuitPort(
                    id=name,
                    name=name,
                    type="bidirectional",
                    position=CircuitPortPosition(
                        side=side,
                        x=default_pos["x"],
                        y=default_pos["y"] if side != "bottom" else 20.0 + (60.0 * index / last_index),
                        align="center",
                    ),
                )
            )
        return ports

    @classmethod
    def _build_connections_from_nets(
        cls, nets_to_ports: Dict[str, List[CircuitPortReference]]
    ) -> List[CircuitConnection]:
        connections: List[CircuitConnection] = []
        skip_nets = {"0", "gnd", "ground", "vss"}
        for net_name, refs in nets_to_ports.items():
            if len(refs) < 2 or net_name.lower() in skip_nets:
                continue
            for index, (source, target) in enumerate(combinations(refs, 2), start=1):
                if source.element_id == target.element_id:
                    continue
                connections.append(
                    CircuitConnection(
                        id=f"{net_name}_{index}",
                        source=source,
                        target=target,
                        net_name=net_name,
                        connection_type="electrical",
                    )
                )
        return connections

    @classmethod
    def _parse_connection_payload(
        cls, raw_connection: Any, index: int
    ) -> Optional[CircuitConnection]:
        if not isinstance(raw_connection, dict):
            return None

        source = cls._port_reference_from_payload(
            raw_connection.get("source") or raw_connection.get("from"),
            raw_connection.get("source_port")
            or raw_connection.get("sourcePort")
            or raw_connection.get("from_port")
            or raw_connection.get("fromPort"),
        )
        target = cls._port_reference_from_payload(
            raw_connection.get("target") or raw_connection.get("to"),
            raw_connection.get("target_port")
            or raw_connection.get("targetPort")
            or raw_connection.get("to_port")
            or raw_connection.get("toPort"),
        )
        if source is None or target is None:
            return None

        return CircuitConnection(
            id=str(raw_connection.get("id") or f"conn_{index}"),
            source=source,
            target=target,
            net_name=str(raw_connection.get("net") or raw_connection.get("net_name") or ""),
            connection_type=str(raw_connection.get("type") or "electrical"),
        )

    @classmethod
    def _port_reference_from_payload(
        cls, payload: Any, explicit_port_id: Any = None
    ) -> Optional[CircuitPortReference]:
        if isinstance(payload, dict):
            element_id = payload.get("elementId") or payload.get("element_id") or payload.get("id")
            port_id = payload.get("portId") or payload.get("port_id") or explicit_port_id
            if element_id and port_id:
                return CircuitPortReference(str(element_id), str(port_id))
            return None
        if isinstance(payload, str):
            if "." in payload:
                return cls._parse_port_reference(payload)
            if explicit_port_id:
                return CircuitPortReference(str(payload), str(explicit_port_id))
            return CircuitPortReference(str(payload), "port1")
        return None

    @classmethod
    def _parse_port_reference(cls, raw_reference: Optional[str]) -> Optional[CircuitPortReference]:
        if not raw_reference or "." not in raw_reference:
            return None
        element_id, port_id = raw_reference.split(".", 1)
        return CircuitPortReference(element_id=element_id, port_id=port_id)

    @classmethod
    def _parse_key_value_pairs(cls, raw_content: str) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for match in cls.KEY_VALUE_PATTERN.finditer(raw_content):
            if match.group(1) is not None:
                values[match.group(1)] = match.group(2)
            elif match.group(3) is not None:
                values[match.group(3)] = match.group(4)
        return values

    @classmethod
    def _extract_spice_ports_and_value(
        cls, component_type: str, tokens: List[str]
    ) -> tuple[List[str], str]:
        if component_type in {
            "resistor",
            "capacitor",
            "inductor",
            "diode",
            "diode_zener",
            "diode_led",
            "diode_schottky",
        }:
            if len(tokens) < 4:
                return [], ""
            return tokens[1:3], tokens[3]
        if component_type in {
            "dc_source",
            "ac_source",
            "current_source",
            "pulse_source",
            "pwm_source",
            "sine_source",
        }:
            if len(tokens) < 4:
                return [], ""
            return tokens[1:3], " ".join(tokens[3:])
        if component_type in {
            "bjt",
            "bjt_pnp",
            "transistor_npn",
            "transistor_pnp",
            "mosfet",
        }:
            if len(tokens) < 5:
                return [], ""
            return tokens[1:4], " ".join(tokens[4:])
        if component_type == "opamp":
            if len(tokens) < 5:
                return [], ""
            return tokens[1:4], " ".join(tokens[4:])
        if len(tokens) >= 3:
            return tokens[1:3], " ".join(tokens[3:])
        return [], ""

    @classmethod
    def _infer_type_from_reference(cls, reference: str) -> str:
        if not reference:
            return "unknown"
        return cls.SPICE_COMPONENT_PREFIXES.get(reference[0].upper(), "unknown")

    @classmethod
    def _normalize_type(cls, raw_type: str) -> str:
        normalized = str(raw_type or "unknown").strip().lower().replace("-", "_")
        return cls.TYPE_ALIASES.get(normalized, normalized)

    @staticmethod
    def _value_from_map(raw_value: Any, key: str, default: str = "") -> str:
        if isinstance(raw_value, dict):
            return str(raw_value.get(key) or default)
        return default


class CircuitSpiceConverter:
    """Generate a SPICE-like representation from a CircuitDesign."""

    COMPONENT_PREFIXES = {
        "resistor": "R",
        "capacitor": "C",
        "inductor": "L",
        "diode": "D",
        "diode_zener": "D",
        "diode_led": "D",
        "diode_schottky": "D",
        "bjt": "Q",
        "bjt_pnp": "Q",
        "transistor_npn": "Q",
        "transistor_pnp": "Q",
        "mosfet": "M",
        "dc_source": "V",
        "ac_source": "V",
        "current_source": "I",
        "pulse_source": "V",
        "pwm_source": "V",
        "sine_source": "V",
        "opamp": "X",
    }
    COMMENT_ONLY_TYPES = {"ground", "ammeter", "voltmeter", "oscilloscope", "wire", "junction"}

    def generate_netlist(self, design: Optional[CircuitDesign]) -> str:
        if design is None:
            return ""

        endpoint_to_net = self._build_endpoint_to_net(design)
        title = design.metadata.title or "Untitled Circuit"
        lines = [f"* {title}", "* Generated by RAG-Anything CircuitModalProcessor"]

        for element in design.elements:
            component_line = self._element_to_spice_line(element, endpoint_to_net)
            if component_line:
                lines.append(component_line)

        lines.append(".end")
        return "\n".join(lines)

    def _build_endpoint_to_net(self, design: CircuitDesign) -> Dict[str, str]:
        parent: Dict[str, str] = {}

        def find(key: str) -> str:
            parent.setdefault(key, key)
            if parent[key] != key:
                parent[key] = find(parent[key])
            return parent[key]

        def union(a: str, b: str) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        explicit_net_names: Dict[str, str] = {}

        for element in design.elements:
            for port in element.ports:
                find(f"{element.id}:{port.id}")

        for connection in design.connections:
            source_key = f"{connection.source.element_id}:{connection.source.port_id}"
            target_key = f"{connection.target.element_id}:{connection.target.port_id}"
            union(source_key, target_key)
            if connection.net_name:
                explicit_net_names[find(source_key)] = connection.net_name

        groups: Dict[str, List[str]] = {}
        for endpoint in list(parent.keys()):
            groups.setdefault(find(endpoint), []).append(endpoint)

        endpoint_to_net: Dict[str, str] = {}
        node_index = 1
        for root, endpoints in groups.items():
            net_name = explicit_net_names.get(root) or self._infer_special_net(endpoints, design)
            if not net_name:
                net_name = f"N{node_index}"
                node_index += 1
            for endpoint in endpoints:
                endpoint_to_net[endpoint] = net_name

        return endpoint_to_net

    def _infer_special_net(self, endpoints: List[str], design: CircuitDesign) -> str:
        element_map = {element.id: element for element in design.elements}
        for endpoint in endpoints:
            element_id, port_id = endpoint.split(":", 1)
            element = element_map.get(element_id)
            if element is None:
                continue
            if element.type == "ground" or port_id.lower() in {"ground", "gnd"}:
                return "0"
        return ""

    def _element_to_spice_line(
        self, element: CircuitElement, endpoint_to_net: Dict[str, str]
    ) -> str:
        if element.type in self.COMMENT_ONLY_TYPES:
            return self._measurement_comment(element, endpoint_to_net)

        prefix = self.COMPONENT_PREFIXES.get(element.type)
        if prefix is None:
            return ""
        node_refs = [
            endpoint_to_net.get(f"{element.id}:{port.id}", "0") for port in element.ports
        ]
        node_refs = [node for node in node_refs if node]
        if len(node_refs) < 2:
            return ""

        value = (
            element.value
            or str(element.properties.get("value", ""))
            or str(element.properties.get("model", ""))
        )

        if element.type == "resistor":
            resistance = self._property_text(element, "resistance", value or "1000")
            return " ".join(
                [f"{prefix}{self._normalize_reference_suffix(element.id, prefix)}", *node_refs[:2], resistance]
            )
        if element.type == "capacitor":
            capacitance = self._property_text(element, "capacitance", value or "1e-6")
            return " ".join(
                [f"{prefix}{self._normalize_reference_suffix(element.id, prefix)}", *node_refs[:2], capacitance]
            )
        if element.type == "inductor":
            inductance = self._property_text(element, "inductance", value or "1e-3")
            return " ".join(
                [f"{prefix}{self._normalize_reference_suffix(element.id, prefix)}", *node_refs[:2], inductance]
            )
        if element.type in {"diode", "diode_zener", "diode_led", "diode_schottky"}:
            model = self._property_text(element, "model", value or "DDEFAULT")
            return " ".join(
                [f"{prefix}{self._normalize_reference_suffix(element.id, prefix)}", *node_refs[:2], model]
            )
        if element.type == "dc_source":
            voltage = self._property_text(element, "voltage", value or "5")
            return " ".join(
                [
                    f"{prefix}{self._normalize_reference_suffix(element.id, prefix)}",
                    *node_refs[:2],
                    "DC",
                    voltage,
                ]
            )
        if element.type == "current_source":
            current = self._property_text(element, "current", value or "0.01")
            return " ".join(
                [
                    f"{prefix}{self._normalize_reference_suffix(element.id, prefix)}",
                    *node_refs[:2],
                    "DC",
                    current,
                ]
            )
        if element.type == "ac_source":
            amplitude = self._property_text(element, "amplitude", "1.0")
            frequency = self._property_text(element, "frequency", "1000.0")
            return " ".join(
                [
                    f"{prefix}{self._normalize_reference_suffix(element.id, 'VAC')}",
                    *node_refs[:2],
                    f"SIN(0 {amplitude} {frequency})",
                ]
            )
        if element.type == "pulse_source":
            v1 = self._property_text(element, "v1", "0")
            v2 = self._property_text(element, "v2", "5")
            td = self._property_text(element, "td", "0")
            tr = self._property_text(element, "tr", "1e-9")
            tf = self._property_text(element, "tf", "1e-9")
            pw = self._property_text(element, "pw", "1e-6")
            per = self._property_text(element, "per", "2e-6")
            return " ".join(
                [
                    f"{prefix}{self._normalize_reference_suffix(element.id, 'VPULSE')}",
                    *node_refs[:2],
                    f"PULSE({v1} {v2} {td} {tr} {tf} {pw} {per})",
                ]
            )
        if element.type == "pwm_source":
            low = self._property_text(element, "vlow", "0")
            high = self._property_text(element, "vhigh", "5")
            freq_hz = self._float_property(element, "freqHz", 1000.0)
            duty = self._float_property(element, "duty", 0.5)
            period = 1.0 / freq_hz if freq_hz > 0 else 1e-3
            width = max(period * duty, 1e-9)
            return " ".join(
                [
                    f"{prefix}{self._normalize_reference_suffix(element.id, 'VPWM')}",
                    *node_refs[:2],
                    f"PULSE({low} {high} 0 1n 1n {width} {period})",
                ]
            )
        if element.type == "sine_source":
            offset = self._property_text(element, "vo", "0")
            amplitude = self._property_text(element, "va", "1")
            frequency = self._property_text(element, "freq", "1000")
            phase = self._property_text(element, "phase", "0")
            return " ".join(
                [
                    f"{prefix}{self._normalize_reference_suffix(element.id, 'VSIN')}",
                    *node_refs[:2],
                    f"SIN({offset} {amplitude} {frequency} 0 {phase})",
                ]
            )
        if element.type in {"bjt", "bjt_pnp", "transistor_npn", "transistor_pnp", "mosfet"}:
            default_model = "Q2N3906" if element.type in {"bjt_pnp", "transistor_pnp"} else "Q2N3904"
            return " ".join(
                [
                    f"{prefix}{self._normalize_reference_suffix(element.id, prefix)}",
                    *node_refs[:3],
                    self._property_text(element, "model", value or default_model),
                ]
            )
        if element.type == "opamp":
            return " ".join(
                [
                    f"{prefix}{self._normalize_reference_suffix(element.id, prefix)}",
                    *node_refs[:3],
                    self._property_text(element, "model", value or "LM741"),
                ]
            )
        return ""

    def _measurement_comment(
        self, element: CircuitElement, endpoint_to_net: Dict[str, str]
    ) -> str:
        nets = ", ".join(
            endpoint_to_net.get(f"{element.id}:{port.id}", "0") for port in element.ports
        )
        return (
            f"* Instrument {element.display_name()} [{element.id}] "
            f"(type: {element.type}, nets: {nets})"
        )

    @staticmethod
    def _property_text(element: CircuitElement, key: str, default: str) -> str:
        raw_value = element.properties.get(key, default)
        text = str(raw_value).strip()
        return text or default

    @staticmethod
    def _float_property(element: CircuitElement, key: str, default: float) -> float:
        raw_value = element.properties.get(key, default)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_reference_suffix(reference: str, prefix: str) -> str:
        if reference.upper().startswith(prefix):
            return reference[len(prefix) :]
        return reference


class CircuitDetector:
    """Heuristic detector for routing image content to circuit processing."""

    KEYWORDS = {
        "circuit",
        "schematic",
        "netlist",
        "resistor",
        "capacitor",
        "inductor",
        "op-amp",
        "opamp",
        "op amp",
        "operational amplifier",
        "non-inverting",
        "inverting input",
        "terminal voltage",
        "terminal current",
        "transistor",
        "diode",
        "ground",
        "gnd",
        "vcc",
        "vdd",
        "feedback",
        "filter",
        "amplifier",
        "电路",
        "原理图",
        "运放",
        "电阻",
        "电容",
        "电感",
        "三极管",
        "二极管",
        "反馈",
        "滤波",
        "放大",
    }
    REFERENCE_PATTERNS = [
        re.compile(r"\bR\d+\b", re.IGNORECASE),
        re.compile(r"\bC\d+\b", re.IGNORECASE),
        re.compile(r"\bL\d+\b", re.IGNORECASE),
        re.compile(r"\bQ\d+\b", re.IGNORECASE),
        re.compile(r"\bU\d+\b", re.IGNORECASE),
        re.compile(r"\bV(?:in|out|cc|dd|ss)\b", re.IGNORECASE),
    ]

    @classmethod
    def is_likely_circuit(
        cls, item: Optional[Dict[str, Any]], context_text: str = "", threshold: float = 2.0
    ) -> bool:
        return cls.score_item(item, context_text) >= threshold

    @classmethod
    def score_item(cls, item: Optional[Dict[str, Any]], context_text: str = "") -> float:
        if item is None:
            return 0.0

        text_parts = [
            str(item.get("img_path", "")),
            " ".join(item.get("image_caption", item.get("img_caption", [])) or []),
            " ".join(item.get("image_footnote", item.get("img_footnote", [])) or []),
            str(item.get("text", "")),
            context_text or "",
            str(item.get("modal_subtype", "")),
        ]
        evidence = " ".join(part for part in text_parts if part).lower()
        if not evidence:
            return 0.0

        score = 0.0
        keyword_hits = {keyword for keyword in cls.KEYWORDS if keyword in evidence}
        score += min(len(keyword_hits), 4) * 0.75

        for pattern in cls.REFERENCE_PATTERNS:
            if pattern.search(evidence):
                score += 0.8

        if any(token in evidence for token in ("circuit", "schematic", "电路", "原理图")):
            score += 1.2
        if any(
            token in evidence
            for token in (
                "operational amplifier",
                "op amp",
                "op-amp",
                "non-inverting",
                "inverting input",
            )
        ):
            score += 1.0
        if any(token in evidence for token in ("wiring", "topology", "反馈", "放大")):
            score += 0.6
        return score
