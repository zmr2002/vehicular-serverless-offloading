from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import random


@dataclass(slots=True, frozen=True)
class RouteGenerationSummary:
    requested_vehicles: int
    written_vehicles: int
    simulation_end_s: float
    seed: int
    output_file: str


def generate_exact_routes(
    net_file: str | Path,
    output_file: str | Path,
    vehicle_count: int,
    simulation_end_s: float,
    seed: int,
) -> RouteGenerationSummary:
    if vehicle_count <= 0 or simulation_end_s <= 0:
        raise ValueError("vehicle_count and simulation_end_s must be positive")
    import sumolib

    network = sumolib.net.readNet(str(net_file))
    edges = sorted(
        (
            edge
            for edge in network.getEdges()
            if not edge.getID().startswith(":") and edge.allows("passenger") and edge.getLength() > 10.0
        ),
        key=lambda edge: edge.getID(),
    )
    if len(edges) < 2:
        raise RuntimeError("network does not contain enough passenger edges")
    rng = random.Random(seed)
    routes: list[tuple[str, ...]] = []
    attempts = 0
    max_attempts = vehicle_count * 100
    while len(routes) < vehicle_count and attempts < max_attempts:
        attempts += 1
        origin, destination = rng.sample(edges, 2)
        # Restrict the complete path, not merely its endpoints, to the vehicle
        # class declared below. Without vClass SUMO may return a geometrically
        # connected path containing lanes that reject passenger vehicles.
        path, _ = network.getShortestPath(origin, destination, vClass="passenger")
        if path and len(path) >= 2:
            routes.append(tuple(edge.getID() for edge in path))
    if len(routes) != vehicle_count:
        raise RuntimeError(f"generated {len(routes)} of {vehicle_count} routes after {attempts} attempts")

    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    interval = simulation_end_s / vehicle_count
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        '  <vType id="passenger" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="25"/>',
    ]
    for index, route in enumerate(routes):
        depart = min(simulation_end_s - 0.001, index * interval)
        edge_text = " ".join(escape(edge_id, quote=True) for edge_id in route)
        lines.append(f'  <vehicle id="veh-{index:05d}" type="passenger" depart="{depart:.3f}">')
        lines.append(f'    <route edges="{edge_text}"/>')
        lines.append("  </vehicle>")
    lines.append("</routes>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RouteGenerationSummary(vehicle_count, len(routes), simulation_end_s, seed, str(output))


def prepare_sumo_scenario(
    net_file: str | Path,
    output_dir: str | Path,
    vehicle_count: int,
    simulation_end_s: float,
    seed: int,
) -> Path:
    net_path = Path(net_file).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    end_tag = f"{simulation_end_s:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    stem = f"paper-{vehicle_count}-seed{seed}-end{end_tag}"
    route_file = destination / f"{stem}.rou.xml"
    config_file = destination / f"{stem}.sumocfg"
    expected_tag = '<vehicle id="'
    existing_count = route_file.read_text(encoding="utf-8").count(expected_tag) if route_file.exists() else 0
    if existing_count != vehicle_count:
        generate_exact_routes(net_path, route_file, vehicle_count, simulation_end_s, seed)
    config_file.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                "<configuration>",
                "  <input>",
                f'    <net-file value="{escape(str(net_path), quote=True)}"/>',
                f'    <route-files value="{escape(str(route_file), quote=True)}"/>',
                "  </input>",
                "  <time>",
                '    <begin value="0"/>',
                f'    <end value="{simulation_end_s:.3f}"/>',
                "  </time>",
                "</configuration>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_file
