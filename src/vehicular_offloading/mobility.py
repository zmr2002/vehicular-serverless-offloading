from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Protocol
import uuid
import xml.etree.ElementTree as ET

from .config import SimulationConfig


@dataclass(slots=True, frozen=True)
class MobilityVehicle:
    vehicle_id: str
    position: tuple[float, float]
    speed_mps: float


@dataclass(slots=True, frozen=True)
class MobilityFrame:
    step: int
    vehicles: tuple[MobilityVehicle, ...]
    departed_ids: tuple[str, ...] = ()


class MobilityProvider(Protocol):
    def start(self) -> None: ...
    def step(self, step: int) -> MobilityFrame: ...
    def close(self) -> None: ...


class SyntheticMobilityProvider:
    def __init__(self, config: SimulationConfig):
        self.config = config
        rng = random.Random(config.seed)
        self._states: dict[str, tuple[float, float, float, float]] = {}
        for index in range(config.vehicle_count):
            speed = rng.uniform(5.0, 25.0)
            angle = rng.uniform(0.0, math.tau)
            self._states[f"veh-{index:05d}"] = (
                rng.uniform(0.0, config.area_width_m),
                rng.uniform(0.0, config.area_height_m),
                speed * math.cos(angle),
                speed * math.sin(angle),
            )

    def start(self) -> None:
        return None

    def step(self, step: int) -> MobilityFrame:
        output: list[MobilityVehicle] = []
        for vehicle_id in sorted(self._states):
            x, y, vx, vy = self._states[vehicle_id]
            x = (x + vx) % self.config.area_width_m
            y = (y + vy) % self.config.area_height_m
            self._states[vehicle_id] = (x, y, vx, vy)
            output.append(MobilityVehicle(vehicle_id, (x, y), math.hypot(vx, vy)))
        departed = tuple(sorted(self._states)) if step == 0 else ()
        return MobilityFrame(step, tuple(output), departed)

    def close(self) -> None:
        return None


class SumoMobilityProvider:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._traci = None
        self._constants = None

    def start(self) -> None:
        import sumolib
        import traci
        import traci.constants as tc

        binary = sumolib.checkBinary(self.config.sumo_binary)
        traci.start(
            [
                binary,
                "-c",
                str(self.config.scenario_config),
                "--seed",
                str(self.config.seed),
                "--no-step-log",
                "true",
                "--no-warnings",
                "true",
            ]
        )
        self._traci = traci
        self._constants = tc

    def step(self, step: int) -> MobilityFrame:
        if self._traci is None:
            raise RuntimeError("SUMO provider has not been started")
        self._traci.simulationStep()
        departed = tuple(sorted(self._traci.simulation.getDepartedIDList()))
        for vehicle_id in departed:
            self._traci.vehicle.subscribe(
                vehicle_id,
                (self._constants.VAR_POSITION, self._constants.VAR_SPEED),
            )
        subscription_results = self._traci.vehicle.getAllSubscriptionResults()
        ids = tuple(sorted(self._traci.vehicle.getIDList()))
        vehicles_list: list[MobilityVehicle] = []
        for vehicle_id in ids:
            values = subscription_results.get(vehicle_id)
            if (
                values is None
                or self._constants.VAR_POSITION not in values
                or self._constants.VAR_SPEED not in values
            ):
                position = tuple(self._traci.vehicle.getPosition(vehicle_id))
                speed = float(self._traci.vehicle.getSpeed(vehicle_id))
            else:
                position = tuple(values[self._constants.VAR_POSITION])
                speed = float(values[self._constants.VAR_SPEED])
            vehicles_list.append(MobilityVehicle(vehicle_id, position, speed))
        vehicles = tuple(vehicles_list)
        return MobilityFrame(step, vehicles, departed)

    def close(self) -> None:
        if self._traci is not None:
            self._traci.close()
            self._traci = None
            self._constants = None


class TraceCachingMobilityProvider:
    """Record a strategy-independent mobility trace once and replay it safely."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        config: SimulationConfig,
        delegate: MobilityProvider,
        cache_path: Path,
        signature: str,
    ):
        self.config = config
        self.delegate = delegate
        self.cache_path = cache_path
        self.signature = signature
        self._reader = None
        self._writer = None
        self._temporary_path: Path | None = None
        self._recorded_steps = 0
        self._replaying = False

    def _expected_header(self) -> dict:
        return {
            "format": self.FORMAT_VERSION,
            "signature": self.signature,
            "steps": self.config.steps,
        }

    def cache_is_valid(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            with gzip.open(self.cache_path, "rt", encoding="utf-8") as reader:
                return json.loads(reader.readline()) == self._expected_header()
        except (OSError, EOFError, json.JSONDecodeError):
            return False

    def start(self) -> None:
        if self.cache_is_valid():
            reader = gzip.open(self.cache_path, "rt", encoding="utf-8")
            reader.readline()
            self._reader = reader
            self._replaying = True
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary_path = self.cache_path.with_name(
            f"{self.cache_path.name}.{uuid.uuid4().hex}.tmp"
        )
        self.delegate.start()
        self._writer = gzip.open(self._temporary_path, "wt", encoding="utf-8", compresslevel=3)
        self._writer.write(
            json.dumps(
                self._expected_header(),
                separators=(",", ":"),
            )
            + "\n"
        )

    def step(self, step: int) -> MobilityFrame:
        if self._replaying:
            if self._reader is None:
                raise RuntimeError("mobility trace reader is not open")
            line = self._reader.readline()
            if not line:
                raise RuntimeError(f"mobility trace ended before step {step}")
            payload = json.loads(line)
            if payload[0] != step:
                raise RuntimeError(f"mobility trace step mismatch: expected {step}, got {payload[0]}")
            vehicles = tuple(
                MobilityVehicle(item[0], (float(item[1]), float(item[2])), float(item[3]))
                for item in payload[2]
            )
            return MobilityFrame(step, vehicles, tuple(payload[1]))

        if self._writer is None:
            raise RuntimeError("mobility trace writer is not open")
        frame = self.delegate.step(step)
        payload = [
            step,
            list(frame.departed_ids),
            [
                [vehicle.vehicle_id, vehicle.position[0], vehicle.position[1], vehicle.speed_mps]
                for vehicle in frame.vehicles
            ],
        ]
        self._writer.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._recorded_steps += 1
        return frame

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None
            return
        try:
            self.delegate.close()
        finally:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            if self._temporary_path is not None:
                if self._recorded_steps == self.config.steps:
                    self._publish_completed_trace()
                elif self._temporary_path.exists():
                    self._temporary_path.unlink()

    def _publish_completed_trace(self) -> None:
        """Publish once while allowing concurrent writers to discard duplicates."""
        if self._temporary_path is None:
            return
        for attempt in range(20):
            if self.cache_is_valid():
                self._temporary_path.unlink(missing_ok=True)
                self._temporary_path = None
                return
            try:
                self._temporary_path.replace(self.cache_path)
                self._temporary_path = None
                return
            except (PermissionError, FileExistsError):
                if attempt == 19:
                    raise
                time.sleep(0.05)


def _sumo_trace_signature(config: SimulationConfig) -> str:
    digest = hashlib.sha256()
    stable = {
        "format": TraceCachingMobilityProvider.FORMAT_VERSION,
        "steps": config.steps,
        "vehicles": config.vehicle_count,
        "seed": config.seed,
        "sumo_binary": config.sumo_binary,
    }
    digest.update(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for raw_path in (config.scenario_config, config.scenario_net):
        if raw_path:
            path = Path(raw_path)
            digest.update(str(path.resolve()).encode("utf-8"))
            digest.update(path.read_bytes())
    if config.scenario_config:
        scenario_path = Path(config.scenario_config)
        root = ET.parse(scenario_path).getroot()
        route_element = root.find("./input/route-files")
        if route_element is not None and route_element.get("value"):
            route_path = Path(route_element.get("value"))
            if not route_path.is_absolute():
                route_path = scenario_path.parent / route_path
            digest.update(str(route_path.resolve()).encode("utf-8"))
            digest.update(route_path.read_bytes())
    return digest.hexdigest()[:20]


def create_mobility(config: SimulationConfig) -> MobilityProvider:
    if config.mobility == "synthetic":
        return SyntheticMobilityProvider(config)
    signature = _sumo_trace_signature(config)
    cache_path = Path(config.route_output_dir) / "mobility-cache" / f"{signature}.jsonl.gz"
    return TraceCachingMobilityProvider(
        config,
        SumoMobilityProvider(config),
        cache_path,
        signature,
    )
