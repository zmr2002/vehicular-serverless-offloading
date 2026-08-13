from __future__ import annotations

import math
import heapq
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import EnergyConfig, NetworkConfig
from .domain import OffloadAction, OffloadEstimate, ServiceQuote, Task, VehicleState


@dataclass(slots=True, frozen=True)
class _ServiceNode:
    vehicle_id: str
    position: tuple[float, float]
    axis: int
    left: _ServiceNode | None
    right: _ServiceNode | None


class ServiceSpatialIndex:
    """Static per-step spatial index with dynamic quote eligibility."""

    def __init__(self, vehicles: Mapping[str, VehicleState], target_ids: Iterable[str]):
        self.vehicles = vehicles
        ids = sorted(vehicle_id for vehicle_id in target_ids if vehicle_id in vehicles)
        self._target_ids = tuple(ids)
        self._tree: cKDTree | None = None
        self._prepared: dict[str, tuple[str, ...]] = {}
        self.root = self._build(ids, depth=0)

    def prepare_sources(
        self,
        source_ids: Iterable[str],
        radius_m: float,
        candidate_count: int = 16,
        minimum_pair_count: int = 50_000,
    ) -> None:
        """Batch-query likely targets while retaining an exact dynamic fallback."""
        if not self._target_ids or candidate_count <= 0:
            return
        sources = tuple(dict.fromkeys(source_id for source_id in source_ids if source_id in self.vehicles))
        if not sources or len(sources) * len(self._target_ids) < minimum_pair_count:
            return
        if self._tree is None:
            self._tree = cKDTree(
                np.asarray(
                    [self.vehicles[vehicle_id].position for vehicle_id in self._target_ids],
                    dtype=np.float64,
                )
            )
        count = min(candidate_count, len(self._target_ids))
        positions = np.asarray([self.vehicles[source_id].position for source_id in sources], dtype=np.float64)
        distances, indices = self._tree.query(
            positions,
            k=count,
            distance_upper_bound=radius_m,
            workers=1,
        )
        if count == 1:
            distances = np.asarray(distances).reshape(-1, 1)
            indices = np.asarray(indices).reshape(-1, 1)
        for row, source_id in enumerate(sources):
            ranked = sorted(
                (
                    (float(distance), self._target_ids[int(index)])
                    for distance, index in zip(distances[row], indices[row])
                    if math.isfinite(float(distance)) and int(index) < len(self._target_ids)
                    and self._target_ids[int(index)] != source_id
                ),
                key=lambda item: (item[0], item[1]),
            )
            self._prepared[source_id] = tuple(vehicle_id for _, vehicle_id in ranked)

    def nearest(
        self,
        source_id: str,
        eligible_ids: Mapping[str, ServiceQuote] | set[str],
        radius_m: float,
    ) -> str | None:
        if self.root is None or source_id not in self.vehicles:
            return None
        prepared = self._prepared.get(source_id)
        if prepared is not None:
            for vehicle_id in prepared:
                if vehicle_id in eligible_ids:
                    return vehicle_id
        source = self.vehicles[source_id].position
        best_id: str | None = None
        best_distance_squared = radius_m * radius_m

        def visit(node: _ServiceNode | None) -> None:
            nonlocal best_id, best_distance_squared
            if node is None:
                return
            delta_x = source[0] - node.position[0]
            delta_y = source[1] - node.position[1]
            distance_squared = delta_x * delta_x + delta_y * delta_y
            if node.vehicle_id != source_id and node.vehicle_id in eligible_ids:
                if distance_squared < best_distance_squared or (
                    distance_squared == best_distance_squared
                    and (best_id is None or node.vehicle_id < best_id)
                ):
                    best_id = node.vehicle_id
                    best_distance_squared = distance_squared
                elif distance_squared == best_distance_squared and best_id is None:
                    best_id = node.vehicle_id

            axis_delta = source[node.axis] - node.position[node.axis]
            near, far = (node.left, node.right) if axis_delta <= 0 else (node.right, node.left)
            visit(near)
            if axis_delta * axis_delta <= best_distance_squared:
                visit(far)

        visit(self.root)
        return best_id

    def _build(self, vehicle_ids: list[str], depth: int) -> _ServiceNode | None:
        if not vehicle_ids:
            return None
        axis = depth % 2
        other_axis = 1 - axis
        vehicle_ids.sort(
            key=lambda vehicle_id: (
                self.vehicles[vehicle_id].position[axis],
                self.vehicles[vehicle_id].position[other_axis],
                vehicle_id,
            )
        )
        middle = len(vehicle_ids) // 2
        vehicle_id = vehicle_ids[middle]
        return _ServiceNode(
            vehicle_id=vehicle_id,
            position=self.vehicles[vehicle_id].position,
            axis=axis,
            left=self._build(vehicle_ids[:middle], depth + 1),
            right=self._build(vehicle_ids[middle + 1 :], depth + 1),
        )


class LazyNeighborGraph(Mapping[str, tuple[str, ...]]):
    """Exact radius graph that materializes only nodes requested by a policy."""

    def __init__(self, vehicles: Mapping[str, VehicleState], radius_m: float, config: NetworkConfig):
        self.vehicles = vehicles
        self.radius_m = radius_m
        self.config = config
        self._vehicle_ids = tuple(sorted(vehicles))
        self._buckets: dict[tuple[int, int], list[str]] = {}
        self._neighbors: dict[str, tuple[str, ...]] = {}
        self._throughputs: dict[str, dict[str, float]] = {}
        self._reachability_key: tuple[frozenset[str], int] | None = None
        self._reachable_nodes: set[str] = set()
        for vehicle_id in self._vehicle_ids:
            position = vehicles[vehicle_id].position
            cell = (math.floor(position[0] / radius_m), math.floor(position[1] / radius_m))
            self._buckets.setdefault(cell, []).append(vehicle_id)
        for bucket in self._buckets.values():
            bucket.sort()

    def __getitem__(self, vehicle_id: str) -> tuple[str, ...]:
        self._ensure(vehicle_id)
        return self._neighbors[vehicle_id]

    def __iter__(self):
        return iter(self._vehicle_ids)

    def __len__(self) -> int:
        return len(self._vehicle_ids)

    def throughputs(self, vehicle_id: str) -> dict[str, float]:
        self._ensure(vehicle_id)
        return self._throughputs[vehicle_id]

    def neighbor_count(self, vehicle_id: str, limit: int | None = None) -> int:
        """Count radius neighbors without materializing sorted link metadata."""
        if vehicle_id not in self.vehicles:
            return 0
        source_position = self.vehicles[vehicle_id].position
        source_cell = (
            math.floor(source_position[0] / self.radius_m),
            math.floor(source_position[1] / self.radius_m),
        )
        radius_squared = self.radius_m * self.radius_m
        count = 0
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for neighbor_id in self._buckets.get(
                    (source_cell[0] + offset_x, source_cell[1] + offset_y), ()
                ):
                    if neighbor_id == vehicle_id:
                        continue
                    neighbor_position = self.vehicles[neighbor_id].position
                    delta_x = source_position[0] - neighbor_position[0]
                    delta_y = source_position[1] - neighbor_position[1]
                    if delta_x * delta_x + delta_y * delta_y <= radius_squared:
                        count += 1
                        if limit is not None and count >= limit:
                            return count
        return count

    def can_reach_any(self, source_id: str, target_ids, max_hops: int) -> bool:
        key = (frozenset(target_ids), max_hops)
        if key != self._reachability_key:
            reached = set(key[0])
            frontier = set(key[0])
            for _ in range(max_hops):
                next_frontier: set[str] = set()
                for vehicle_id in sorted(frontier):
                    next_frontier.update(self[vehicle_id])
                next_frontier.difference_update(reached)
                if not next_frontier:
                    break
                reached.update(next_frontier)
                frontier = next_frontier
            self._reachability_key = key
            self._reachable_nodes = reached
        return source_id in self._reachable_nodes

    @property
    def materialized_nodes(self) -> int:
        return len(self._neighbors)

    def _ensure(self, vehicle_id: str) -> None:
        if vehicle_id in self._neighbors:
            return
        if vehicle_id not in self.vehicles:
            raise KeyError(vehicle_id)
        source_position = self.vehicles[vehicle_id].position
        source_cell = (
            math.floor(source_position[0] / self.radius_m),
            math.floor(source_position[1] / self.radius_m),
        )
        ranked: list[tuple[float, str]] = []
        links: dict[str, float] = {}
        radius_squared = self.radius_m * self.radius_m
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                bucket = self._buckets.get((source_cell[0] + offset_x, source_cell[1] + offset_y), ())
                for neighbor_id in bucket:
                    if neighbor_id == vehicle_id:
                        continue
                    neighbor_position = self.vehicles[neighbor_id].position
                    delta_x = source_position[0] - neighbor_position[0]
                    delta_y = source_position[1] - neighbor_position[1]
                    distance_squared = delta_x * delta_x + delta_y * delta_y
                    if distance_squared > radius_squared:
                        continue
                    throughput = effective_v2v_throughput_mbps(
                        math.sqrt(distance_squared), self.config
                    )
                    links[neighbor_id] = throughput
                    ranked.append((-throughput, neighbor_id))
        ranked.sort()
        self._neighbors[vehicle_id] = tuple(neighbor_id for _, neighbor_id in ranked)
        self._throughputs[vehicle_id] = links


class LazyLinkThroughputMap(Mapping[str, Mapping[str, float]]):
    def __init__(self, graph: LazyNeighborGraph):
        self.graph = graph

    def __getitem__(self, vehicle_id: str) -> Mapping[str, float]:
        return self.graph.throughputs(vehicle_id)

    def __iter__(self):
        return iter(self.graph)

    def __len__(self) -> int:
        return len(self.graph)


class CompactNeighborGraph(Mapping[str, tuple[str, ...]]):
    """Exact radius graph stored as integer-indexed contiguous CSR arrays."""

    def __init__(
        self,
        vehicles: Mapping[str, VehicleState],
        radius_m: float,
        config: NetworkConfig,
    ):
        if radius_m <= 0:
            raise ValueError("radius_m must be positive")
        self.vehicles = vehicles
        self.radius_m = radius_m
        self.config = config
        self.vehicle_ids = tuple(sorted(vehicles))
        self.id_to_index = {
            vehicle_id: index for index, vehicle_id in enumerate(self.vehicle_ids)
        }
        self.positions = np.ascontiguousarray(
            [vehicles[vehicle_id].position for vehicle_id in self.vehicle_ids],
            dtype=np.float64,
        ).reshape(-1, 2)
        count = len(self.vehicle_ids)
        if count < 2:
            self.indptr = np.zeros(count + 1, dtype=np.int64)
            self.indices = np.empty(0, dtype=np.int32)
            self.link_coefficients_s_per_mb = np.empty(0, dtype=np.float64)
            return

        pairs = cKDTree(self.positions).query_pairs(
            radius_m,
            output_type="ndarray",
        )
        if pairs.size == 0:
            self.indptr = np.zeros(count + 1, dtype=np.int64)
            self.indices = np.empty(0, dtype=np.int32)
            self.link_coefficients_s_per_mb = np.empty(0, dtype=np.float64)
            return
        pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        deltas = self.positions[pairs[:, 0]] - self.positions[pairs[:, 1]]
        distances = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
        throughput = _effective_v2v_throughput_batch(distances, config)
        coefficients = 8.0 / throughput
        sources = np.concatenate((pairs[:, 0], pairs[:, 1]))
        targets = np.concatenate((pairs[:, 1], pairs[:, 0]))
        edge_coefficients = np.concatenate((coefficients, coefficients))
        order = np.lexsort((targets, edge_coefficients, sources))
        sources = sources[order]
        self.indices = np.ascontiguousarray(targets[order], dtype=np.int32)
        self.link_coefficients_s_per_mb = np.ascontiguousarray(
            edge_coefficients[order],
            dtype=np.float64,
        )
        self.indptr = np.zeros(count + 1, dtype=np.int64)
        np.add.at(self.indptr, sources + 1, 1)
        np.cumsum(self.indptr, out=self.indptr)

    def __getitem__(self, vehicle_id: str) -> tuple[str, ...]:
        index = self.id_to_index[vehicle_id]
        start, end = int(self.indptr[index]), int(self.indptr[index + 1])
        return tuple(self.vehicle_ids[int(item)] for item in self.indices[start:end])

    def __iter__(self):
        return iter(self.vehicle_ids)

    def __len__(self) -> int:
        return len(self.vehicle_ids)

    def neighbor_count(self, vehicle_id: str, limit: int | None = None) -> int:
        index = self.id_to_index.get(vehicle_id)
        if index is None:
            return 0
        count = int(self.indptr[index + 1] - self.indptr[index])
        return min(count, limit) if limit is not None else count

    def throughputs(self, vehicle_id: str) -> dict[str, float]:
        index = self.id_to_index[vehicle_id]
        start, end = int(self.indptr[index]), int(self.indptr[index + 1])
        return {
            self.vehicle_ids[int(target)]: 8.0 / float(coefficient)
            for target, coefficient in zip(
                self.indices[start:end],
                self.link_coefficients_s_per_mb[start:end],
            )
        }


class CompactLinkThroughputMap(Mapping[str, Mapping[str, float]]):
    def __init__(self, graph: CompactNeighborGraph):
        self.graph = graph

    def __getitem__(self, vehicle_id: str) -> Mapping[str, float]:
        return self.graph.throughputs(vehicle_id)

    def __iter__(self):
        return iter(self.graph)

    def __len__(self) -> int:
        return len(self.graph)


@dataclass(slots=True, frozen=True)
class _ReversePathLabel:
    coefficient_s_per_mb: float
    queue_delay_s: float
    target_index: int
    path: tuple[int, ...]


class ReverseParetoV2VIndex:
    """Exact bounded-hop reverse index shared by every task in one step."""

    def __init__(
        self,
        graph: CompactNeighborGraph,
        vehicles: Mapping[str, VehicleState],
        quotes: Mapping[str, ServiceQuote],
        max_hops: int,
        uniform_service_compute_hz: float,
        energy: EnergyConfig,
    ):
        self.graph = graph
        self.vehicles = vehicles
        self.quotes = quotes
        self.max_hops = max_hops
        self.uniform_service_compute_hz = uniform_service_compute_hz
        self.energy = energy
        self.labels: list[list[_ReversePathLabel]] = [
            [] for _ in graph.vehicle_ids
        ]
        if not quotes or max_hops <= 0:
            return
        frontier: dict[int, list[_ReversePathLabel]] = {}
        for target_id in sorted(quotes):
            target_index = graph.id_to_index.get(target_id)
            if target_index is None:
                continue
            quote = quotes[target_id]
            frontier.setdefault(target_index, []).append(
                _ReversePathLabel(
                    coefficient_s_per_mb=0.0,
                    queue_delay_s=(
                        vehicles[target_id].workload_cycles / quote.compute_hz
                    ),
                    target_index=target_index,
                    path=(target_index,),
                )
            )

        for _hop in range(1, max_hops + 1):
            candidates: dict[int, list[_ReversePathLabel]] = {}
            for current_index, current_labels in frontier.items():
                start = int(graph.indptr[current_index])
                end = int(graph.indptr[current_index + 1])
                for edge_offset in range(start, end):
                    source_index = int(graph.indices[edge_offset])
                    edge_coefficient = float(
                        graph.link_coefficients_s_per_mb[edge_offset]
                    )
                    bucket = candidates.setdefault(source_index, [])
                    for label in current_labels:
                        if source_index in label.path:
                            continue
                        bucket.append(
                            _ReversePathLabel(
                                coefficient_s_per_mb=(
                                    edge_coefficient
                                    + label.coefficient_s_per_mb
                                ),
                                queue_delay_s=label.queue_delay_s,
                                target_index=label.target_index,
                                path=(source_index,) + label.path,
                            )
                        )
            if not candidates:
                break
            frontier = {
                source_index: _pareto_path_labels(source_labels)
                for source_index, source_labels in candidates.items()
            }
            for source_index, source_labels in frontier.items():
                self.labels[source_index] = _pareto_path_labels(
                    self.labels[source_index] + source_labels
                )

    def estimate(self, source_id: str, task: Task) -> OffloadEstimate:
        source_index = self.graph.id_to_index.get(source_id)
        if source_index is None:
            return _infeasible_v2v()
        labels = self.labels[source_index]
        if not labels:
            return _infeasible_v2v()
        compute_delay = task.compute_cycles / self.uniform_service_compute_hz
        label = min(
            labels,
            key=lambda item: (
                task.data_size_mb * item.coefficient_s_per_mb
                + item.queue_delay_s
                + compute_delay,
                len(item.path),
                item.path,
            ),
        )
        transmission_delay = task.data_size_mb * label.coefficient_s_per_mb
        target_id = self.graph.vehicle_ids[label.target_index]
        quote = self.quotes[target_id]
        return OffloadEstimate(
            action=OffloadAction.V2V,
            delay_s=transmission_delay + label.queue_delay_s + compute_delay,
            energy_j=(
                transmission_delay * self.energy.v2v_transmit_power_w
                + compute_delay * self.energy.service_compute_power_w
            ),
            payment=quote.price * (task.compute_cycles / 1e9),
            target_id=target_id,
            path=tuple(self.graph.vehicle_ids[index] for index in label.path),
        )


def _pareto_path_labels(
    labels: list[_ReversePathLabel],
) -> list[_ReversePathLabel]:
    """Remove labels that can never win for any positive task data size."""
    if not labels:
        return []
    ordered = sorted(
        labels,
        key=lambda item: (
            item.coefficient_s_per_mb,
            item.queue_delay_s,
            len(item.path),
            item.path,
        ),
    )
    result: list[_ReversePathLabel] = []
    best_queue = math.inf
    for label in ordered:
        if label.queue_delay_s < best_queue:
            result.append(label)
            best_queue = label.queue_delay_s
    return result


def _infeasible_v2v() -> OffloadEstimate:
    return OffloadEstimate(
        OffloadAction.V2V,
        math.inf,
        math.inf,
        math.inf,
        feasible=False,
    )


def _effective_v2v_throughput_batch(
    distances: np.ndarray,
    config: NetworkConfig,
) -> np.ndarray:
    base = config.v2v_channel_bandwidth_mhz * config.v2v_resource_efficiency
    if config.channel_capacity_model == "distance_only":
        return np.full(distances.shape, base, dtype=np.float64)
    bounded = np.maximum(distances, config.minimum_channel_distance_m)
    snr_db = config.v2v_reference_snr_db - (
        10.0
        * config.v2v_path_loss_exponent
        * np.log10(bounded / config.v2v_reference_distance_m)
    )
    efficiency = np.log2(1.0 + np.power(10.0, snr_db / 10.0))
    return base * np.minimum(
        config.v2v_max_spectral_efficiency_bps_hz,
        efficiency,
    )


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def transmission_delay_s(
    data_size_mb: float,
    distance: float,
    action: OffloadAction,
    config: NetworkConfig,
) -> float:
    if data_size_mb < 0 or distance < 0:
        raise ValueError("data size and distance must be non-negative")
    if action == OffloadAction.V2V:
        throughput_mbps = effective_v2v_throughput_mbps(distance, config)
    elif action == OffloadAction.V2I:
        throughput_mbps = effective_v2i_throughput_mbps(distance, config)
    else:
        return 0.0
    return data_size_mb * 8.0 / throughput_mbps


def effective_v2v_throughput_mbps(distance: float, config: NetworkConfig) -> float:
    """Return V2V throughput using MHz * bit/s/Hz = Mbit/s."""
    return (
        config.v2v_channel_bandwidth_mhz
        * config.v2v_resource_efficiency
        * spectral_efficiency_bps_hz(distance, OffloadAction.V2V, config)
    )


def effective_v2i_throughput_mbps(distance: float, config: NetworkConfig) -> float:
    """Return V2I throughput using separately configured radio resources."""
    return (
        config.v2i_channel_bandwidth_mhz
        * config.v2i_resource_efficiency
        * spectral_efficiency_bps_hz(distance, OffloadAction.V2I, config)
    )


def spectral_efficiency_bps_hz(
    distance: float,
    action: OffloadAction,
    config: NetworkConfig,
) -> float:
    """Return the configured distance-only or Shannon spectral-efficiency factor."""
    if config.channel_capacity_model == "distance_only":
        return 1.0
    bounded_distance = max(distance, config.minimum_channel_distance_m)
    if action == OffloadAction.V2V:
        exponent = config.v2v_path_loss_exponent
        reference_snr_db = config.v2v_reference_snr_db
        reference_distance_m = config.v2v_reference_distance_m
        maximum_efficiency = config.v2v_max_spectral_efficiency_bps_hz
    else:
        exponent = config.v2i_path_loss_exponent
        reference_snr_db = config.v2i_reference_snr_db
        reference_distance_m = config.v2i_reference_distance_m
        maximum_efficiency = config.v2i_max_spectral_efficiency_bps_hz
    # Log-distance path loss: received SNR loses 10*n*log10(d/d0) dB.
    snr_db = reference_snr_db - 10.0 * exponent * math.log10(
        bounded_distance / reference_distance_m
    )
    spectral_efficiency = math.log2(1.0 + 10.0 ** (snr_db / 10.0))
    return min(maximum_efficiency, spectral_efficiency)


def cloud_queue_delay_s(queue_length: int, config: NetworkConfig) -> float:
    queue_length = max(0, queue_length)
    return (
        math.log1p(queue_length) * config.queue_delay_coefficient
        + config.queue_delay_extra_s * max(0, queue_length - config.queue_delay_threshold)
    )


def best_v2v_path(
    source_id: str,
    vehicles: dict[str, VehicleState],
    quotes: Iterable[ServiceQuote] | Mapping[str, ServiceQuote],
    task: Task,
    config: NetworkConfig,
    adjacency: Mapping[str, tuple[str, ...]] | None = None,
    throughput_by_link: Mapping[str, Mapping[str, float]] | None = None,
    uniform_service_compute_hz: float | None = None,
    service_spatial_index: ServiceSpatialIndex | None = None,
    energy_config: EnergyConfig | None = None,
) -> OffloadEstimate:
    energy = energy_config or EnergyConfig()
    quote_by_id = quotes if isinstance(quotes, Mapping) else {quote.vehicle_id: quote for quote in quotes}
    if source_id not in vehicles or not quote_by_id:
        return OffloadEstimate(OffloadAction.V2V, math.inf, math.inf, math.inf, feasible=False)

    best_candidate: OffloadEstimate | None = None
    if uniform_service_compute_hz is not None and service_spatial_index is not None:
        target_id = service_spatial_index.nearest(
            source_id, quote_by_id, config.neighbor_radius_m
        )
        if target_id is not None:
            quote = quote_by_id[target_id]
            link_distance = distance_m(vehicles[source_id].position, vehicles[target_id].position)
            link_delay = transmission_delay_s(
                task.data_size_mb, link_distance, OffloadAction.V2V, config
            )
            queue_delay = vehicles[target_id].workload_cycles / quote.compute_hz
            compute_delay = task.compute_cycles / quote.compute_hz
            best_candidate = OffloadEstimate(
                action=OffloadAction.V2V,
                delay_s=link_delay + queue_delay + compute_delay,
                energy_j=(
                    link_delay * energy.v2v_transmit_power_w
                    + compute_delay * energy.service_compute_power_w
                ),
                payment=quote.price * (task.compute_cycles / 1e9),
                target_id=target_id,
                path=(source_id, target_id),
            )
            if _cannot_improve_at_deeper_hop(
                best_candidate.delay_s,
                quote_by_id,
                task,
                config,
                2,
                uniform_service_compute_hz,
            ):
                return best_candidate
    source_neighbors = adjacency.get(source_id, ()) if adjacency is not None else tuple(sorted(vehicles))
    source_throughputs = throughput_by_link[source_id] if throughput_by_link is not None else None

    # The simulator gives every service vehicle the same configured CPU rate,
    # but their persistent queues can differ.  Consequently every reachable
    # service must be compared by transmission plus queueing delay; the nearest
    # or highest-bandwidth service is only an initial upper bound.
    if uniform_service_compute_hz is not None and source_throughputs is not None:
        direct_neighbors = (
            neighbor_id for neighbor_id in source_neighbors if neighbor_id in quote_by_id
        )
    else:
        direct_neighbors = iter(source_neighbors)

    for neighbor_id in direct_neighbors:
        quote = quote_by_id.get(neighbor_id)
        if quote is None:
            continue
        if source_throughputs is None:
            link_distance = distance_m(vehicles[source_id].position, vehicles[neighbor_id].position)
            if link_distance > config.neighbor_radius_m:
                continue
            link_delay = transmission_delay_s(task.data_size_mb, link_distance, OffloadAction.V2V, config)
        else:
            link_delay = task.data_size_mb * 8.0 / source_throughputs[neighbor_id]
        compute_delay = task.compute_cycles / quote.compute_hz
        if (
            best_candidate is not None
            and uniform_service_compute_hz is not None
            and source_throughputs is not None
            and link_delay + compute_delay > best_candidate.delay_s
        ):
            # Direct neighbors are ordered by non-increasing throughput. Even
            # an empty queue cannot make this or any later service faster.
            break
        queue_delay = vehicles[neighbor_id].workload_cycles / quote.compute_hz
        candidate = OffloadEstimate(
            action=OffloadAction.V2V,
            delay_s=link_delay + queue_delay + compute_delay,
            energy_j=(
                link_delay * energy.v2v_transmit_power_w
                + compute_delay * energy.service_compute_power_w
            ),
            payment=quote.price * (task.compute_cycles / 1e9),
            target_id=neighbor_id,
            path=(source_id, neighbor_id),
        )
        if best_candidate is None or (candidate.delay_s, len(candidate.path), candidate.path) < (
            best_candidate.delay_s,
            len(best_candidate.path),
            best_candidate.path,
        ):
            best_candidate = candidate
    if best_candidate and _cannot_improve_at_deeper_hop(
        best_candidate.delay_s,
        quote_by_id,
        task,
        config,
        2,
        uniform_service_compute_hz,
    ):
        return best_candidate
    if config.max_hops == 1:
        return best_candidate or OffloadEstimate(
            OffloadAction.V2V, math.inf, math.inf, math.inf, feasible=False
        )

    if uniform_service_compute_hz is not None and throughput_by_link is not None:
        if isinstance(adjacency, LazyNeighborGraph) and not adjacency.can_reach_any(
            source_id, quote_by_id, config.max_hops
        ):
            return OffloadEstimate(OffloadAction.V2V, math.inf, math.inf, math.inf, feasible=False)
        return _best_uniform_v2v_path(
            source_id,
            vehicles,
            quote_by_id,
            task,
            config,
            adjacency,
            throughput_by_link,
            energy,
            uniform_service_compute_hz,
        )

    # Dynamic programming by exact hop count. Positive link weights mean that
    # retaining only the best deterministic path to a node at each depth is
    # sufficient; this avoids repeatedly re-enqueuing the same state in dense
    # traffic graphs.
    frontier: dict[str, tuple[float, tuple[str, ...]]] = {}
    for neighbor_id in source_neighbors:
        if source_throughputs is None:
            link_distance = distance_m(vehicles[source_id].position, vehicles[neighbor_id].position)
            if link_distance > config.neighbor_radius_m:
                continue
            link_delay = transmission_delay_s(task.data_size_mb, link_distance, OffloadAction.V2V, config)
        else:
            link_delay = task.data_size_mb * 8.0 / source_throughputs[neighbor_id]
        frontier[neighbor_id] = (link_delay, (source_id, neighbor_id))

    for _hop in range(2, config.max_hops + 1):
        next_frontier: dict[str, tuple[float, tuple[str, ...]]] = {}
        for current, (transmission, path) in frontier.items():
            current_vehicle = vehicles[current]
            neighbor_ids = adjacency.get(current, ()) if adjacency is not None else tuple(sorted(vehicles))
            cached_throughputs = throughput_by_link[current] if throughput_by_link is not None else None
            for neighbor_id in neighbor_ids:
                if neighbor_id in path:
                    continue
                if cached_throughputs is None:
                    neighbor = vehicles[neighbor_id]
                    link_distance = distance_m(current_vehicle.position, neighbor.position)
                    if link_distance > config.neighbor_radius_m:
                        continue
                    link_delay = transmission_delay_s(task.data_size_mb, link_distance, OffloadAction.V2V, config)
                else:
                    throughput = cached_throughputs[neighbor_id]
                    link_delay = task.data_size_mb * 8.0 / throughput
                candidate = (transmission + link_delay, path + (neighbor_id,))
                previous = next_frontier.get(neighbor_id)
                if previous is None or candidate < previous:
                    next_frontier[neighbor_id] = candidate
        frontier = next_frontier
        for current, (transmission, path) in frontier.items():
            if current not in quote_by_id:
                continue
            quote = quote_by_id[current]
            queue_delay = vehicles[current].workload_cycles / quote.compute_hz
            compute_delay = task.compute_cycles / quote.compute_hz
            candidate = OffloadEstimate(
                action=OffloadAction.V2V,
                delay_s=transmission + queue_delay + compute_delay,
                energy_j=(
                    transmission * energy.v2v_transmit_power_w
                    + compute_delay * energy.service_compute_power_w
                ),
                payment=quote.price * (task.compute_cycles / 1e9),
                target_id=current,
                path=path,
            )
            if best_candidate is None or (candidate.delay_s, len(candidate.path), candidate.path) < (
                best_candidate.delay_s,
                len(best_candidate.path),
                best_candidate.path,
            ):
                best_candidate = candidate
        if best_candidate and _cannot_improve_at_deeper_hop(
            best_candidate.delay_s,
            quote_by_id,
            task,
            config,
            _hop + 1,
            uniform_service_compute_hz,
        ):
            break
        if not frontier:
            break

    if best_candidate is None:
        return OffloadEstimate(OffloadAction.V2V, math.inf, math.inf, math.inf, feasible=False)
    return best_candidate


def _best_uniform_v2v_path(
    source_id: str,
    vehicles: Mapping[str, VehicleState],
    quotes: Mapping[str, ServiceQuote],
    task: Task,
    config: NetworkConfig,
    adjacency: Mapping[str, tuple[str, ...]],
    throughput_by_link: Mapping[str, Mapping[str, float]],
    energy: EnergyConfig,
    uniform_service_compute_hz: float,
) -> OffloadEstimate:
    """Find the exact bounded-hop route without expanding every dense layer."""
    start_path = (source_id,)
    frontier: list[tuple[float, int, tuple[str, ...], str]] = [(0.0, 0, start_path, source_id)]
    best_by_state: dict[tuple[str, int], tuple[float, tuple[str, ...]]] = {
        (source_id, 0): (0.0, start_path)
    }
    best_candidate: OffloadEstimate | None = None
    compute_delay = task.compute_cycles / uniform_service_compute_hz
    while frontier:
        transmission, hops, path, current = heapq.heappop(frontier)
        if best_by_state.get((current, hops)) != (transmission, path):
            continue
        if best_candidate is not None and transmission + compute_delay > best_candidate.delay_s:
            break
        if hops > 0 and current in quotes:
            quote = quotes[current]
            queue_delay = vehicles[current].workload_cycles / quote.compute_hz
            service_compute_delay = task.compute_cycles / quote.compute_hz
            candidate = OffloadEstimate(
                action=OffloadAction.V2V,
                delay_s=transmission + queue_delay + service_compute_delay,
                energy_j=(
                    transmission * energy.v2v_transmit_power_w
                    + service_compute_delay * energy.service_compute_power_w
                ),
                payment=quote.price * (task.compute_cycles / 1e9),
                target_id=current,
                path=path,
            )
            if best_candidate is None or (
                candidate.delay_s,
                len(candidate.path),
                candidate.path,
            ) < (
                best_candidate.delay_s,
                len(best_candidate.path),
                best_candidate.path,
            ):
                best_candidate = candidate
        if hops == config.max_hops:
            continue
        cached_throughputs = throughput_by_link[current]
        for neighbor_id in adjacency.get(current, ()):
            if neighbor_id in path:
                continue
            link_delay = task.data_size_mb * 8.0 / cached_throughputs[neighbor_id]
            next_path = path + (neighbor_id,)
            candidate = (transmission + link_delay, next_path)
            state = (neighbor_id, hops + 1)
            previous = best_by_state.get(state)
            if previous is None or candidate < previous:
                best_by_state[state] = candidate
                heapq.heappush(frontier, (candidate[0], hops + 1, candidate[1], neighbor_id))
    return best_candidate or OffloadEstimate(
        OffloadAction.V2V, math.inf, math.inf, math.inf, feasible=False
    )


def _cannot_improve_at_deeper_hop(
    best_delay_s: float,
    quotes: Mapping[str, ServiceQuote],
    task: Task,
    config: NetworkConfig,
    next_hop_count: int,
    maximum_compute_hz: float | None = None,
) -> bool:
    if next_hop_count > config.max_hops:
        return True
    maximum_throughput = (
        config.v2v_channel_bandwidth_mhz * config.v2v_resource_efficiency
    )
    if config.channel_capacity_model == "shannon":
        maximum_throughput *= config.v2v_max_spectral_efficiency_bps_hz
    minimum_link_delay = task.data_size_mb * 8.0 / maximum_throughput
    if maximum_compute_hz is None:
        maximum_compute_hz = max(quote.compute_hz for quote in quotes.values())
    lower_bound = next_hop_count * minimum_link_delay + task.compute_cycles / maximum_compute_hz
    # Equal latency is resolved in favor of fewer hops, then path ID. Therefore
    # a deeper route cannot improve an equal lower bound.
    return best_delay_s <= lower_bound


def neighbor_count(vehicle_id: str, vehicles: dict[str, VehicleState], radius_m: float) -> int:
    source = vehicles[vehicle_id]
    return sum(
        1
        for other_id, other in vehicles.items()
        if other_id != vehicle_id and distance_m(source.position, other.position) <= radius_m
    )


def build_neighbor_map(
    vehicles: dict[str, VehicleState], radius_m: float
) -> Mapping[str, tuple[str, ...]]:
    graph, _ = build_neighbor_map_with_throughput(
        vehicles, radius_m, NetworkConfig(neighbor_radius_m=radius_m)
    )
    return graph


def build_neighbor_map_with_throughput(
    vehicles: dict[str, VehicleState],
    radius_m: float,
    config: NetworkConfig,
) -> tuple[Mapping[str, tuple[str, ...]], Mapping[str, Mapping[str, float]]]:
    """Build a deterministic, lazily materialized radius graph."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    graph = LazyNeighborGraph(vehicles, radius_m, config)
    return graph, LazyLinkThroughputMap(graph)


def build_compact_neighbor_graph(
    vehicles: Mapping[str, VehicleState],
    radius_m: float,
    config: NetworkConfig,
) -> CompactNeighborGraph:
    """Build one exact integer-indexed graph for all decisions in a step."""
    return CompactNeighborGraph(vehicles, radius_m, config)


def should_build_reverse_pareto_index(
    vehicle_count: int,
    service_count: int,
    task_count: int,
) -> bool:
    """Select reverse search only when sparse targets amortize its labels."""
    if vehicle_count <= 0 or service_count <= 0:
        return False
    return task_count >= 64 and service_count / vehicle_count <= 0.10
