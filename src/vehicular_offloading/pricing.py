from __future__ import annotations

import math
from dataclasses import dataclass

from .domain import ServiceQuote, VehicleState


@dataclass(slots=True, frozen=True)
class LeaderPriceEvaluation:
    price: float
    predicted_cloud_share: float
    predicted_cloud_cycles: float
    predicted_cloud_requests: float
    predicted_late_share: float
    predicted_request_share: float
    revenue_score: float
    capacity_violation: float
    timeout_violation: float
    capacity_penalty: float
    timeout_penalty: float
    leader_score: float


def cloud_price_candidates(
    minimum_price: float,
    maximum_price: float,
    candidate_count: int,
    anchor_prices: tuple[float, ...] = (),
) -> tuple[float, ...]:
    """Return a deterministic candidate set including important prior prices."""
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    step = (maximum_price - minimum_price) / (candidate_count - 1)
    values = {
        min(max(minimum_price + index * step, minimum_price), maximum_price)
        for index in range(candidate_count)
    }
    values.update(
        min(max(float(price), minimum_price), maximum_price)
        for price in anchor_prices
    )
    return tuple(sorted(values))


def evaluate_cloud_leader_response(
    price: float,
    predicted_cloud_cycles: float,
    total_cycles: float,
    predicted_late_tasks: float,
    task_count: int,
    target_offload_ratio: float,
    _base_price: float,
    maximum_price: float,
    capacity_weight: float,
    timeout_weight: float,
    predicted_cloud_requests: float | None = None,
    late_tolerance: float = 0.0,
) -> LeaderPriceEvaluation:
    """Evaluate constrained, revenue-oriented leader utility.

    Revenue follows the thesis objective p*N_cloud. Capacity and late-admission
    terms are positive-part constraint violations, represented through a
    Lagrangian relaxation rather than a system-success objective.
    """
    cloud_share = predicted_cloud_cycles / max(total_cycles, 1.0)
    cloud_requests = (
        cloud_share * task_count
        if predicted_cloud_requests is None
        else predicted_cloud_requests
    )
    request_share = cloud_requests / max(task_count, 1)
    late_share = predicted_late_tasks / max(task_count, 1)
    revenue_score = (
        max(price, 0.0)
        / max(maximum_price, 1e-12)
        * request_share
    )
    capacity_violation = max(
        0.0,
        max(cloud_share, request_share) - target_offload_ratio,
    ) / max(target_offload_ratio, 1e-12)
    timeout_violation = max(0.0, late_share - late_tolerance)
    capacity_penalty = capacity_weight * capacity_violation
    timeout_penalty = timeout_weight * timeout_violation
    return LeaderPriceEvaluation(
        price=price,
        predicted_cloud_share=cloud_share,
        predicted_cloud_cycles=predicted_cloud_cycles,
        predicted_cloud_requests=cloud_requests,
        predicted_late_share=late_share,
        predicted_request_share=request_share,
        revenue_score=revenue_score,
        capacity_violation=capacity_violation,
        timeout_violation=timeout_violation,
        capacity_penalty=capacity_penalty,
        timeout_penalty=timeout_penalty,
        leader_score=revenue_score - capacity_penalty - timeout_penalty,
    )


def cloud_price(base_price: float, queue_length: int, queue_threshold: int) -> float:
    load_ratio = min(max(queue_length, 0) / max(queue_threshold, 1), 2.0)
    return base_price * (1.0 + 0.5 * load_ratio)


def cloud_leader_price(
    previous_price: float,
    observed_offload_ratio: float,
    base_price: float,
    target_offload_ratio: float,
    demand_sensitivity: float,
    smoothing: float,
    queue_length: int,
    queue_threshold: int,
    capacity_ratio: float,
    capacity_target: float,
    capacity_weight: float,
    minimum_price: float,
    maximum_price: float,
) -> float:
    """Return a bounded, capacity-aware leader best response.

    The thesis specifies cloud revenue and the follower quote equation but not
    a closed demand curve.  Exponential demand supplies that missing closure:
    1/alpha is its monopoly price and -log(target)/alpha is the minimum price
    that clears demand at the configured capacity target.  Their maximum is
    the constrained leader optimum.  The previous-step observation and queue
    pressure then form a deterministic, smoothed capacity response.
    """
    observed = min(max(observed_offload_ratio, 0.0), 1.0)
    target = min(max(target_offload_ratio, 1e-9), 1.0)
    monopoly_price = 1.0 / demand_sensitivity
    target_price = -math.log(target) / demand_sensitivity
    demand_correction = (observed - target) / demand_sensitivity
    queue_pressure = min(max(queue_length, 0) / max(queue_threshold, 1), 2.0)
    compute_pressure = (
        max(capacity_ratio, 0.0) / max(capacity_target, 1e-9)
    ) ** 2
    congestion_surcharge = (
        base_price * capacity_weight * (queue_pressure + compute_pressure)
    )
    best_response = (
        max(monopoly_price, target_price)
        + demand_correction
        + congestion_surcharge
    )
    updated = (1.0 - smoothing) * previous_price + smoothing * best_response
    return min(max(updated, minimum_price), maximum_price)


def service_quote(
    vehicle: VehicleState,
    cloud_unit_price: float,
    sensitivity: float,
    min_energy: float,
    max_queue: int,
    anticipated_demand_ratio: float = 0.0,
    demand_price_weight: float = 0.0,
) -> ServiceQuote | None:
    if vehicle.energy_level < min_energy or vehicle.queue_length >= max_queue:
        return None
    utilization = min(max(vehicle.queue_length / max(max_queue, 1), 0.0), 1.0)
    anticipated_utilization = min(
        1.0,
        utilization
        + demand_price_weight * min(max(anticipated_demand_ratio, 0.0), 2.0),
    )
    price = cloud_unit_price * (
        1.0 - sensitivity * (1.0 - anticipated_utilization)
    )
    marginal_cost = 0.02 + 0.03 * utilization + 0.02 * (1.0 - vehicle.energy_level)
    utility = price - marginal_cost
    if utility < 0.0:
        return None
    return ServiceQuote(vehicle.vehicle_id, price, vehicle.compute_hz, utility)
