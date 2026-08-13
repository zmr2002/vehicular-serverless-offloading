from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import tomllib

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.mobility import TraceCachingMobilityProvider, create_mobility
from vehicular_offloading.routes import prepare_sumo_scenario
from vehicular_offloading.simulation import SimulationRunner


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify exactness and timing of batched candidate pricing"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    repository = config_path.parent.parent
    with config_path.open("rb") as handle:
        benchmark = tomllib.load(handle)["benchmark"]
    base_path = (config_path.parent / benchmark["base_config"]).resolve()
    base = SimulationConfig.from_toml(base_path)
    base.steps = int(args.steps or benchmark["steps"])
    base.vehicle_count = int(benchmark["vehicle_count"])
    base.seed = int(benchmark["seed"])
    base.strategy = "hybrid_stackelberg"
    base.backend = "analytical"
    base.dqn.mode = "evaluate"
    base.dqn.checkpoint_path = str(
        _discover_checkpoint(
            repository / benchmark["checkpoint_root"],
            base.vehicle_count,
        )
    )
    base.record_task_records = False
    base.record_decision_diagnostics = False
    base.minimum_free_disk_gb = float(benchmark["minimum_free_disk_gb"])
    output = (repository / benchmark["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _prepare_mobility(base)

    rows = []
    summaries = []
    pricing_text = []
    for name, batch_candidates in (("sequential", False), ("batched", True)):
        config = copy.deepcopy(base)
        config.cloud_price_batch_candidates = batch_candidates
        config.output_dir = str(output / name)
        config.validate()
        started = perf_counter()
        summary, run_dir = SimulationRunner(config).run()
        wall_s = perf_counter() - started
        timing = json.loads(
            (Path(run_dir) / "timing.json").read_text(encoding="utf-8")
        )
        row = {
            "name": name,
            "wall_clock_s": wall_s,
            "pricing_phase_s": float(
                timing["phase_seconds"]["topology_and_pricing"]
            ),
            "run_dir": run_dir,
        }
        rows.append(row)
        summaries.append(asdict(summary))
        pricing_text.append(
            (Path(run_dir) / "pricing.jsonl").read_text(encoding="utf-8")
        )
        print(
            f"DONE {name} wall={wall_s:.2f}s "
            f"pricing={row['pricing_phase_s']:.2f}s",
            flush=True,
        )

    if summaries[0] != summaries[1] or pricing_text[0] != pricing_text[1]:
        raise RuntimeError(
            "batched candidate pricing changed the scientific result"
        )
    sequential, batched = rows
    result = {
        "steps": base.steps,
        "vehicle_count": base.vehicle_count,
        "seed": base.seed,
        "scientifically_identical": True,
        "sequential": sequential,
        "batched": batched,
        "wall_speedup": sequential["wall_clock_s"] / max(
            batched["wall_clock_s"], 1e-12
        ),
        "pricing_speedup": sequential["pricing_phase_s"] / max(
            batched["pricing_phase_s"], 1e-12
        ),
    }
    (output / "benchmark-result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (output / "benchmark-summary.md").write_text(
        "\n".join(
            [
                "# Candidate-price batching benchmark",
                "",
                f"- Scientific outputs identical: {result['scientifically_identical']}",
                f"- Wall-clock speedup: {result['wall_speedup']:.3f}x",
                f"- Pricing-phase speedup: {result['pricing_speedup']:.3f}x",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"COMPLETE {output / 'benchmark-summary.md'}", flush=True)
    return 0


def _prepare_mobility(config: SimulationConfig) -> None:
    if config.mobility != "sumo":
        return
    if config.scenario_net:
        config.scenario_config = str(
            prepare_sumo_scenario(
                config.scenario_net,
                config.route_output_dir,
                config.vehicle_count,
                config.route_departure_end_s,
                config.seed,
            )
        )
        config.scenario_net = None
    mobility = create_mobility(config)
    if not isinstance(mobility, TraceCachingMobilityProvider):
        return
    if mobility.cache_is_valid():
        return
    mobility.start()
    try:
        for step in range(config.steps):
            mobility.step(step)
    finally:
        mobility.close()


def _discover_checkpoint(root: Path, vehicles: int) -> Path:
    candidates = sorted(
        root.glob(
            f"run-*/training/hybrid_stackelberg-{vehicles}/*/dqn-policy.pt"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Hybrid checkpoint not found below {root} for {vehicles} vehicles"
        )
    return candidates[0].resolve()


if __name__ == "__main__":
    raise SystemExit(main())
