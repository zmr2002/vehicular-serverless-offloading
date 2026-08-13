from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter, process_time
import tomllib

from vehicular_offloading.config import SimulationConfig
from vehicular_offloading.simulation import SimulationRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate capacity-aware Hybrid policies")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    sweep = raw["sweep"]
    profiles = raw["profile"]
    repository = config_path.parent.parent
    base = SimulationConfig.from_toml((config_path.parent / sweep["base_config"]).resolve())
    checkpoint_root = (repository / sweep["checkpoint_session"]).resolve()
    checkpoints = {
        int(vehicles): _find_checkpoint(checkpoint_root, int(vehicles))
        for vehicles in sweep["vehicle_counts"]
    }
    cases = _cases(sweep, profiles)
    configurations = [
        _configuration(base, sweep, profile, name, strategy, vehicles, checkpoints, Path("dry-run"))
        for vehicles, name, strategy, profile in cases
    ]
    if args.dry_run:
        for configuration in configurations:
            configuration.validate()
        print(f"DRY RUN OK: {len(configurations)} configurations validated")
        return 0

    commit = _git_commit(repository) or "unknown"
    signature = sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    session = (repository / sweep["output_dir"] / f"run-{commit[:8]}-{signature}").resolve()
    session.mkdir(parents=True, exist_ok=True)
    state_path = session / "sweep-state.json"
    state = _load_state(state_path, commit, signature)
    rows_by_key = {item["key"]: item["row"] for item in state.get("runs", {}).values()}

    for index, (vehicles, name, strategy, profile) in enumerate(cases, start=1):
        key = f"{name}:{vehicles}"
        saved = rows_by_key.get(key)
        if saved and Path(saved["run_dir"], "summary.json").exists():
            print(f"SKIP {index}/{len(cases)} {name} vehicles={vehicles}", flush=True)
            continue
        configuration = _configuration(
            base, sweep, profile, name, strategy, vehicles, checkpoints, session
        )
        print(f"START {index}/{len(cases)} {name} vehicles={vehicles}", flush=True)
        row = _execute(configuration, name)
        state.setdefault("runs", {})[key] = {"key": key, "row": row}
        state.update({"git_commit": commit, "signature": signature})
        _write_json_atomic(state_path, state)
        rows_by_key[key] = row
        print(
            f"DONE {index}/{len(cases)} success={100 * row['success_rate']:.2f}% "
            f"queue={row['avg_cloud_queue_length']:.2f} wall={row['wall_clock_s']:.1f}s",
            flush=True,
        )

    rows = [rows_by_key[f"{name}:{vehicles}"] for vehicles, name, _, _ in cases]
    _write_csv(session / "capacity-sweep-results.csv", rows)
    _write_json_atomic(session / "capacity-sweep-results.json", {"runs": rows})
    (session / "capacity-sweep-summary.md").write_text(
        _markdown(rows), encoding="utf-8"
    )
    print(f"COMPLETE {session}")
    print(f"SUMMARY {session / 'capacity-sweep-summary.md'}")
    return 0


def _cases(sweep: dict, profiles: list[dict]) -> list[tuple[int, str, str, dict]]:
    cases = []
    for value in sweep["vehicle_counts"]:
        vehicles = int(value)
        if sweep.get("include_stackelberg_reference", True):
            cases.append((vehicles, "stackelberg_reference", "stackelberg", {}))
        cases.extend(
            (vehicles, profile["name"], "hybrid_stackelberg", profile)
            for profile in profiles
        )
    return cases


def _configuration(
    base: SimulationConfig,
    sweep: dict,
    profile: dict,
    name: str,
    strategy: str,
    vehicles: int,
    checkpoints: dict[int, Path],
    session: Path,
) -> SimulationConfig:
    config = copy.deepcopy(base)
    config.strategy = strategy
    config.vehicle_count = vehicles
    config.steps = int(sweep["steps"])
    config.seed = int(sweep["seed"])
    config.record_decision_diagnostics = bool(sweep.get("record_q_values", False))
    config.output_dir = str(session / "runs" / f"{name}-{vehicles}")
    _apply_profile(config, profile)
    if strategy == "hybrid_stackelberg":
        config.dqn.mode = "evaluate"
        config.dqn.checkpoint_path = str(checkpoints[vehicles])
    else:
        config.dqn.mode = "train"
        config.dqn.checkpoint_path = None
    config.validate()
    return config


def _apply_profile(config: SimulationConfig, profile: dict) -> None:
    mappings = {
        "fusion_mode": ("hybrid_fusion_mode", str),
        "residual_weight": ("hybrid_residual_weight", float),
        "residual_congestion_adaptation": ("hybrid_residual_congestion_adaptation", bool),
        "residual_decay_start_ratio": ("hybrid_residual_decay_start_ratio", float),
        "residual_min_scale": ("hybrid_residual_min_scale", float),
        "cloud_capacity_guard": ("hybrid_cloud_capacity_guard", bool),
        "cloud_guard_ratio": ("hybrid_cloud_guard_ratio", float),
    }
    unknown = set(profile) - {"name", *mappings}
    if unknown:
        raise ValueError(f"unknown profile settings: {sorted(unknown)}")
    for source, (target, converter) in mappings.items():
        if source in profile:
            setattr(config.decision, target, converter(profile[source]))


def _find_checkpoint(root: Path, vehicles: int) -> Path:
    candidates = sorted(
        (root / "training" / f"hybrid_stackelberg-{vehicles}").glob("*/dqn-policy.pt")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one Hybrid checkpoint for {vehicles} vehicles under {root}, found {len(candidates)}"
        )
    return candidates[0].resolve()


def _execute(config: SimulationConfig, profile: str) -> dict:
    wall_started = perf_counter()
    cpu_started = process_time()
    summary, run_dir = SimulationRunner(config).run()
    return {
        "profile": profile,
        **asdict(summary),
        "wall_clock_s": perf_counter() - wall_started,
        "process_cpu_s": process_time() - cpu_started,
        "run_dir": run_dir,
    }


def _markdown(rows: list[dict]) -> str:
    references = {
        int(row["configured_vehicle_count"]): row
        for row in rows
        if row["profile"] == "stackelberg_reference"
    }
    ranked = sorted(
        (row for row in rows if row["profile"] != "stackelberg_reference"),
        key=lambda row: (
            int(row["configured_vehicle_count"]),
            -float(row["success_rate"]),
            float(row["avg_latency_s"]),
        ),
    )
    lines = [
        "# Capacity-aware Hybrid sweep",
        "",
        "Frozen policies are evaluated on the same task stream as the Stackelberg reference.",
        "",
        "| Vehicles | Profile | Success | vs Stackelberg | Latency (s) | Cloud queue | DQN decisions | Deviations |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        vehicles = int(row["configured_vehicle_count"])
        delta = float(row["success_rate"]) - float(references[vehicles]["success_rate"])
        lines.append(
            f"| {vehicles} | {row['profile']} | {100 * float(row['success_rate']):.2f}% | "
            f"{100 * delta:+.2f} pp | {float(row['avg_latency_s']):.4f} | "
            f"{float(row['avg_cloud_queue_length']):.2f} | "
            f"{100 * float(row['dqn_decision_ratio']):.2f}% | "
            f"{100 * float(row['hybrid_deviation_ratio']):.2f}% |"
        )
    lines.extend([
        "",
        "The recommended profile is the one that improves success consistently across loads, not merely the highest single result.",
        "",
    ])
    return "\n".join(lines)


def _git_commit(repository: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_state(path: Path, commit: str, signature: str) -> dict:
    if not path.exists():
        return {"runs": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("git_commit") != commit or state.get("signature") != signature:
        raise RuntimeError("sweep state does not match the current commit and configuration")
    return state


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
