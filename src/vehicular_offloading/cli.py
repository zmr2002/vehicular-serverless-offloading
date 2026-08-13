from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .config import SimulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vehicular-offloading")
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate", help="run one reproducible simulation")
    simulate.add_argument("--config", type=Path)
    simulate.add_argument("--strategy", choices=("random", "greedy", "dqn", "stackelberg", "hybrid_stackelberg"))
    simulate.add_argument("--backend", choices=("analytical", "knative"))
    simulate.add_argument("--mobility", choices=("synthetic", "sumo"))
    simulate.add_argument("--steps", type=int)
    simulate.add_argument("--vehicles", type=int)
    simulate.add_argument("--seed", type=int)
    simulate.add_argument("--output-dir")
    simulate.add_argument("--endpoint", help="Knative or local function base URL")
    simulate.add_argument("--dqn-mode", choices=("train", "evaluate"))
    simulate.add_argument("--checkpoint", type=Path)

    experiment = sub.add_parser("experiment", help="run a strategy/vehicle/seed matrix")
    experiment.add_argument("--config", type=Path, required=True)

    routes = sub.add_parser("generate-routes", help="generate exactly N valid SUMO vehicle routes")
    routes.add_argument("--net", type=Path, required=True)
    routes.add_argument("--output", type=Path, required=True)
    routes.add_argument("--vehicles", type=int, required=True)
    routes.add_argument("--end", type=float, default=2_000.0)
    routes.add_argument("--seed", type=int, default=42)

    benchmark = sub.add_parser("serverless-benchmark", help="measure cold/warm and burst request latency")
    benchmark.add_argument("--endpoint", required=True)
    benchmark.add_argument("--output-dir", default="results/verified/serverless")
    benchmark.add_argument("--requests", type=int, default=50)
    benchmark.add_argument("--work-units", type=int, default=25_000)

    plot = sub.add_parser("plot-results", help="create figures only from a saved matrix summary CSV")
    plot.add_argument("--input", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        from .simulation import SimulationRunner

        config = SimulationConfig.from_toml(args.config) if args.config else SimulationConfig()
        for argument, attribute in (
            (args.strategy, "strategy"), (args.backend, "backend"), (args.mobility, "mobility"),
            (args.steps, "steps"), (args.vehicles, "vehicle_count"), (args.seed, "seed"),
            (args.output_dir, "output_dir"),
        ):
            if argument is not None:
                setattr(config, attribute, argument)
        if args.endpoint is not None:
            config.serverless.endpoint = args.endpoint
        if args.dqn_mode is not None:
            config.dqn.mode = args.dqn_mode
        if args.checkpoint is not None:
            config.dqn.checkpoint_path = str(args.checkpoint)
        config.validate()
        summary, run_dir = SimulationRunner(config).run()
        print(json.dumps({"run_dir": run_dir, "summary": asdict(summary)}, indent=2))
        return 0
    if args.command == "experiment":
        from .experiments import run_matrix

        print(run_matrix(args.config))
        return 0
    if args.command == "generate-routes":
        from .routes import generate_exact_routes

        summary = generate_exact_routes(args.net, args.output, args.vehicles, args.end, args.seed)
        print(json.dumps(asdict(summary), indent=2))
        return 0
    if args.command == "serverless-benchmark":
        from .experiments import benchmark_serverless

        print(
            benchmark_serverless(
                args.endpoint,
                args.output_dir,
                requests_per_level=args.requests,
                work_units=args.work_units,
            )
        )
        return 0
    if args.command == "plot-results":
        from .plotting import plot_matrix_summary

        print(plot_matrix_summary(args.input, args.output))
        return 0
    return 2
