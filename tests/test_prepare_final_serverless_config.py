from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest


def _load_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare-final-serverless-config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_final_serverless_config",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load_runner()


class PrepareFinalServerlessConfigTests(unittest.TestCase):
    def test_discovers_checkpoint_from_completed_pipeline_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "policy.pt"
            checkpoint.write_bytes(b"policy")
            session = root / "run-one"
            session.mkdir()
            (session / "pipeline-state.json").write_text(
                json.dumps(
                    {
                        "runs": {
                            "train:hybrid_stackelberg:2000": {
                                "checkpoint": str(checkpoint)
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                RUNNER._discover_checkpoint(root, 2000),
                checkpoint.resolve(),
            )

    def test_discovers_shared_dqn_checkpoint_for_decoupled_hybrid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "dqn-policy.pt"
            checkpoint.write_bytes(b"policy")
            session = root / "run-one"
            session.mkdir()
            (session / "pipeline-state.json").write_text(
                json.dumps(
                    {
                        "runs": {
                            "train:dqn:4000": {
                                "checkpoint": str(checkpoint)
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                RUNNER._discover_checkpoint(root, 4000, "dqn"),
                checkpoint.resolve(),
            )

    def test_selects_requested_replicates_without_changing_seed_pairing(self) -> None:
        self.assertEqual(
            RUNNER._select_replicates([87, 88, 89, 90], [1, 3]),
            [(1, 87), (3, 89)],
        )
        with self.assertRaisesRegex(ValueError, "within 1..4"):
            RUNNER._select_replicates([87, 88, 89, 90], [5])

    def test_final_statistics_use_selected_hybrid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "evaluation-results.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "replicate",
                        "phase",
                        "strategy",
                        "configured_vehicle_count",
                        "total_tasks",
                        "v2i_offload_ratio",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "replicate": 1,
                            "phase": "evaluate",
                            "strategy": "hybrid_stackelberg",
                            "configured_vehicle_count": 1000,
                            "total_tasks": 100,
                            "v2i_offload_ratio": 0.5,
                        },
                        {
                            "replicate": 2,
                            "phase": "evaluate",
                            "strategy": "hybrid_stackelberg",
                            "configured_vehicle_count": 1000,
                            "total_tasks": 200,
                            "v2i_offload_ratio": 0.75,
                        },
                        {
                            "replicate": 1,
                            "phase": "evaluate",
                            "strategy": "dqn",
                            "configured_vehicle_count": 1000,
                            "total_tasks": 999,
                            "v2i_offload_ratio": 1.0,
                        },
                    ]
                )
            estimates, budgets = RUNNER._final_evaluation_statistics(
                root,
                [(1, 87), (2, 88)],
                [1000],
            )
            self.assertEqual(estimates, {1000: 200})
            self.assertEqual(budgets, {1000: 10225})

    def test_render_uses_seed_specific_checkpoint_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = Path(temp)
            config = repository / "configs" / "base.toml"
            policy = repository / "results" / "policy.pt"
            config.parent.mkdir()
            policy.parent.mkdir()
            config.write_text("[simulation]\n", encoding="utf-8")
            policy.write_bytes(b"policy")
            rendered = RUNNER._render(
                repository,
                config,
                [1000],
                [61],
                {(1000, 61): policy},
            )
            parsed = tomllib.loads(rendered)["validation"]
            self.assertEqual(parsed["seeds"], [61])
            self.assertEqual(parsed["base_config"], "base.toml")
            self.assertEqual(
                parsed["checkpoints"]["1000:61"],
                "results/policy.pt",
            )

    def test_render_accepts_generated_base_config_outside_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repository = Path(temp)
            config = repository / "results" / "generated" / "winner.toml"
            policy = repository / "results" / "policy.pt"
            config.parent.mkdir(parents=True)
            policy.parent.mkdir(exist_ok=True)
            config.write_text("[simulation]\n", encoding="utf-8")
            policy.write_bytes(b"policy")
            rendered = RUNNER._render(
                repository,
                config,
                [2000],
                [81],
                {(2000, 81): policy},
                evaluation_steps=500,
                validation_output_dir=Path("results/validation"),
                analytical_parallelism=4,
            )
            parsed = tomllib.loads(rendered)["validation"]
            self.assertEqual(parsed["base_config"], config.as_posix())
            self.assertEqual(parsed["steps"], [500])
            self.assertEqual(parsed["output_dir"], "results/validation")
            self.assertEqual(parsed["analytical_parallelism"], 4)


if __name__ == "__main__":
    unittest.main()
