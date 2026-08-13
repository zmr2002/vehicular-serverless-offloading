from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from vehicular_offloading.config import SimulationConfig


SCRIPT = Path(__file__).parents[1] / "scripts" / "run-training-evaluation.py"
SPEC = importlib.util.spec_from_file_location("training_evaluation_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class TrainingEvaluationTests(unittest.TestCase):
    def test_resume_metadata_does_not_change_experiment_signature(self):
        base = SimulationConfig()
        pipeline = {
            "training_steps": 10,
            "evaluation_steps": 10,
            "vehicle_counts": [10],
            "training_strategies": ["dqn", "hybrid_stackelberg"],
            "evaluation_strategies": ["random"],
        }
        original = RUNNER._pipeline_signature(base, pipeline)
        compatible = RUNNER._pipeline_signature(
            base,
            {**pipeline, "compatible_resume_commits": ["old-commit"]},
        )
        self.assertEqual(original, compatible)

    def test_compatible_commit_selects_existing_session(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            existing = output / "run-12345678-signature"
            existing.mkdir()
            selected = RUNNER._select_session(
                output,
                "abcdef012345",
                "signature",
                ["1234567890ab"],
            )
            self.assertEqual(selected, existing)

    def test_completed_run_is_recovered_after_state_write_interruption(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            run_dir = output / "completed"
            run_dir.mkdir()
            config = SimulationConfig(
                steps=3,
                vehicle_count=8,
                seed=17,
                strategy="greedy",
                output_dir=str(output),
            )
            summary = {
                "strategy": "greedy",
                "configured_vehicle_count": 8,
                "configured_steps": 3,
                "completed_steps": 3,
                "seed": 17,
                "success_rate": 1.0,
            }
            timing = {
                "wall_clock_s": 2.5,
                "phase_seconds": {"mobility": 1.0},
            }
            (run_dir / "summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            (run_dir / "timing.json").write_text(
                json.dumps(timing),
                encoding="utf-8",
            )
            recovered = RUNNER._recover_completed_run(config, "evaluate")
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered["run_dir"], str(run_dir))
            self.assertEqual(recovered["wall_clock_s"], 2.5)

    def test_diagnostics_compare_identical_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stack_dir = root / "stack"
            hybrid_dir = root / "hybrid"
            stack_dir.mkdir()
            hybrid_dir.mkdir()
            fields = [
                "task_id", "task_deadline_s", "local_estimate_s", "v2v_estimate_s",
                "v2i_estimate_s", "source_workload_s", "max_service_workload_s",
                "cloud_queue_length", "q_local", "q_v2v", "q_v2i", "success",
                "reward", "delay_s",
            ]
            common = {
                "task_id": "task-1",
                "task_deadline_s": "1.0",
                "local_estimate_s": "1.2",
                "v2v_estimate_s": "0.8",
                "v2i_estimate_s": "0.6",
                "source_workload_s": "0.2",
                "max_service_workload_s": "0.4",
                "cloud_queue_length": "2",
                "q_local": "",
                "q_v2v": "",
                "q_v2i": "",
            }
            self._write_tasks(stack_dir / "tasks.csv", fields, [{**common, "success": "0", "reward": "-2", "delay_s": "1.2"}])
            self._write_tasks(hybrid_dir / "tasks.csv", fields, [{**common, "success": "1", "reward": "2", "delay_s": "0.6", "q_local": "0.1", "q_v2v": "0.2", "q_v2i": "0.3"}])
            rows = [
                self._summary("stackelberg", stack_dir, 0.0),
                self._summary("hybrid_stackelberg", hybrid_dir, 1.0),
            ]
            diagnostics = RUNNER._summarize_diagnostics(rows)
            comparison = diagnostics["hybrid_vs_stackelberg"][0]
            self.assertEqual(comparison["common_tasks"], 1)
            self.assertEqual(comparison["hybrid_success_wins"], 1)
            hybrid = next(row for row in diagnostics["runs"] if row["strategy"] == "hybrid_stackelberg")
            self.assertEqual(hybrid["v2i_deadline_feasible_rate"], 1.0)
            self.assertAlmostEqual(hybrid["q_v2i_mean"], 0.3)

    @staticmethod
    def _write_tasks(path: Path, fields: list[str], rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _summary(strategy: str, run_dir: Path, success_rate: float) -> dict:
        return {
            "strategy": strategy,
            "configured_vehicle_count": 1000,
            "success_rate": success_rate,
            "oracle_success_rate": 1.0,
            "avg_latency_s": 0.5,
            "avg_energy_j": 10.0,
            "avg_cost_per_task": 0.2,
            "avg_reward": 20.0,
            "hybrid_deviation_ratio": 0.5 if strategy.startswith("hybrid") else 0.0,
            "hybrid_beneficial_deviation_rate": 1.0 if strategy.startswith("hybrid") else 0.0,
            "run_dir": str(run_dir),
        }

    def test_decoupled_hybrid_evaluates_the_pure_dqn_checkpoint(self):
        pipeline = {
            "training_steps": 10,
            "evaluation_steps": 10,
            "vehicle_counts": [10],
            "training_strategies": ["dqn"],
            "evaluation_strategies": [
                "random", "greedy", "dqn", "stackelberg", "hybrid_stackelberg",
            ],
            "hybrid_checkpoint_strategy": "dqn",
        }
        RUNNER._validate_pipeline(pipeline)
        self.assertEqual(
            RUNNER._checkpoint_strategy(pipeline, "hybrid_stackelberg"), "dqn"
        )
        self.assertEqual(RUNNER._checkpoint_strategy(pipeline, "dqn"), "dqn")

    def test_hybrid_evaluation_requires_its_checkpoint_source_to_be_trained(self):
        pipeline = {
            "training_steps": 10,
            "evaluation_steps": 10,
            "vehicle_counts": [10],
            "training_strategies": ["dqn"],
            "evaluation_strategies": ["hybrid_stackelberg"],
        }
        with self.assertRaises(ValueError):
            RUNNER._validate_pipeline(pipeline)
        RUNNER._validate_pipeline(
            {**pipeline, "hybrid_checkpoint_strategy": "dqn"}
        )
        with self.assertRaises(ValueError):
            RUNNER._validate_pipeline(
                {**pipeline, "hybrid_checkpoint_strategy": "greedy"}
            )


if __name__ == "__main__":
    unittest.main()
