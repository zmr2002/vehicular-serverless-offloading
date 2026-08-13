from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run-final-multiseed.py"
SPEC = importlib.util.spec_from_file_location("final_multiseed_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class FinalMultiseedTests(unittest.TestCase):
    def test_subset_pipeline_allows_hybrid_only_screen(self):
        pipeline = {
            "base_config": "base.toml",
            "output_dir": "results/screen",
            "training_steps": 50,
            "evaluation_steps": 50,
            "vehicle_counts": [100],
            "training_seeds": [1, 2],
            "evaluation_seeds": [3, 4],
            "training_strategies": ["hybrid_stackelberg"],
            "evaluation_strategies": ["hybrid_stackelberg"],
            "training_task_sample_rate": 0.0,
            "evaluation_task_sample_rate": 0.001,
            "parallelism": 2,
            "minimum_free_disk_gb": 0.0,
            "storage_upper_bound_gb": 1.0,
            "allow_strategy_subset": True,
        }
        RUNNER._validate_pipeline(pipeline)
        pipeline["evaluation_strategies"] = ["dqn"]
        with self.assertRaisesRegex(ValueError, "requires a trained checkpoint"):
            RUNNER._validate_pipeline(pipeline)

    def test_global_worker_allocation_never_exceeds_budget(self):
        replicates, workers = RUNNER._global_worker_allocation(3, 6)
        self.assertEqual((replicates, workers), (3, 2))
        self.assertLessEqual(replicates * workers, 6)
        replicates, workers = RUNNER._global_worker_allocation(5, 4)
        self.assertEqual((replicates, workers), (4, 1))
        self.assertLessEqual(replicates * workers, 4)

    def test_student_interval_uses_sample_standard_deviation(self):
        mean_value, sample_std, ci95 = RUNNER._mean_std_ci(
            [0.80, 0.82, 0.84, 0.86, 0.88]
        )
        self.assertAlmostEqual(mean_value, 0.84)
        self.assertAlmostEqual(sample_std, 0.0316227766)
        self.assertAlmostEqual(ci95, 0.03926486, places=7)

    def test_paired_comparison_uses_within_replicate_differences(self):
        rows = []
        for replicate, hybrid, stack in (
            (1, 0.90, 0.80),
            (2, 0.70, 0.75),
        ):
            for strategy, success in (
                ("random", 0.40),
                ("greedy", 0.50),
                ("dqn", 0.60),
                ("stackelberg", stack),
                ("hybrid_stackelberg", hybrid),
            ):
                rows.append(
                    {
                        "replicate": replicate,
                        "configured_vehicle_count": 1000,
                        "strategy": strategy,
                        "success_rate": success,
                        "avg_success_latency_s": 1.0,
                        "avg_energy_j": 2.0,
                        "avg_cost_per_task": 3.0,
                        "avg_reward": 4.0,
                    }
                )
        comparisons = RUNNER._paired_comparisons(rows)
        stack_row = next(
            row
            for row in comparisons
            if row["comparison"] == "hybrid_stackelberg-minus-stackelberg"
        )
        self.assertAlmostEqual(stack_row["success_rate_mean_delta"], 0.025)
        self.assertEqual(stack_row["replicates"], 2)

    def test_result_validation_requires_identical_task_streams(self):
        pipeline = {
            "training_seeds": [1, 2],
            "vehicle_counts": [1000],
            "evaluation_steps": 10,
        }
        rows = []
        for replicate in (1, 2):
            for index, strategy in enumerate(RUNNER.STRATEGIES):
                rows.append(
                    {
                        "replicate": replicate,
                        "configured_vehicle_count": 1000,
                        "strategy": strategy,
                        "success_rate": 0.5,
                        "local_offload_ratio": 0.2,
                        "v2v_offload_ratio": 0.3,
                        "v2i_offload_ratio": 0.5,
                        "completed_steps": 10,
                        "total_tasks": 100 + (index if replicate == 2 else 0),
                        "run_dir": "test",
                    }
                )
        with self.assertRaisesRegex(RuntimeError, "task streams differ"):
            RUNNER._validate_results(rows, pipeline)


if __name__ == "__main__":
    unittest.main()
