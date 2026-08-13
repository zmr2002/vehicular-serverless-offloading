from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path("scripts/run-hybrid-cross-evaluation.py")
SPEC = importlib.util.spec_from_file_location("cross_evaluation", SCRIPT)
CROSS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CROSS)


class CrossEvaluationTests(unittest.TestCase):
    def test_training_and_evaluation_effects_are_separated(self):
        training_seeds = [1, 2, 3]
        evaluation_seeds = [11, 12, 13]
        rows = []
        for vehicles in CROSS.VEHICLE_COUNTS:
            for training_index, training_seed in enumerate(training_seeds):
                for evaluation_index, evaluation_seed in enumerate(evaluation_seeds):
                    rows.append(
                        {
                            "training_seed": training_seed,
                            "evaluation_seed": evaluation_seed,
                            "vehicle_count": vehicles,
                            "success_rate": (
                                0.70
                                + 0.05 * training_index
                                + 0.01 * evaluation_index
                            ),
                        }
                    )

        analysis = CROSS._analyze(
            rows,
            training_seeds,
            evaluation_seeds,
        )

        for vehicles in CROSS.VEHICLE_COUNTS:
            shares = analysis["vehicle_counts"][str(vehicles)][
                "sum_of_squares_share"
            ]
            self.assertGreater(shares["training_seed"], shares["evaluation_seed"])
            self.assertAlmostEqual(shares["interaction"], 0.0)
            self.assertAlmostEqual(sum(shares.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
