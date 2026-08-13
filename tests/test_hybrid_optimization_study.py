from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run-hybrid-optimization-study.py"
SPEC = importlib.util.spec_from_file_location("hybrid_optimization_study", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class HybridOptimizationStudyTests(unittest.TestCase):
    def test_screen_selection_penalizes_seed_instability(self):
        winner = RUNNER._select_screen_winner(
            {
                "unstable": {
                    "robust_success": 0.70,
                    "success": 0.90,
                    "reward": 2.0,
                    "queue": 1.0,
                },
                "stable": {
                    "robust_success": 0.82,
                    "success": 0.84,
                    "reward": 1.0,
                    "queue": 2.0,
                },
            }
        )
        self.assertEqual(winner, "stable")

    def test_confirmation_requires_worst_load_advantage(self):
        def loads(high_delta: float, low_delta: float):
            return {
                2000: {
                    "dqn": {"success": 0.80},
                    "stackelberg": {"success": 0.82},
                    "hybrid_stackelberg": {
                        "success": 0.82 + low_delta,
                        "reward": 1.0,
                    },
                },
                4000: {
                    "dqn": {"success": 0.70},
                    "stackelberg": {"success": 0.72},
                    "hybrid_stackelberg": {
                        "success": 0.72 + high_delta,
                        "reward": 1.0,
                    },
                },
            }

        winner = RUNNER._select_confirmation_winner(
            {
                "fragile": loads(0.10, -0.01),
                "robust": loads(0.02, 0.01),
            }
        )
        self.assertEqual(winner, "robust")

    def test_generated_profiles_enable_exact_diagnostics(self):
        repository = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as temp:
            profiles = RUNNER._generate_causal_profiles(
                repository,
                Path(temp),
                {
                    "current": {"base_config": "paper-thesis-hybrid.toml"},
                    "legacy_retrained": {
                        "base_config": "pre-serverless-adaptive-gate.toml"
                    },
                },
            )
            for path in profiles.values():
                self.assertIn(
                    "record_decision_diagnostics = true",
                    path.read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
