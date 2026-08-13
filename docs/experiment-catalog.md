# Experiment catalog

The repository retains intermediate experiments for auditability, but only a
small set of entry points defines the final reported model. This catalog keeps
the final workflow distinct without moving files and breaking old run records.

## Final model and evidence

| Purpose | Entry point | Configuration or output |
|---|---|---|
| Final six-replicate analytical matrix | `scripts/run-final-v2.ps1` | `configs/final-decoupled-v2.toml` |
| Final Hybrid model | used by the final runner | `configs/hybrid-decoupled.toml` |
| Final real Knative validation | `scripts/run-final-v2-knative-validation.ps1` | generated from the completed final matrix |
| Compact verified evidence | publication-safe CSV bundle | `results/verified/published/` |

`run-final-v2.ps1` performs the declared epsilon screen and then runs the final
five-strategy paired matrix. The Hybrid is evaluated over each replicate's
frozen pure-DQN checkpoint, as declared by
`hybrid_checkpoint_strategy = "dqn"`. The Knative runner reuses exactly those
checkpoints and seeds in analytical, action-replay, and closed-loop modes.

## Final-model supporting tools

- `scripts/run_eps_screen.py`: deterministic epsilon-recipe screen used before
  the final matrix.
- `scripts/run_cross_checkpoint_eval.py`: evaluates Hybrid arbitration over a
  shared DQN checkpoint.
- `scripts/run-adequacy-validation.ps1`: validates adequacy-gated arbitration.
- `scripts/run-final-multiseed.py`: reusable paired training/evaluation engine.
- `scripts/prepare-final-serverless-config.py`: binds completed paired
  checkpoints to the Knative validation matrix.
- `scripts/run-knative-validation.py`: resumable analytical/replay/closed-loop
  comparison and full-request Serverless metrics.

## Reproduction baselines

The following remain useful for reproducing earlier milestones and isolating
individual changes, but they are not the source of the final reported table:

- `scripts/run-training-evaluation-diagnostics.ps1`
- `scripts/run-reviewed-model-validation.ps1`
- `scripts/run-synchronous-model-paper-scale.ps1`
- `scripts/run-final-model-comparison.ps1`
- `scripts/run-hybrid-seed-stability.ps1`
- `scripts/run-knative-complete-validation.ps1`

Their matching TOML files retain the same descriptive stem under `configs/`.

## Exploratory and historical sweeps

Scripts and configurations containing `adaptive-gate`, `capacity`,
`follower-game`, `fusion-sweep`, `optimization-sweep`, `task-load-calibration`,
or `paper-single-seed` document earlier hypothesis tests and performance work.
They are retained so the final model's development can be audited. They should
not be combined with the final six-replicate estimates or presented as
independent confirmation data.

## Result retention boundary

- `results/legacy_reported/` preserves the thesis-era table and figure as
  historical, unverified output.
- `results/verified/published/` contains the compact final evidence committed to
  version control.
- Other `results/verified/` content is generated locally and ignored, including
  raw task records, traces, checkpoints, mobility caches, and intermediate
  screens.
