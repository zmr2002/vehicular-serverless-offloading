# Decoupled Hybrid

## Why the co-trained Hybrid failed to dominate

The final paired multi-seed matrices trained the Hybrid's internal DQN under
the arbitration gate. Because the gate executes most decisions (the DQN acts
in only 12-35% of training steps) and epsilon decays per task (exploration
effectively ends after ~30 of 2,000 steps at paper scale), whether the
internal DQN learns a useful override policy depends on the training seed:

- Final matrix at 2,000 vehicles: 90.29 / 91.11 / 97.42% across replicates
  (`results/verified/hybrid-optimization-study/final`).
- Eight-seed stability study: 85.7-98.5% at 2,000 vehicles and 68.1-84.5% at
  4,000 vehicles from the training seed alone, while evaluation-seed noise is
  0.2-1.4 pp (`results/verified/hybrid-seed-stability`).
- Every historical "globally optimal" Hybrid table traces to either a selected
  checkpoint (`baseline-seed48`) or a lucky seed; the causal screen's
  `legacy_retrained` profile confirmed that retraining the old configuration
  does not reproduce those wins.

## The decoupled model

Training and fusion are separated:

1. Train only the pure DQN (its usual epsilon-greedy exploration is never
   gated).
2. At evaluation, `hybrid_stackelberg` loads that frozen checkpoint and applies
   the load-adaptive arbitration unchanged
   ([hybrid-adaptive-arbitration](hybrid-adaptive-arbitration.md)).
3. The online reliability defense
   (`decision.hybrid_online_reliability = "evaluate"`) suppresses a
   miscalibrated checkpoint when its overrides demonstrably flip task success
   against the game estimate.

This matches the thesis text, which prescribes a follower cost comparison
with a DQN consultation for ambiguous cases and never requires the DQN to be
trained under the gate. It also removes every Hybrid training run from the
experiment matrices: only the DQN is trained, which roughly halves training
cost at paper scale without touching the physical model.

Cross-checkpoint evidence (arbitration over the same replicate's pure-DQN
checkpoint, no code changes) at 2,000 vehicles: 97.32 / 97.48 / 97.53%
against best baselines of 93.19 / 94.81 / 92.75% — strictly dominant in every
replicate with the training-seed spread reduced from 3.90 to 0.11 pp
(`results/verified/hybrid-cross-checkpoint-eval`).

## Validation protocol

`scripts/run-decoupled-validation.ps1` performs, resumably:

1. **Cross-checkpoint arms** on the existing final-matrix checkpoints
   (evaluation only): `dqnckpt` (no defense), `dqnckpt-reliability`, and
   `internal-reliability` (defense over the co-trained checkpoints) at
   1,000/2,000/4,000 vehicles for replicates 1-3. Output:
   `results/verified/hybrid-cross-checkpoint-eval/arms-summary.md`.
2. **Fresh-seed final matrix** (`configs/final-decoupled.toml`): training
   seeds 31641-31643 and evaluation seeds 84-86 were not used during the
   diagnosis, so this stage carries the confirmatory claim. Output:
   `results/verified/final-decoupled/final-summary.md`.

Decision criteria:

- The claim of global optimality rests only on stage 2's paired differences.
- Stage 1 separates the contribution of decoupling (dqnckpt vs the original
  matrix) from the defense (dqnckpt-reliability vs dqnckpt) and shows whether
  the defense alone can rescue a co-trained checkpoint
  (internal-reliability).
- If `dqnckpt-reliability` underperforms `dqnckpt` at 2,000 vehicles by more
  than noise, drop the defense from `configs/hybrid-decoupled.toml` before
  interpreting stage 2, and rerun only the hybrid evaluations.

## 2026-08-10 arm outcomes and the adequacy revision

The arms triggered the criterion above and localized every remaining gap
(`results/verified/hybrid-cross-checkpoint-eval/arms-summary.md`):

| Scale | dqnckpt (no defense) | dqnckpt-reliability | best baseline |
|---:|---:|---:|---:|
| 1000 | 98.88% (-1.12) | 99.99% (-0.01) | 100.00% |
| 2000 | 97.45% (+3.86) | 85.10% (-8.49) | ~93.5% |
| 4000 | 84.40% (-0.56) | 81.81% (-3.15) | ~85.0% |

The always-on defense rescues bad checkpoints at light load (a fresh-seed
70.9% DQN was lifted to 99.96%) but removes the collectively beneficial
congestion-avoiding overrides at 2,000 vehicles. Window diagnostics at 4,000
vehicles additionally show the arbitration returning authority to the game at
congestion onset (DQN decision share 84% -> 18%) because the follower margin
grows unboundedly with congestion while the Q opposition is range-normalized;
the replicate that kept DQN authority (87.6%) beat pure DQN, the two that did
not (82.9/82.7%) lost to it.

The revision conditions both mechanisms on the game's *demonstrated adequacy*
`A` — the rolling success of game-followed decisions (see
[hybrid-adaptive-arbitration](hybrid-adaptive-arbitration.md)): the defense
is trusted only while `A` is near one, and the game evidence is damped as it
degrades. Because `A` is measured on the trajectory the mechanism itself
shapes (game-followed decisions are disproportionately the easy agreement
cases), the adaptive damping carries a feedback risk at 2,000 vehicles; a
stateless alternative fixes the same normalization asymmetry by capping the
normalized game evidence (`hybrid_game_evidence_cap`). Both candidates are
screened before either touches the fresh seeds:

- `dqnckpt-adequacy`: adequacy-gated defense plus adaptive `A^p` damping
  (`configs/hybrid-decoupled.toml`).
- `dqnckpt-cap`: adequacy-gated defense plus the structural evidence cap
  (`configs/cross-checkpoint-eval-cap.toml`).
- `dqnckpt-damping`: damping without any defense — ablation only, never
  selected.

`scripts/run-adequacy-validation.ps1` runs the three screening arms on the
study checkpoints (evaluation seeds 81-83), selects between the two
candidates by the pre-declared rule — best worst-scale mean margin over the
strongest baseline, ties broken by overall mean margin
(`scripts/select_screening_winner.py`) — then re-evaluates only the hybrid
rows of the fresh-seed matrix on the unchanged final-decoupled checkpoints
and writes `results/verified/final-decoupled/corrected-final-summary.md`.

Scope note: at 4,000 vehicles the fused model sits within ~3 pp of its own
environment's oracle while local queue-induced timeouts approach 20% for
every strategy; the system is supercritical and arbitration alone is near
its ceiling there. A strict, comfortable win at 4,000 vehicles would require
leader-side load shaping (pricing or admission control), which changes the
economic model and is deliberately out of scope for this revision.
