# Optimization sweeps

The screening workflow deliberately separates two questions.

- The speed sweep keeps the workload, physical model, station layout, reward, and decision weights fixed. It compares Stackelberg as a non-DQN lower bound, DQN and Hybrid training intervals, and CPU thread counts. The spatial-index and capped-neighbor-count changes are exact optimizations shared by all profiles.
- The result sweep keeps the task workload unchanged and tests numerical channel assumptions that the thesis does not fix: V2V channel width, reference SNR, V2V path-loss exponent, and analytical cloud elasticity. It exists to measure sensitivity of the mathematical ceiling, not to tune against the reported table.

Run both resumable sweeps in order from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-all-optimization-sweeps.ps1
```

The command runs 14 speed cases and 16 result cases. Each case uses 500 steps, seed 11, and either 1000 or 4000 configured vehicles. This is a screening matrix, not the final five-seed paper matrix. Re-running the command skips complete cases whose summary and timing files are present.

Outputs are written to:

- `results/verified/speed-optimization-sweep/optimization-results.csv`
- `results/verified/result-optimization-sweep/optimization-results.csv`
- `results/verified/combined-optimization-sweep/screening-summary.md`

Every row includes wall and process CPU time, phase timing, success and oracle-success rates, action ratios, DQN decision ratio, replay/update counts, the effective channel parameters, and the complete run directory. Channel profiles vary MHz and SNR separately because the thesis does not state their numerical allocation.

## Decision ablation

The channel screening showed that the improved Hybrid profile delegated only 2.8% of decisions to DQN at 4,000 vehicles because deadline masking and near-optimal candidate guidance usually left one action. The thesis specifies the full three-action DQN space and reward-based deadline feedback but does not specify either hard mask. The decision ablation holds mobility, tasks, channel parameters, and seed fixed while independently toggling these two improvements for Stackelberg, DQN, and Hybrid:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-decision-ablation.ps1
```

It runs 16 resumable 500-step cases over 1,000 and 4,000 vehicles. Results are written to `results/verified/decision-ablation-sweep/optimization-results.csv`.
