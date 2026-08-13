# Verified result bundle

This directory contains the compact, publication-safe evidence retained from
the final experiments. Raw task records, decision traces, mobility caches, and
model checkpoints remain excluded because they are generated artifacts and can
be large. None of the retained CSV files contains a local filesystem path.

## Final analytical comparison

The final comparison used six independently paired training/evaluation
replicates, 2,000 simulation steps, and identical task and mobility streams for
all five strategies.

| Vehicles | Random | Greedy | DQN | Stackelberg | Hybrid |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 69.96% | 94.46% | 94.24% | 99.99% | **99.99%** |
| 2,000 | 55.89% | 68.82% | 94.45% | 92.86% | **96.97%** |
| 4,000 | 39.25% | 60.82% | 83.70% | 72.04% | **84.25%** |

Hybrid achieved the highest mean success rate at every vehicle scale. At 1,000
vehicles it was effectively tied with Stackelberg. Its mean advantage over DQN
was 5.75, 2.52, and 0.56 percentage points at 1,000, 2,000, and 4,000 vehicles.
Its corresponding advantage over Stackelberg was approximately 0.00, 4.10,
and 12.21 percentage points. The paired confidence intervals are retained in
`analytical-paired-comparisons.csv`; the closest Hybrid-versus-DQN differences
at 2,000 and 4,000 vehicles are positive in mean but not statistically
significant with six replicates.

## Knative validation

The final Hybrid was also evaluated with the same frozen policies and inputs in
analytical, Knative replay, and Knative closed-loop modes. Three predeclared
replicates were run at each vehicle scale.

| Vehicles | Analytical Hybrid | Knative closed loop | Mean change |
|---:|---:|---:|---:|
| 1,000 | 99.995% | 99.997% | +0.003 pp |
| 2,000 | 97.54% | 98.99% | +1.45 pp |
| 4,000 | 84.66% | 84.56% | -0.10 pp |

The live validation issued approximately 3.16 million real HTTP requests across
replay and closed-loop modes. It measured cold starts, client dispatch, HTTP
round trips, processing time, platform overhead, retries, and pod scaling. The
Knative study validates deployment fidelity for Hybrid; the five-strategy
ranking comes from the analytical paired matrix, not from a five-strategy live
Knative comparison.

## Files

- `analytical-aggregate.csv`: means, sample standard deviations, and 95%
  Student-t confidence intervals for all strategies and metrics.
- `analytical-paired-comparisons.csv`: paired Hybrid-minus-comparator effects.
- `knative-comparison.csv`: all 18 analytical/live paired backend comparisons.
- `provenance.json`: configurations, seeds, commits, and source checksums.
