# Reproducibility protocol

## Controlled inputs

- Python and package versions are pinned in `requirements.lock`.
- Corrected legacy-layout parameters live in `configs/paper.toml`.
- Reviewed improvements and independently calibrated base stations live in `configs/paper-improved.toml`.
- Experiment dimensions and five fixed seeds live in `configs/paper-matrix.toml`.
- The exact number of SUMO routes is generated from the committed network for each `(vehicle_count, seed)` pair.
- Environment and policy random-number streams are independent, preventing action exploration from changing later task generation.
- The default channel model is `B_eff log2(1 + SNR)`, with the reference SNR, path-loss exponents, and spectral-efficiency cap recorded in every effective configuration. `distance_only` remains available only for controlled regression tests.
- In the thesis `dynamic_idle` role model, every physical vehicle retains the same 2 GHz CPU whether it generates a task or temporarily serves another vehicle. Configuration validation rejects role-dependent CPU rates in this mode.
- Because the thesis defines a task deadline symbol but gives no numerical distribution, the reviewed profile samples deadlines continuously from `2-3 s`. With continuous `1-5 Gcycle` demand and 2 GHz vehicle CPUs, the analytically intrinsic empty-queue timeout probability is 6.25%; queue-induced timeouts are reported separately.
- The thesis V2V equation leaves radio resources unspecified. Configuration separates channel bandwidth in MHz, spectral efficiency in bit/s/Hz, and a dimensionless resource-efficiency factor. The reviewed V2V design produces about 469 Mbit/s at 500 m, above the correctly unit-converted 404 Mbit/s required for the mean task's one-second communication budget; V2I retains more spectrum and the more stable path-loss profile.

## Run artifacts

Each run directory contains:

- `config.json`: the effective configuration after command-line overrides;
- `environment.json`: timestamp, OS, Python version, and Git commit;
- `tasks.csv`: one row per task with stable numeric fields;
- `summary.json`: aggregate metrics and realized mobility counts.

Task rows also retain the nearest base-station distance, the predicted minimum-delay action, predicted minimum delay, decision regret, source queue, selected V2V target queue, and whether any action could meet the deadline. Real HTTP rows additionally retain processing time, end-to-end client latency, platform overhead, instance ID, cold-start flag, and checksum. Summaries separate intrinsic single-task infeasibility, queue-induced local timeout, V2V latency advantage, V2V-rescuable timeout, and avoidable policy failure.

Matrix execution produces a detailed CSV and a grouped summary containing mean, standard deviation, and 95% confidence-interval half-width.

Every DQN/Hybrid decision is saved as a transition. The corrected legacy-layout profile performs one gradient update per four transitions. The improved profile uses interval `32` and one intra-op thread, selected by the fixed-seed speed screening without changing the decision-based epsilon schedule. The smoke profile uses `1` so tests exercise optimizer updates quickly.

Task-load calibration chooses the thesis-unspecified generation probability solely by distance to the declared offered vehicle-compute load target `0.90`. For the continuous `1-5` Gcycle workload and 2 GHz vehicles, this identifies the analytically relevant region around `p=0.60`. Channel-width sensitivity uses explicit MHz candidates and a predeclared V2V opportunity target derived from service-capacity utilization. Success rate and Hybrid advantage do not participate in either selection. Screening and training may suppress per-task CSV rows to reduce runtime and disk use, but their complete configuration and aggregate summary remain recorded; paired final evaluations always retain full task logs.

## Comparison policy

Analytical and live Knative runs answer different questions. Analytical runs compare algorithms under controlled latency equations. Knative runs demonstrate container execution, cold/warm behavior, and autoscaling under real machine load. Live measurements are not merged into the analytical baseline table.
