# Architecture

## Simulation data flow

1. A mobility provider supplies active vehicle IDs, positions, speeds, and newly departed vehicles.
2. The seeded environment generator creates the same tasks for every strategy under the same mobility trace.
3. Vehicles that do not generate a task in the current step become service candidates, matching the role definition in the thesis.
4. At the start of a step, the cloud publishes either the literal queue-sensitive baseline price or a bounded, smoothed leader best response using observed demand and capacity pressure. Service vehicles accept only when energy, queue, and utility constraints pass, then quote according to the thesis follower equation.
5. A shared estimator computes Local, V2V, and V2I delay, energy, and payment. V2I energy includes both transmission and cloud-computation energy. Radio throughput has explicit units: `R[Mbit/s] = B[MHz] * eta[bit/s/Hz] * resource_efficiency`; SNR follows the log-distance path-loss model `SNR(d)=SNR(d0)-10*n*log10(d/d0)`. V2V searches deterministic relay paths up to three hops.
6. Every task vehicle chooses from the same start-of-step state snapshot and the same published prices. No task can observe queue changes caused by another decision in that step. The shared DQN is parameter sharing, not a centralized controller: transitions remain vehicle-local and rewards contain only that vehicle's delay, energy, payment, and success.
7. If an on-time action exists, the reviewed policy removes avoidable timeouts. If all actions are late, all physically feasible actions remain available. Hybrid directly accepts only a Pareto-dominant game solution; otherwise it combines immediate private reward with a conservative online/target-network advantage. Cloud capacity pressure is internalized by the public Stackelberg price, not by a hidden global reward term.
8. After the complete decision batch is fixed, Local, V2V, and V2I arrivals enter their queues in a seeded, strategy-independent order. V2I executes through either the deterministic analytical backend or the real HTTP backend.
9. The reward and next decision state form a real temporal replay transition. DQN/Hybrid store every transition, use masked Double-DQN targets, load-stratified replay in the reviewed profile, and update after replay warm-up. Network weights and epsilon remain fixed throughout one decision batch.
10. Task records and run summaries are written to separate schemas. The summary
    reports how often every action was already late, what fraction of those
    tasks was still admitted to cloud, and how much cloud capacity those
    predicted failures consumed.

## Serverless boundary

`ServerlessBackend.execute()` is the only simulation-to-cloud execution boundary. The analytical and HTTP implementations return the same measurement type, so strategy logic and metric collection do not diverge between backends.

The HTTP function performs bounded deterministic CPU work. Client latency includes Knative routing, activation, queueing, and execution. Function `processing_ms` measures only user-container work. The difference is retained as `platform_overhead_ms` rather than hidden in a constant. The simulator does not add the analytical 0.1-second cold-start term to a real HTTP measurement.

All V2I requests selected in one simulation step are submitted concurrently, bounded by `client_concurrency`, after that step's complete decision batch is fixed. Each worker owns its HTTP session. This exposes real queueing and Knative concurrency while keeping replay and metrics deterministic.

The analytical backend models the same bounded autoscaling controls as the Knative manifest. Requests are distributed by concurrency target up to the configured maximum instance count before queue delay is calculated. Cold-start prediction and execution share the same backend state.
