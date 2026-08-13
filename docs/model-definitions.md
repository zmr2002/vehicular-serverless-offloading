# Formal model definitions

The five reported strategies share the same mobility, task stream, physical
queues, radio model, and execution backend. They differ only in pricing and
offloading decisions.

| Strategy | Pricing | Offloading decision |
|---|---|---|
| Random | Queue-based non-game price | Uniform random choice among physically feasible actions |
| Greedy | Queue-based non-game price | Minimum current delay, with deterministic action-order tie breaking |
| DQN | Queue-based non-game price | Per-vehicle DQN over Local, V2V, and V2I |
| Stackelberg | Three-stage follower-responsive pricing | Finite private-utility best response |
| Hybrid | Three-stage follower-responsive pricing | Strict dominance filtering followed by a per-vehicle DQN |

The DQN uses shared network parameters, but it is not a central dispatcher.
Every transition connects two decisions of the same vehicle, and every reward
contains that vehicle's own delay, energy, payment, and completion outcome.
Public cloud load, price, and service availability may enter the state; future
simulation data and centrally assigned actions may not.

## Synchronous within-step congestion

All vehicles in one simulation step decide from the same published price and
start-of-step state. They cannot observe an arbitrary task-order update made
earlier in that step. For a task \(i\) considering service vehicle \(s\), the
model therefore exposes the expected queue created by the other simultaneous
followers:

\[
\widehat T^{\mathrm{batch}}_{i,s}
=\frac{1}{2F_s}\sum_{j\ne i}\pi_{j,s}C_j.
\]

Here \(\pi_{j,s}\) is vehicle \(j\)'s autonomous probability of selecting V2V
target \(s\), \(C_j\) is its compute demand, and the factor \(1/2\) is the
expected fraction of work arriving before task \(i\) under the seeded,
strategy-independent within-step admission order. Each vehicle still selects
its own Local, V2V, or V2I action. The probabilities are used only to forecast
congestion; they are not a central assignment.

During execution, the forecast is removed and replaced by the exact workload
of tasks actually admitted before the selected task. Thus the expected and
realized queues are never counted twice. This closes the same-batch V2V
congestion gap already handled for cloud-price demand response.

## Thesis and enhanced Hybrid

`configs/paper-thesis-hybrid.toml` is the thesis-structure profile. A uniquely
Pareto-dominant feasible action is selected directly. If no action dominates
all alternatives, the DQN chooses autonomously from the remaining actions. The
profile disables the auxiliary game-guidance loss.

`configs/paper-follower-game.toml` is the enhanced profile. It preserves the
same strict-dominance rule and autonomous DQN decision, then uses normalized
game confidence, DQN opposition, and public cloud pressure to arbitrate cases
where the two experts disagree. This profile is an improved model rather than
a literal transcription of the thesis.

## Constrained Stackelberg leader

The cloud remains revenue-oriented:

\[
\max_p \quad pN_{\mathrm{cloud}}(p).
\]

Capacity and deadline quality are constraints:

\[
W_{\mathrm{cloud}}(p)\leq C_{\mathrm{available}},
\qquad
r_{\mathrm{late}}(p)\leq\varepsilon.
\]

Candidate prices are evaluated with a normalized Lagrangian relaxation of
these constraints. The cloud does not maximize global success and does not
assign vehicle actions.

Service vehicles retain the thesis quote:

\[
P_{\mathrm{service}}
=P_{\mathrm{cloud}}\left(1-\phi(1-u_{\mathrm{cpu}})\right).
\]

Predicted demand may update the anticipated CPU utilization, but it does not
change the functional form. A service vehicle rejects a quote when energy,
queue, or reservation-utility constraints are violated.

## Serverless delay semantics

Both backends use the same physical compute and backlog model:

\[
T_{\mathrm{physical}}
=T_{\mathrm{preprocess}}+T_{\mathrm{radio}}
+T_{\mathrm{cloud\ compute}}+T_{\mathrm{physical\ queue}}.
\]

The live Knative backend adds measured client dispatch and platform overhead:

\[
T_{\mathrm{live}}
=T_{\mathrm{physical}}+T_{\mathrm{dispatch}}+T_{\mathrm{platform}}.
\]

The function's scaled hash workload is retained as a diagnostic measurement
and is never substituted for the physical term \(C/F_{\mathrm{cloud}}\).
