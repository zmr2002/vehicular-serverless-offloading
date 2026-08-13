# Load-adaptive Hybrid arbitration

The cloud remains the Stackelberg leader: it broadcasts one price before the
vehicles in a simulation step make their decisions. Each task vehicle remains
an autonomous agent. The arbitration below only decides how that vehicle
combines its follower utility with its own DQN value estimates.

## Synchronous batch demand

All vehicles in one step decide from the same pre-arrival snapshot, so using
only the previous cloud queue makes every follower underestimate its position
in the new batch. For an anticipated batch of \(N_c\) cloud requests and an
existing backlog \(N_0\), a symmetric follower uses the expected rank

\[
N_0+\frac{N_c}{2}
\]

for its delay estimate. The capacity signal uses the full anticipated batch:

\[
u=\max\left(
\frac{N_0+N_c}{N_{\mathrm{instances}}N_{\mathrm{concurrency}}},
\frac{W_0+W_c}{N_{\mathrm{instances}}f_{\mathrm{cloud}}}
\right).
\]

Cloud demand affects queue delay, queue delay affects follower demand, and the
leader must therefore solve a fixed point rather than evaluate every vehicle
against the empty batch. Three relaxed response iterations with factor 0.5 are
used for each candidate price. The selected price, expected-rank delay and
full-batch utilization are then broadcast before autonomous actions are
chosen.

## Expert arbitration

For the finite feasible action set \(A\), let \(a_G\) be the follower
best-response action and let \(C_{(1)}\) and \(C_{(2)}\) be the lowest and
second-lowest generalized costs. The normalized game evidence is

\[
E_G =
\frac{(C_{(2)}-C_{(1)})/\max(|C_{(1)}|,1)}
     {\tau_G}.
\]

Let \(a_Q=\arg\max_{a\in A}Q(s,a)\). The normalized opposition of the DQN to
the game action is

\[
m_Q =
\frac{\max(0,Q(s,a_Q)-Q(s,a_G))}
     {\max_{a\in A}Q(s,a)-\min_{a\in A}Q(s,a)+\epsilon},
\qquad
E_Q = \rho\frac{m_Q}{\tau_Q},
\]

where \(\rho\in[0,1]\) is the policy reliability. It is one during frozen
evaluation and rises with replay-buffer warm-up during training. Normalizing
by the within-state Q range makes the comparison invariant to positive affine
changes in the Q-value scale.

The instantaneous follower cost does not contain the delay that one extra
cloud job imposes on later vehicles. The arbitration therefore uses predicted
cloud utilization \(u\), the configured utilization target \(u^\star\), and

\[
p=\max(0,u/u^\star-1), \qquad M=1+\lambda p.
\]

When the game selects V2I and the DQN selects a non-cloud action, \(E_Q\) is
multiplied by \(M\). When the DQN selects V2I and the game selects a non-cloud
action, \(E_G\) is multiplied by \(M\). Thus overload strengthens the evidence
for whichever expert avoids adding shared cloud work; it does not prohibit
V2I when both experts select it.

The vehicle uses \(a_G\) when both experts agree or \(E_G\ge E_Q\). Otherwise,
it uses its DQN action. The default values use the existing game confidence
threshold for both evidence scales and the Serverless capacity-utilization
target for \(u^\star\):

- \(\tau_G=0.15\)
- \(\tau_Q=0.15\)
- \(\lambda=1.0\)
- \(u^\star=0.85\)

These defaults preserve confident low-load follower decisions while allowing
learned long-term queue value to prevail when shared capacity is under
pressure. Per-task logs include both evidence values, Q opposition, and cloud
pressure so later calibration can be based on mechanism diagnostics rather
than success-rate-only parameter fitting.

## Online reliability (optional)

The fixed evaluation reliability \(\rho=1\) trusts any frozen checkpoint
equally, so a miscalibrated Q function with large margins can override the
game for long stretches without ever being right. With
`hybrid_online_reliability` set to `evaluate` (frozen evaluation only) or
`always`, the vehicle-shared runner keeps exponentially decayed counts of
resolved overrides that *flipped* task success relative to the game action's
decision-time estimate: an override is beneficial when its task succeeded
while the game action was estimated late, and harmful when its task failed
while the game action was estimated on time. Overrides that leave success
unchanged are neutral and never move the estimate, so learned congestion
avoidance whose benefit is collective rather than per-task is not penalized.

With decayed weights \(b\) and \(h\) (decay `hybrid_reliability_decay` per
step), the online factor

\[
\rho_{\mathrm{online}}
= \max\left(\rho_{\min}, \frac{1+b}{1+b+h}\right)
\]

multiplies the existing reliability before \(E_Q\) is formed;
\(\rho_{\min}\) is `hybrid_reliability_floor`. The factor starts at one, is
identical for every vehicle in a step, and is disabled (`off`) by default.

## Game adequacy (optional)

The refutation defense above is asymmetric: it can only distrust the DQN.
Under congestion this is wrong twice over. First, the collectively
beneficial overrides that route around a building queue fail occasionally at
the task level and would be counted as harmful, so the defense removes
exactly the behavior that wins at 2,000 vehicles. Second, the follower
margin \((C_{(2)}-C_{(1)})/|C_{(1)}|\) grows without bound as congestion
inflates cost differences, while the Q opposition is range-normalized to
\([0,1]\) — so the game's arbitration voice becomes loudest precisely where
its myopic estimate is least trustworthy.

With `hybrid_game_adequacy_arbitration` set to `evaluate` or `always`, the
runner also keeps decayed success/failure counts of decisions that *followed*
the game action. Their smoothed success rate

\[
A = \frac{1+s}{1+s+f}
\]

is the game's demonstrated adequacy. It enters the arbitration twice:

1. **Defense gating.** The online reliability is blended as
   \(\rho' = 1 - w(A)\,(1-\rho_{\mathrm{online}})\), where \(w(A)\) rises
   linearly from zero at `hybrid_adequacy_defense_floor` to one at
   `hybrid_adequacy_defense_full`. A refuted DQN stays suppressed only while
   following the game is empirically near-perfect (light load); once game
   decisions demonstrably fail, the defense withdraws.
2. **Evidence damping.** The game evidence is multiplied by
   \(A^{p}\) (`hybrid_adequacy_game_exponent`), smoothly handing authority to
   the learned expert exactly when following the game fails — the regime
   observed at 4,000 vehicles, where the arbitration otherwise returns
   control to a follower model that only achieves ~64% late-run success.

Defaults: floor 0.99, full 0.999, \(p = 8\). At demonstrated adequacy
\(A \ge 0.999\) the model behaves like the defended arbitration; at
\(A \approx 0.98\) (the 2,000-vehicle regime) the defense is inert and the
game evidence is damped by \(0.98^{8} \approx 0.85\); at \(A \approx 0.8\)
(late 4,000-vehicle windows) the game evidence shrinks to
\(0.8^{8} \approx 0.17\) of its undamped value.
