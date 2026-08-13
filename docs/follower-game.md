# Learning-based follower Stackelberg model

## Model boundary

This profile refines the thesis Hybrid without changing its action space, DQN architecture, task variables, or decentralized decision boundary. The DQN remains `20 -> 256 -> 128 -> 3`; payment remains part of both state and reward. A shared network is parameter sharing, not central action assignment: each task vehicle evaluates its own state and chooses its own action.

The comparison is explicit:

- Random and Greedy use the queue price and their named action rules.
- DQN uses the queue price and its learned autonomous policy.
- Stackelberg uses follower-responsive pricing and the finite-utility best response.
- Hybrid uses follower-responsive pricing and load-adaptive arbitration between the finite-utility response and autonomous DQN values.

This separation gives Hybrid an actual game-learning interaction. Applying the same follower-responsive pricing to pure DQN would already make it a bilevel game-learning method and would erase the algorithmic distinction being tested.

## Three synchronous stages

At each time step all decisions use one immutable vehicle, queue, channel, and workload snapshot. The cloud response solver additionally broadcasts the expected queue rank and full-batch capacity pressure implied by the anticipated simultaneous demand.

1. The cloud broadcasts one price selected from a bounded candidate set.
2. Each idle service vehicle independently accepts or rejects work and quotes from its utilization, residual energy, the cloud reference price, and anticipated demand.
3. Each task vehicle independently chooses Local, V2V, or V2I.

Candidate-price probes are hypothetical evaluations only. They do not enqueue work, consume energy, advance epsilon, or create replay transitions. Only the final action changes the environment.

## Leader response

For price \(p\), the current Hybrid follower policy supplies an anticipated cloud probability for every task. Game evidence is compared with normalized DQN opposition; ambiguous learned responses use a temperature-scaled softmax over currently feasible Q values. During early training, this prediction moves continuously from the finite-utility response to the learned response as replay warm-up completes.

Let \(d(p)\) be predicted cloud cycles divided by total task cycles, \(d^*\) the current reserve target, and \(\ell(p)\) the predicted fraction of cloud requests already expected to miss their deadline. The cloud maximizes

\[
U_c(p)=
\frac{\max(0,p-p_0)}{p_{\max}-p_0}d(p)
-w_c\left[\frac{\max(0,d(p)-d^*)}{d^*}\right]^2
-w_\ell\ell(p).
\]

The first term is normalized revenue above the configured base marginal price. The second prevents profit seeking from silently exceeding the cloud reserve. The third prevents the leader from treating already-late work as useful demand. The chosen candidate is smoothed against the previous broadcast price. For each candidate, cloud demand and synchronous-batch queue delay are closed by relaxed fixed-point iterations. This prevents every vehicle from evaluating V2I as if it were first in the same new batch.

V2V topology and physical delays do not depend on price. When a price makes the previously selected service vehicle reject work, the implementation performs an exact fallback V2V search over the remaining quotes.

## Vehicle response and fusion

The private finite utility retains the thesis components:

\[
J_a=w_t\frac{T_a}{D}
+w_e\frac{E_a}{E_0}
+w_p\frac{P_a}{P_0}.
\]

The game action is the feasible action with minimum \(J_a\). Its confidence is

\[
c=\frac{J_{(2)}-J_{(1)}}{\max(|J_{(1)}|,1)}.
\]

Hybrid compares this normalized game evidence with the DQN's normalized long-term advantage over the game action. If the two experts disagree about using V2I, predicted overload strengthens the evidence for the non-cloud action. The vehicle uses the game response when both experts agree or when the game evidence remains stronger; otherwise it uses its own DQN action. This avoids both an arbitrary linear mixture of differently scaled values and a central action allocator. The complete derivation is in [load-adaptive Hybrid arbitration](hybrid-adaptive-arbitration.md).

## Training

The main loss remains Double-DQN Huber temporal-difference loss. For a high-confidence feasible game action \(a_g\), a small auxiliary term is added:

\[
L=L_{\mathrm{TD}}+\lambda
\max\left(0,m-Q(s,a_g)+\max_{a\in A_s\setminus a_g}Q(s,a)\right).
\]

The feasible action mask \(A_s\) is stored with the guidance, so an unavailable action is never used as a ranking rival. Under adaptive arbitration the auxiliary weight is zero when DQN evidence defeats the game evidence; this prevents the regularizer from teaching the policy to reproduce a rejected congested response. Pure DQN transitions carry zero guidance weight.

## Diagnostics and interpretation

Each run writes `pricing.jsonl` with selected price, predicted cloud share and request count, target share, predicted late share, and leader score. Task records contain `cloud_price`, `game_action`, `game_confidence`, `hybrid_game_evidence`, `hybrid_dqn_evidence`, `hybrid_q_opposition`, and `hybrid_cloud_pressure`.

This model should be judged with ablations, not only final success rate:

- DQN versus Hybrid tests whether game information adds value.
- Stackelberg versus Hybrid tests whether learning adds value beyond the finite utility.
- predicted versus realized cloud share tests leader calibration.
- DQN-decision ratio and confidence distribution test whether the learned component is materially active.
- success, successful-task latency, energy, and payment must be reported together.
