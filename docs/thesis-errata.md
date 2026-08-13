# Thesis-to-code errata

The corrected implementation resolves the following inconsistencies in the submitted document and legacy code:

- Chapter 4 specifies DQN and a three-action value function. The Actor/Critic wording in the implementation chapter is treated as a textual error.
- The final network is `20 -> 256 -> 128 -> 3`; legacy scripts used `64 -> 64`.
- Simulation steps and vehicle count are independent configuration values. `steps=2000` no longer requires editing source code.
- Vehicle count means an exact number of generated SUMO vehicle routes. Realized and peak-active counts are reported separately.
- DQN transitions are actually temporal, include `done` and the next feasible-action mask; Double DQN separates online selection from target evaluation.
- The implementation uses one parameter-shared DQN policy, but each vehicle supplies its own temporal transitions and private reward. It is therefore decentralized execution with shared parameters, not one cooperative fleet controller.
- The three Stackelberg stages are synchronized per simulation step: price and service quotes are published first, all task vehicles decide from one state snapshot, and only then does the batch enter the queues.
- Gradient frequency is explicit and independent from epsilon decay: every transition is stored, while optimizer updates use a configured interval.
- Payment is a cost metric, not a latency term.
- V2V/V2I radio units are separated into MHz, bit/s/Hz, and effective Mbit/s. The log-distance loss is `10*n*log10(d/d0)`; the earlier name `bandwidth_mbps` and its extra distance multiplier mixed channel width with final data rate.
- Serverless cold-start and queue behavior can be measured through a real Knative Service instead of being represented only by constants.
- Legacy spreadsheet values are preserved but are not considered verified outputs of the corrected code.
- The original six base-station coordinates are restored in the baseline. A separate improved layout is calibrated on seed 42 and evaluated on seed 11, so base placement is not selected from final outcomes.
