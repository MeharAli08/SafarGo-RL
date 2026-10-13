# SafarGo: RL Formulation, Research Notes, and Two-Week MVP

## 1. Problem formulation

**Agent:** a fleet controller making one aggregate repositioning decision every five minutes. A production design would need coordinated per-vehicle recommendations; this prototype deliberately uses one action over an aggregate fleet.

**Environment:** 40 zones in Rawalpindi and Islamabad, aggregate supply and stochastic passenger requests, plus stylized time-of-day, weather, and local demand pulses. The prototype uses synthetic data only.

**State:** a compact teaching state `(4-hour time block, weather flag, supply-imbalance bucket, demand bucket)`. The actual problem state should include per-zone supply, forecast requests, estimated pickup travel time, traffic, weather, calendar/event signals, current idle vehicles, and recent service/fairness outcomes.

**Actions:** select one of 40 destination zones or stay. The prototype moves at most one aggregated vehicle from the most supplied zone. A real system needs a coordinated destination assignment under capacity, route, driver-consent, and safety constraints.

**Reward:** `2 × completed rides − 1.5 × max(0, mean wait − 7) − 0.8 × reposition fuel − 0.5 × cancellations − 0.15 × service disparity`. In code, revenue is reported as a proxy but is not separately added to reward to avoid double-counting completed rides. We reward completed rides, penalize service above the 7-minute target, cancellations, fuel, and cross-zone disparity. These coefficients are illustrative and must be tuned with stakeholders and offline sensitivity analysis. Hard safety/service limits should be enforced as constraints, not trusted to a soft reward alone.

**Episode:** one simulated day (288 decisions, five minutes each); training uses many independent synthetic days with changing random seeds.

**Reward hacking risks:** a rides-only reward can ignore low-demand zones; a wait-only reward could waste fuel by spreading vehicles everywhere; a weak fuel penalty can cause excessive repositioning; optimizing a citywide average can hide a severely underserved neighborhood; rewarding requests rather than completed service may encourage artificial request-count behavior. Use zone-level minimum service constraints, monitor tail waits/cancellations/fuel, and audit by zone and time. Never assume a weighted reward guarantees fairness.

## 2. Research notes: deployed RL for ride-hailing

### DiDi order dispatching

Qin et al. describe the evolution of DiDi's order-dispatch system from combinatorial optimization to a semi-Markov decision process and deep RL, including production pilots and practical requirements such as reliability, latency, and multiple business objectives. A later KDD paper describes a scalable RL dispatcher evaluated with offline simulation and then deployed online in multiple cities; its abstract reports A/B testing and subsequent primary-mode deployment in one international market. The paper reports over 1.3% improvement in total driver income in A/B testing and up to 5.3% in selected metrics after broader deployment analysis. These are authors' reported results, not guaranteed transferable effects.

### DiDi vehicle repositioning

Jiao et al. describe deep RL plus decision-time planning for idle-vehicle repositioning, including large-fleet coordination features. Their paper reports a real-world experiment program comparing recommendations with experienced drivers' repositioning strategies on a major ride-hailing platform. It highlights the need for real-time execution, coordination, and a practical decision engine. Repositioning differs from matching a new request to a driver: it acts during idle time under uncertain future demand.

### Difference for SafarGo

The published cases work with much larger operational systems and/or richer data, engineering, and experimentation capacity. SafarGo has only three months of history, highly local traffic and calendar effects, and can expose at most 10% of its vehicles to a live test. Its fairness requirement across 40 zones also needs explicit zone-level constraints and reporting. A small history may miss rare weather/events and seasonal shifts. Therefore this MVP uses synthetic simulation for software learning only; it does not claim the published methods or outcomes will transfer. Before pilot decisions, SafarGo needs data-quality checks, time-based validation, an uncertainty-aware simulator, historical replay, and staged approval.

## 3. Q-learning example

Update rule: `Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]` (omit the future term at terminal states).

Example parameters: learning rate `α = 0.15`, discount `γ = 0.90`, epsilon-greedy exploration starts at `ε = 1.0`, decays by `0.995` after each training episode, and stops at `0.05`. A real data workflow should train offline and select hyperparameters on held-out days, not explore randomly in live service.

Numeric illustration: suppose `Q(s,a)=4`, reward `r=3`, best next value `max Q(s',·)=5`, `α=.15`, and `γ=.90`. Then target is `3 + .90×5 = 7.5`, TD error is `7.5−4=3.5`, and updated value is `4 + .15×3.5 = 4.525`.

A raw state over 40 vehicle positions, times, weather and demand explodes combinatorially. This prototype aggregates supply and bins features, yielding a small table (at most 6×2×5×5 states by 41 actions). It is useful to teach Q-learning and make the UI responsive. It cannot distinguish important zone layouts and its apparent state size understates the real problem. For production scale, use function approximation (e.g. neural value model, structured/graph model, or a contextual bandit/optimization baseline), offline evaluation, and explicit coordination. With only three months of data, a simple demand forecast plus constrained heuristic may be safer than deep RL at first.

## 4. Two-week MVP plan

- **Days 1–2:** define metrics, zone map abstraction, data dictionary, synthetic scenarios, and current-policy baseline.
- **Days 3–5:** implement simulator, demand/time/weather features, tests and reproducible scenarios (in a real project, validate the simulator against held-out history).
- **Days 6–7:** implement Q-learning and rule baselines; inspect reward sensitivity and failure cases.
- **Days 8–9:** add metrics and plots: completed rides, revenue proxy, mean and tail wait, cancellations, fuel proxy, and per-zone service/fairness.
- **Days 10–11:** build Streamlit UI and make runs reproducible.
- **Days 12–13:** document limitations, deployment, and pilot/fallback playbook; review with operations and safety owners.
- **Day 14:** demonstrate simulation results; deploy the simulation UI to Streamlit Community Cloud if desired.

## References

1. Qin, Z. et al. (2020), “Ride-Hailing Order Dispatching at DiDi via Reinforcement Learning,” *INFORMS Journal on Applied Analytics*. [Publisher article](https://pubsonline.informs.org/doi/10.1287/inte.2020.1047).
2. Xu, Z. et al. (2022), “Reinforcement Learning in the Wild: Scalable RL Dispatching Algorithm Deployed in Ridehailing Marketplace,” *KDD 2022*. [ACM DOI](https://doi.org/10.1145/3534678.3539095), [arXiv record](https://arxiv.org/abs/2202.05118).
3. Jiao, Y. et al. (2021), “Real-world Ride-hailing Vehicle Repositioning using Deep Reinforcement Learning,” *Transportation Research Part C*. [Publisher page](https://www.sciencedirect.com/science/article/pii/S0968090X21003004), [arXiv record](https://arxiv.org/abs/2103.04555).

## Ethics and practical risks

- **Zone fairness:** city-wide averages can hide poor service in low-income or peripheral zones. Compare wait, cancellation and coverage metrics for every zone; use a minimum service constraint before any pilot.
- **Driver trust:** explain why a reposition suggestion was made, make acceptance voluntary during a pilot, and provide a way to report unsafe or impractical recommendations.
- **Privacy:** use aggregated, access-controlled trip data; remove direct identifiers and define retention limits. This MVP uses synthetic data only.
- **Operational risk:** a wrong recommendation can add fuel cost or worsen service. Start with historical replay and shadow mode, cap any approved test at 10% of vehicles, and keep a fast fallback to current dispatch.

## Code layout used by this project

The runnable project keeps `app.py`, `environment.py`, `agent.py`, `baselines.py`, and `evaluation.py` side by side in the project root. This keeps the files easy to locate for a beginner while separating the interface, simulator, learning agent, baseline rules, and evaluation loop.
