# ERCS Project Guide

Emergency Response Coordination Simulator -- MSc Dissertation, University of Liverpool (2026).

## 1. What This Project Does

ERCS is a discrete-event simulation that answers one research question:

> Can adaptive scheduling algorithms integrated with delay-tolerant communication
> architectures improve emergency resource coordination effectiveness when operating
> under intermittent connectivity conditions?

It does this by simulating 50 emergency responder nodes in a disaster zone, generating
tasks via Poisson arrivals, routing assignment messages through a PRoPHETv2
delay-tolerant network, and comparing two coordination algorithms across three
connectivity degradation levels -- running the whole thing 180 times for statistical
rigour.

---

## 2. Architecture Overview

The system is built in 6 phases, each depending on the ones below it:

```
Phase 6  VISUALIZATION       plots.py, dashboard.py, notebook, animation.py, diagnostics.py
Phase 5  SIMULATION ENGINE   engine.py -- event queue, orchestration
Phase 4  COORDINATION        algorithms.py -- Adaptive vs Baseline
Phase 3  SCENARIOS           generator.py -- Poisson task arrivals
Phase 2  COMMUNICATION       prophet.py -- PRoPHETv2 DTN, buffers, predictability
Phase 1  NETWORK             topology.py + mobility.py -- 50 nodes, edges, role-based movement
```

### File Map

| File | Phase | What it does |
|------|-------|-------------|
| `src/ercs/config/parameters.py` | 1-4 | Pydantic config models, `SimulationConfig` |
| `src/ercs/config/schemas.py` | 1-4 | YAML schema validation and loading |
| `src/ercs/network/topology.py` | 1 | `NetworkTopology`, `TopologyGenerator`, node placement, edge management |
| `src/ercs/network/mobility.py` | 1 | `MobilityManager`, role-based Random Waypoint, `calculate_encounters()` |
| `src/ercs/communication/prophet.py` | 2 | PRoPHETv2 protocol, message buffers, predictability matrix |
| `src/ercs/scenario/generator.py` | 3 | `ScenarioGenerator`, Poisson task arrivals, `Task` dataclass |
| `src/ercs/coordination/algorithms.py` | 4 | `AdaptiveCoordinator`, `BaselineCoordinator`, `CoordinationManager` |
| `src/ercs/simulation/engine.py` | 5 | `SimulationEngine`, `ExperimentRunner`, event dispatch |
| `src/ercs/evaluation/metrics.py` | 6 | `PerformanceEvaluator`, `StatisticalAnalyzer`, t-tests, ANOVA, effect sizes |
| `src/ercs/visualization/plots.py` | 6 | Shared matplotlib figures (thesis-quality) |
| `src/ercs/visualization/animation.py` | 6 | Real-time side-by-side animation with frame capture |
| `src/ercs/visualization/diagnostics.py` | 6 | PRoPHET network graphs, message journey tracking |
| `app/dashboard.py` | 6 | Streamlit interactive dashboard (6 tabs) |
| `notebooks/experiment_report.ipynb` | 6 | Jupyter notebook for static thesis-quality analysis |
| `scripts/run_experiment.py` | -- | CLI experiment runner (config validation) |
| `scripts/run_animation.py` | -- | CLI with 6 visualization modes |
| `scripts/pilot_experiment.py` | -- | Quick test harness (5 runs) |
| `scripts/diagnose_coordination.py` | -- | Single coordination cycle comparison |
| `scripts/diagnose_encounters.py` | -- | Network encounter and connectivity diagnostics |
| `scripts/diagnose_mobility_prophet.py` | -- | Mobility + PRoPHET diagnostics |
| `scripts/diagnose_warmup.py` | -- | Warm-up period P-value distribution analysis |
| `scripts/diagnose_delivery.py` | -- | Delivery pipeline diagnostics |
| `scripts/diagnose_recency.py` | -- | Encounter recency scoring diagnostics |
| `scripts/diagnose_seed49.py` | -- | Seed-specific delivery failure debugging |
| `scripts/validate_seeds.py` | -- | Seed connectivity validation utility |
| `tests/conftest.py` | -- | Centralised test constants (mirrors spec defaults) |
| `configs/default.yaml` | -- | Default experiment configuration |

---

## 3. Project Structure

```
resilient-emergency-response/
├── pyproject.toml                     # Project metadata, dependencies, tool config
├── requirements.txt                   # Pinned dependency versions
├── README.md
│
├── src/ercs/                          # Main package
│   ├── __init__.py
│   ├── config/
│   │   ├── parameters.py              # All Pydantic config models
│   │   └── schemas.py                 # YAML schema validation & loading
│   ├── network/
│   │   ├── topology.py                # NetworkTopology, TopologyGenerator
│   │   └── mobility.py                # MobilityManager, role-based Random Waypoint
│   ├── communication/
│   │   └── prophet.py                 # CommunicationLayer, PRoPHETv2 routing
│   ├── scenario/
│   │   └── generator.py               # ScenarioGenerator, Task dataclass
│   ├── coordination/
│   │   └── algorithms.py              # AdaptiveCoordinator, BaselineCoordinator
│   ├── simulation/
│   │   └── engine.py                  # SimulationEngine, ExperimentRunner
│   ├── evaluation/
│   │   └── metrics.py                 # PerformanceEvaluator, StatisticalAnalyzer
│   └── visualization/
│       ├── plots.py                   # Shared matplotlib plotting functions
│       ├── animation.py               # AnimationEngine, FrameData, side-by-side animation
│       └── diagnostics.py             # PRoPHET graphs, message journeys, heatmaps
│
├── app/
│   └── dashboard.py                   # Streamlit interactive dashboard
├── notebooks/
│   └── experiment_report.ipynb        # Thesis-quality static analysis notebook
├── scripts/
│   ├── run_experiment.py              # Full experiment runner
│   ├── run_animation.py               # 6-mode visualization CLI
│   ├── pilot_experiment.py            # Quick test harness
│   ├── diagnose_coordination.py       # Coordination cycle inspector
│   ├── diagnose_encounters.py         # Network encounter diagnostics
│   ├── diagnose_mobility_prophet.py   # Mobility + PRoPHET diagnostics
│   ├── diagnose_warmup.py             # Warm-up period analysis
│   ├── diagnose_delivery.py           # Delivery pipeline diagnostics
│   ├── diagnose_recency.py            # Encounter recency scoring diagnostics
│   ├── diagnose_seed49.py             # Seed-specific debugging
│   └── validate_seeds.py              # Seed connectivity validation
├── configs/
│   └── default.yaml                   # Default experiment configuration
├── docs/
│   └── GUIDE.md                       # This file
├── tests/                             # 384 passing tests
│   ├── conftest.py
│   ├── test_network.py
│   ├── test_mobility.py
│   ├── test_communication.py
│   ├── test_scenario.py
│   ├── test_coordination.py
│   ├── test_simulation.py
│   ├── test_evaluation.py
│   ├── test_warmup.py
│   ├── test_encounter_recency.py
│   ├── test_algorithm_fix.py
│   └── test_diagnostic_anomalies.py
└── outputs/
    └── figures/                       # Saved plots and visualizations
```

---

## 4. Cold-Start Evaluation (No Warm-Up)

The default configuration uses a cold-start approach: PRoPHET predictability values
start at P=0 for all node pairs and build organically through encounters during the
simulation itself.

**Why cold-start:**

- Emergency coordination systems activate from zero encounter history in real
  disasters -- cold-start is the realistic scenario
- Initialisation variability is addressed through multiple replications (30 runs
  per configuration) per Grassmann (2008)
- This avoids introducing artificial bias from pre-seeded P values

**What happens at t=0:**

- All P values are zero
- Tasks begin arriving immediately via Poisson process
- The Adaptive algorithm has no predictability information yet
- Early coordination cycles may have limited reachable candidates
- As the simulation progresses, encounters build P values and Adaptive's
  network-aware scoring becomes effective

**Optional warm-up:**

The `warmup_period_seconds` parameter can be set to a positive value (e.g. 1800)
in the config to delay tasks and coordination until encounter-based P values have
accumulated. When warm-up is enabled:

- Mobility updates and node encounters occur normally
- No tasks are generated and no coordination cycles execute
- After warm-up, tasks and cycles start with realistic P values

---

## 5. The Event Cascade

This is the core of the simulation -- what triggers what and in what order.

### 5.1 Initialization (before any events fire)

When `SimulationEngine.run()` is called:

```
_initialize_components():
  1. generate_topology()              -> 50 nodes (2 coordination + 48 mobile)
                                        Coordination nodes placed in grid within
                                        coordination zone (50x50m at 800,300).
                                        Mobile responders placed randomly in
                                        incident zone (700x600m at 0,450).
                                        Edges between nodes within 100m radio range.

  2. CommunicationLayer()             -> Per-node message buffers (25 MB each)
                                        Delivery predictability matrix (all zeros)
                                        PRoPHETv2 parameters: P_enc_max=0.5,
                                        i_typ=1800, beta=0.9, gamma=0.999885791.
                                        All predictabilities start at P=0 -- values
                                        build through encounters during simulation.

  3. create_coordinator()             -> AdaptiveCoordinator or BaselineCoordinator
     CoordinationManager()              Manages pending task queue, 30-min cycle timer.

  4. ScenarioGenerator.generate()     -> ~200 tasks via Poisson process (2/min, ~100 min)
     manager.add_tasks()                All tasks queued for future coordination cycles.
                                        Each task has urgency (20% H, 50% M, 30% L)
                                        and a location within the incident zone.

  5. _initialize_mobility()           -> Role-based Random Waypoint states for 48
                                        mobile nodes. Roles assigned deterministically
                                        by index: RESCUE ~60% (incident zone, 1-5 m/s),
                                        TRANSPORT ~25% (shuttle, 5-20 m/s),
                                        LIAISON ~15% (full area, 1-10 m/s).
```

### 5.2 Event Scheduling

All events are pre-scheduled into a priority queue sorted by timestamp.

If warm-up is enabled (warmup_period_seconds > 0), only `MOBILITY_UPDATE` and
`NODE_ENCOUNTER` events fire during warm-up. `COORDINATION_CYCLE` events start
after warm-up, and `TASK_CREATED` events are shifted by the warm-up duration.

With default cold-start (warmup_period_seconds=0), all events begin immediately.

| Event Type | When | Frequency | Count (~100 min, no warm-up) |
| --- | --- | --- | --- |
| `MOBILITY_UPDATE` | Entire simulation | Every 1 second | ~6,000 |
| `NODE_ENCOUNTER` | Entire simulation | Every 10 seconds | ~600 |
| `TASK_CREATED` | From t=0 (or after warm-up) | Poisson (~2/min) | ~200 |
| `COORDINATION_CYCLE` | From t=0 (or after warm-up) | Every 30 minutes | 3-4 |

### 5.3 Main Loop

```python
while event_queue is not empty:
    event = pop earliest event
    if event.timestamp > total_simulation_duration: break
    process_event(event)
```

### 5.4 Event Processing -- What Each Event Does

#### TASK_CREATED (Poisson arrivals, ~2/min)

```
Task arrives -> increment total_tasks counter -> record urgency
-> task sits in pending queue, waiting for next coordination cycle
```

No cascade. Tasks just wait.

#### COORDINATION_CYCLE (every 30 minutes)

```
Get pending tasks (created before now, status = PENDING)
  |
  +- ADAPTIVE: sort by urgency (HIGH > MED > LOW), then creation time
  |            for each task, score all reachable responders (P > 0.3)
  |            using weighted formula:
  |            Score = 0.2 x P_abs + 0.2 x R_norm + 0.6 x D_norm - 0.2 x W_penalty
  |            pick responder with highest combined score
  |
  +- BASELINE: sort by creation time (FCFS)
               for each task, pick nearest responder by distance (any)
  |
  v
For each assignment:
  +- Record response_time = now - task.creation_time
  +- Create COORDINATION message (contains task_id, location, urgency)
  +- Store message in coordination node's buffer (round-robin across coord nodes)
  +- Message now waits for encounters to be forwarded/delivered
```

#### MOBILITY_UPDATE (every 1 second) -- physical movement

This handles node movement and detects **new** physical connections:

```
MobilityManager.step()
  +- Each mobile node moves toward its destination
     Role determines speed/zone:
       RESCUE: 1-5 m/s, stays in incident zone
       TRANSPORT: 5-20 m/s, shuttles incident <-> coordination
       LIAISON: 1-10 m/s, roams full simulation area
     When a waypoint is reached, the node pauses then picks a new
     destination within its role's zone constraint.
     |
     v
topology.update_node_position() for each moved node
topology.update_edges_from_positions()
  +- Detects NEW connections (nodes just came within 100m)
     |
     v (if link available)
For each new connection:
  _is_link_available() -- deterministic infrastructure damage check
     |
     v (if link available)
  communication_layer.process_encounter(node_a, node_b)
    +- Age predictabilities: P *= gamma^k  where k = elapsed / update_interval
    +- Update P(a,b) -- encounter:    P_enc = P_enc_max x min(1, dt/I_typ)
    |                                 P(a,b) += (1 - P(a,b)) x P_enc
    +- Update P(a,c) -- transitivity: P(a,c) = max(P(a,c), P(a,b) x P(b,c) x beta)
    +- Exchange messages:
        +- Direct delivery if destination node present -> MESSAGE_DELIVERED
        +- Forward copy if other node has higher P to destination
```

#### NODE_ENCOUNTER (every 10 seconds) -- sustained link reliability

This models **ongoing communication** over existing edges, subject to
infrastructure degradation (Karaman et al., 2026):

```
For each existing edge in topology:
  +- Deterministic check: _is_link_available(node_a, node_b)
     Uses hash(pair, seed) to produce a consistent per-pair decision.
     The same pair always succeeds or fails for a given seed and
     connectivity level -- this models permanent infrastructure damage,
     not random interference.
     +- 0.75 -> 75% of pairs have a working link (mild degradation)
     +- 0.40 -> 40% of pairs work (moderate, 24-48h post-earthquake)
     +- 0.20 -> 20% of pairs work (severe, early disaster phase)
     |
     v (if link available)
  communication_layer.process_encounter()
    +- Same cascade: aging -> predictability update -> message exchange

Also: expire_all_messages() removes messages past their 300-minute TTL
```

Together, MOBILITY_UPDATE and NODE_ENCOUNTER create two pathways for
message delivery: new connections as nodes physically move into range,
and repeated communication attempts over existing links filtered by the
connectivity level.

### 5.5 The Critical Path (how a task assignment actually reaches a responder)

```
Task created (Poisson)
  | waits up to 30 min for next coordination cycle
  v
Coordination cycle assigns it -> message created in coord node buffer
  | message sits in buffer, waiting for encounters
  v
Encounter with intermediate node -> PRoPHETv2 forwards if better path
  | may hop through multiple intermediate nodes
  v
Encounter with assigned responder -> message delivered
  |
  v
MESSAGE_DELIVERED recorded in results
```

The connectivity level controls how hard the delivery step is. At 75%, messages
flow relatively easily. At 20%, many messages expire (TTL = 300 min = 18,000s)
before they can reach their destination -- this is where Adaptive should
outperform Baseline because it considers reachability when assigning tasks.

### 5.6 PRoPHETv2 Equations (Grasic et al., 2011)

Four equations govern the delivery predictability matrix `P`:

**Encounter probability** -- time-based calculation (PRoPHETv2 Eq. 1):

```
P_enc = P_enc_max x min(1, dt / I_typ)
```

If the node pair encountered each other recently (dt < I_typ), the encounter
probability is reduced proportionally. If they have not met for at least I_typ
seconds (1800s = 30 min), the full P_enc_max (0.5) is applied. This prevents
rapid re-encounters from inflating predictability.

**Encounter update** -- when nodes A and B come into contact:

```
P(A,B) = P(A,B)_old + (1 - P(A,B)_old) x P_enc
```

Predictability increases with each encounter. The `(1 - P_old)` factor provides
diminishing returns as P approaches 1.0. Applied symmetrically for both directions.

**Aging** -- time decay applied before each encounter:

```
P(A,B) = P(A,B)_old x gamma^k    where k = elapsed_seconds / update_interval
```

Predictability decays over time if nodes do not meet. With gamma = 0.999885791
and a 30-second aging interval, decay is very slow -- predictability halves
after approximately 6000 intervals (~50 hours). Values below 0.001 are pruned.

**Transitivity** -- MAX-based (PRoPHETv2, prevents saturation):

```
P(A,C) = max(P(A,C)_old, P(A,B) x P(B,C) x beta)
```

When A encounters B, this updates A's predictability to all nodes C that B
has encountered. The MAX operator (instead of the original additive formula)
prevents transitivity from inflating P values beyond what direct encounters
justify. This is the key PRoPHETv2 change that prevents delivery predictability
saturation.

These equations are applied inside `process_encounter()`, which is called
by both MOBILITY_UPDATE (new connections) and NODE_ENCOUNTER (sustained links).

### 5.7 Message Transfer Logic

When two nodes encounter each other, message exchange follows this logic
for each message in the sender's buffer:

1. Skip if expired (TTL exceeded)
2. Skip if recipient already delivered it (prevents duplicates)
3. Skip if recipient already has a copy in its buffer
4. **Direct delivery** if the recipient IS the message destination --
   mark as DELIVERED, remove from sender, record delivery time
5. **Forwarding** if the recipient has >= P to the destination than
   the sender -- create a copy, increment hop count, store in recipient's
   buffer (sender keeps its copy for other encounters)
6. If recipient's buffer is full, apply drop policy (DROP_OLDEST by default)

Transmission time = `message_size_bytes x 8 / transmit_speed_bps`
(512 KB at 2 Mbps = ~2.048 seconds).

---

## 6. The Two Algorithms

### Adaptive (urgency-first, network-aware)

1. **Task ordering**: Sort by urgency (HIGH > MEDIUM > LOW), then by creation time
   within the same urgency tier.
2. **Responder filtering**: Only consider responders where
   `P(coordination_node, responder) > 0.3`. This excludes responders without
   genuine encounter history (transitivity-only P values typically settle at
   0.05-0.20, while direct encounter values converge to ~0.45-0.50).
3. **Weighted scoring**: For each candidate responder, compute a combined score:
   ```
   Score = alpha x P_abs + gamma_r x R_norm + beta x D_norm - lambda x W_penalty
   ```
   where:
   - `alpha = 0.2` (predictability weight, Boondirek et al., 2014)
   - `gamma_r = 0.2` (encounter recency weight, Nelson et al., 2009)
   - `beta = 0.6` (proximity weight, Boondirek et al., 2014)
   - `lambda = 0.2` (workload penalty weight, Cui et al., 2022)
   - `P_abs = raw predictability` (already in [0, 1])
   - `R_norm = 1.0 - min(dt / 1800, 1.0)` (recency: recent encounters score higher)
   - `D_norm = 1.0 - (distance / simulation_diagonal)` (inverted: closer = higher)
   - `W_penalty = 1.0` if responder already assigned this cycle, else 0.0
4. **Selection**: Pick the responder with the highest combined score.
5. **Trade-off**: May fail to assign a task if no responder has P > 0.3, but
   assignments that are made balance reachability, recency, and proximity while
   distributing workload.

### Baseline (FCFS, proximity-only)

1. **Task ordering**: Sort by creation time (first-come, first-served).
   No urgency consideration.
2. **Responder selection**: Pick the nearest responder by Euclidean distance,
   regardless of whether a communication path exists.
3. **Trade-off**: Always assigns all tasks (if responders exist), but assignment
   messages may never reach unreachable responders.

### Key Difference

Adaptive is **conservative and network-aware**: it prioritises urgent tasks and
balances proximity with communication reachability and encounter recency using a
weighted score with workload spreading. Baseline is **optimistic and proximity-only**:
it always assigns to the nearest responder and hopes the message gets through. The
research question is whether the network-aware approach yields better real-world
outcomes (delivery rate, response time) under degraded connectivity.

---

## 7. The Experiment Matrix

```
2 algorithms x 3 connectivity levels x 30 runs = 180 total simulation runs
```

| | Connectivity 75% | 40% | 20% |
|---|---|---|---|
| **Adaptive** | 30 runs | 30 runs | 30 runs |
| **Baseline** | 30 runs | 30 runs | 30 runs |

Each run uses a deterministic seed: `base_seed (42) + run_offset`.

The 30 runs per configuration satisfy the Central Limit Theorem requirement
for statistical inference (Law, 2015).

---

## 8. Configuration Reference

All configuration is defined in Pydantic models in `src/ercs/config/parameters.py`.
The root model is `SimulationConfig`, composed of four parameter groups.

### Network (Phase 1)

| Parameter | Default | Source |
|---|---|---|
| Total nodes | 50 (2 coordination + 48 mobile) | Ullah & Qayyum (2022) |
| Simulation area | 3000 x 1500 m | Ullah & Qayyum (2022) |
| Incident zone | 700 x 600 m, origin (0, 450) | Ullah & Qayyum (2022) |
| Coordination zone | 50 x 50 m, origin (800, 300) | Design decision |
| Radio range | 100 m | Ullah & Qayyum (2022) |
| Buffer size | 25 MB per node (26,214,400 bytes) | Ullah & Qayyum (2022) |
| Message size | 500 KB (512,000 bytes) | Kumar et al. (2023) |
| Connectivity scenarios | [0.75, 0.40, 0.20] | Karaman et al. (2026) |
| Mobility model | role_based_waypoint | Ullah & Qayyum (2022); Aschenbruck et al. (2009) |
| Speed range | 0-20 m/s (min 1 m/s enforced) | Ullah & Qayyum (2022) |
| Sensitivity node counts | [30, 50, 70] | Design decision |

### Role-Based Mobility

| Role | Proportion | Zone | Speed | Pause |
| --- | --- | --- | --- | --- |
| RESCUE | ~60% | Incident zone only | 1-5 m/s | Long pauses |
| TRANSPORT | ~25% | Shuttle incident <-> coordination | 5-20 m/s | Short pauses |
| LIAISON | ~15% | Full simulation area | 1-10 m/s | Medium pauses |

Roles are assigned deterministically by node index: RESCUE for the first 60%
of mobile nodes, TRANSPORT for the next 25%, LIAISON for the remainder. Nodes
with `role=None` fall back to original Random Waypoint behaviour over the full
simulation area.

### Communication / PRoPHETv2 (Phase 2)

| Parameter | Default | Source |
|---|---|---|
| P_enc_max | 0.5 | Grasic et al. (2011) |
| I_typ (inter-encounter interval) | 1,800 seconds | Grasic et al. (2011) |
| Beta (transitivity) | 0.9 | Grasic et al. (2011) |
| Gamma (aging) | 0.999885791 | Grasic et al. (2011) |
| Update interval (aging) | 30 seconds | Kumar et al. (2023) |
| Message TTL | 18,000 seconds (300 min) | Ullah & Qayyum (2022) |
| Transmit speed | 2 Mbps (2,000,000 bps) | Ullah & Qayyum (2022) |
| Drop policy | DROP_OLDEST | Ullah & Qayyum (2022) |

### Scenario (Phase 3)

| Parameter | Default | Source |
|---|---|---|
| Task arrival model | Poisson | Pu et al. (2025) |
| Task rate | 2 per minute | Kumar et al. (2023) |
| Simulation duration | 6,000 seconds (~100 min) | Ullah & Qayyum (2022) |
| Warm-up period | 0 seconds (cold-start) | Grassmann (2008) |
| Urgency distribution | 20% HIGH, 50% MEDIUM, 30% LOW | Li et al. (2025) |
| Runs per config | 30 | Law (2015) |

### Coordination (Phase 4)

| Parameter | Default | Source |
|---|---|---|
| Update interval | 1,800 seconds (30 min) | Kaji et al. (2025) |
| Priority levels | 3 | Rosas et al. (2023) |
| Path threshold | 0.3 (P > 0.3) | Ullah & Qayyum (2022) |
| Proximity method | Euclidean | Keykhaei et al. (2024) |
| Adaptive task order | urgency_first | Kaji et al. (2025) |
| Baseline task order | FCFS | Design decision |
| Predictability weight (alpha) | 0.2 | Boondirek et al. (2014) |
| Recency weight (gamma_r) | 0.2 | Nelson et al. (2009) |
| Proximity weight (beta) | 0.6 | Boondirek et al. (2014) |
| Workload penalty (lambda) | 0.2 | Cui et al. (2022) |

### Root Config

| Parameter | Default | Description |
|---|---|---|
| `experiment_name` | "ercs_experiment" | Label for outputs |
| `random_seed` | None | Base seed for reproducibility |
| `output_directory` | "./outputs" | Results directory |
| `log_level` | "INFO" | Logging verbosity |

**Computed properties** on `SimulationConfig`:

- `total_simulation_duration` -> warmup_period_seconds + simulation_duration_seconds
- `total_experimental_runs` -> 2 x 3 x 30 = 180
- `total_nodes` -> coordination_node_count + mobile_responder_count = 50
- `get_message_transmission_time_seconds()` -> ~2.048 seconds

### Connectivity Scenarios

| Level | Meaning |
|---|---|
| 0.75 (75%) | Mild degradation -- near-normal operation |
| 0.40 (40%) | Moderate -- 24-48 hours post-earthquake |
| 0.20 (20%) | Severe -- early disaster, critical paths only |

---

## 9. What Gets Measured

Each run produces a `SimulationResults` with these metrics:

| Metric | Formula | What it means |
|---|---|---|
| `delivery_rate` | messages_delivered / messages_created | Did the message reach the responder? |
| `assignment_rate` | tasks_assigned / total_tasks | Did the algorithm assign the task? |
| `avg_response_time` | mean(assignment_time - creation_time) | How long did the task wait for assignment? |
| `avg_delivery_time` | mean(delivery_time - creation_time) | How long until the message physically arrived? |

Additional raw counts are tracked: `total_tasks`, `tasks_assigned`,
`tasks_by_urgency`, `messages_created`, `messages_delivered`, `messages_expired`,
plus per-task `response_times` and `delivery_times` lists.

The `PerformanceEvaluator` (in `metrics.py`) then computes:

- **Descriptive stats**: n, mean, std, median, min, max, 95% CI per group
  (CI uses t-distribution: `mean +- t_crit(alpha/2, df=n-1) x SE`)
- **Independent t-tests** (Welch's): Adaptive vs Baseline at each connectivity level
  - Welch-Satterthwaite degrees of freedom (unequal variances assumed)
  - Cohen's d effect size with pooled standard deviation
  - Interpretation: negligible (<0.2), small (<0.5), medium (<0.8), large (>=0.8)
- **One-way ANOVA**: effect of connectivity within each algorithm
  - Eta-squared (eta^2) effect size
  - Interpretation: negligible (<0.01), small (<0.06), medium (<0.14), large (>=0.14)
- **Improvement percentage**: `(adaptive_mean - baseline_mean) / baseline_mean x 100`

All tests use alpha = 0.05 significance level.

---

## 10. How to Run

### Prerequisites

```bash
# Python 3.10+ required (tested on 3.10, 3.11, 3.12)
pip install -e .              # Core dependencies
pip install -e ".[viz]"       # + Streamlit + Jupyter
pip install -e ".[dev]"       # + pytest, black, ruff, mypy
```

### Single simulation (Python API)

```python
from ercs.simulation import run_simulation

results = run_simulation("adaptive", connectivity_level=0.75, random_seed=42)
print(f"Delivery rate: {results.delivery_rate:.2%}")
print(f"Assignment rate: {results.assignment_rate:.2%}")
print(f"Avg decision time: {results.average_decision_time:.1f}s")
```

### Full experiment (180 runs)

```python
from ercs.simulation import run_experiment

results = run_experiment(runs_per_config=30, base_seed=42)
```

Or using the `ExperimentRunner` directly with a progress callback:

```python
from ercs.simulation import ExperimentRunner
from ercs.config.parameters import SimulationConfig

config = SimulationConfig()
runner = ExperimentRunner(config, base_seed=42)
results = runner.run_all(
    progress_callback=lambda current, total: print(f"{current}/{total}")
)
```

### Streamlit dashboard (interactive)

```bash
pip install -e ".[viz]"
streamlit run app/dashboard.py
```

The dashboard has 6 tabs:

1. **Parameters** -- 4 tables showing all config values with literature sources
2. **Run Experiment** -- Start button, live progress bar with ETA, summary stats
3. **Visualizations** -- Grouped bars, box plots, heatmaps, degradation lines
4. **Network Diagnostics** -- PRoPHET graphs, predictability heatmaps, message journeys
5. **Statistical Analysis** -- Welch's t-test table, ANOVA table, effect sizes
6. **Key Findings** -- Headline metrics, largest advantage, significance summary

Quick test mode (5 runs/config instead of 30) is available via a sidebar toggle.

### Jupyter notebook (static, thesis-quality figures)

```bash
jupyter notebook notebooks/experiment_report.ipynb
```

Produces 14+ publication-quality PNG figures saved to `outputs/figures/`.

### Animation and diagnostics (CLI)

```bash
# Side-by-side Adaptive vs Baseline animation (default mode)
python scripts/run_animation.py --duration 3600 --sample-interval 30

# PRoPHET predictability network graph at t=1800s
python scripts/run_animation.py --mode predictability --duration 3600 --time 1800

# Predictability heatmap (coord->mobile matrix)
python scripts/run_animation.py --mode heatmap --duration 3600 --time 1800

# Predictability evolution over time (top 10 node pairs)
python scripts/run_animation.py --mode evolution --duration 3600

# Message journey tracking (spatial path + timeline)
python scripts/run_animation.py --mode journey --duration 3600 --message-id first

# All message paths overview (green=delivered, red=failed)
python scripts/run_animation.py --mode paths --duration 3600

# Save to file
python scripts/run_animation.py --mode predictability --save outputs/pred.png

# Specific connectivity level
python scripts/run_animation.py --connectivity 0.20 --seed 42
```

### Coordination diagnostics

```bash
# Compare one coordination cycle between Adaptive and Baseline
python scripts/diagnose_coordination.py

# Network encounter frequency analysis
python scripts/diagnose_encounters.py --connectivity 0.20 --duration 6000

# Mobility + PRoPHET interaction diagnostics
python scripts/diagnose_mobility_prophet.py

# Warm-up period P-value distribution analysis
python scripts/diagnose_warmup.py

# Delivery pipeline diagnostics
python scripts/diagnose_delivery.py

# Encounter recency scoring diagnostics
python scripts/diagnose_recency.py
```

### CLI experiment runner

```bash
# Validate configuration without running
python scripts/run_experiment.py --config configs/default.yaml --dry-run
```

### Tests

```bash
pytest                      # All 384 tests
pytest --cov=ercs           # With coverage report
pytest -m "not slow"        # Exclude slow tests
```

---

## 11. Spatial Layout

```text
 0        700  850                                                    3000
 +----------+---+-----------------------------------------------------+
 |          |   |                                                     | 1500
 |          |   |                                                     |
 | INCIDENT |   |                                                     |
 |  ZONE    |   |            SIMULATION AREA                          |
 | 700x600  |   |            (3000 x 1500m)                           |
 | y:450-   |   |                                                     |
 |   1050   |   |            Mobile nodes traverse                    |
 |          |+--+            the area per their role:                  |
 | Tasks    ||CO|            RESCUE -> incident zone only              |
 | appear   ||RD|  <-- ~100m gap                                      |
 | here.    ||50|   between zones                                     |
 |          |+--+   (800, 300)                                        |
 +----------+   |            TRANSPORT -> shuttle both zones           |
 |              |            LIAISON -> full area                      |
 +--------------+-----------------------------------------------------+
 Radio range: 100m -- nodes must be this close to form an edge
```

The coordination zone at (800, 300) sits ~100m from the incident zone
boundary. This separation requires relay hops for message delivery,
exercising the PRoPHETv2 routing protocol. Mobile responders start in the
incident zone but move according to their role's zone constraints, creating
heterogeneous encounter patterns that PRoPHET can exploit.

---

## 12. Visualization Reference

### Shared Plotting Functions (`src/ercs/visualization/plots.py`)

All figures use a thesis style: serif font, 12pt labels, 16pt titles, no
top/right spines, light grid, 300 dpi for saved files.

| Function | What it produces |
| --- | --- |
| `build_results_dataframe(results)` | 180-row DataFrame from SimulationResults list |
| `compute_summary_stats(df, metric)` | Mean + 95% CI per algorithm x connectivity |
| `build_parameter_tables(config)` | 4 DataFrames with values and literature sources |
| `build_ttest_table(report)` | Formatted Welch's t-test comparison table |
| `build_anova_table(report)` | Formatted one-way ANOVA results table |
| `plot_grouped_bars(summary, metric)` | Adaptive vs Baseline bars per connectivity |
| `plot_box_distributions(df)` | Box plots showing full distributions |
| `plot_heatmap(df)` | Algorithm x connectivity colour matrix |
| `plot_degradation_lines(summaries)` | Lines with CI bands showing degradation |
| `save_figure(fig, name, dir)` | Export to PNG at 300 dpi |

### Diagnostic Visualizations (`src/ercs/visualization/diagnostics.py`)

| Function | What it shows |
| --- | --- |
| `plot_predictability_graph(frame)` | Network graph with edges coloured by P value |
| `plot_predictability_heatmap(frame)` | Coord->mobile P matrix as colour grid |
| `plot_predictability_evolution(frames)` | Time series of top node-pair P values |
| `plot_message_journey(msg_id, journey)` | Spatial path + hop timeline for one message |
| `plot_all_message_paths(frames, fwd_log)` | Overview map: green=delivered, red=failed |
| `find_message_journeys(fwd_log)` | Group forwarding entries by message ID |

### Animation (`src/ercs/visualization/animation.py`)

`AnimationEngine` subclasses `SimulationEngine` to capture per-frame snapshots
(`FrameData`) at configurable intervals (default 30s). `run_paired_simulation()`
runs Adaptive and Baseline sequentially with the same seed, then
`create_animation()` produces a side-by-side matplotlib animation showing
node movement, radio edges, task locations, message delivery flashes,
buffer utilisation, and live metrics counters.

### Colour Scheme

| Element | Colour |
| --- | --- |
| Adaptive algorithm | #2171B5 (blue) |
| Baseline algorithm | #CB181D (red) |
| Connectivity 75% | #2CA02C (green) |
| Connectivity 40% | #FF7F0E (orange) |
| Connectivity 20% | #D62728 (red) |
| Urgency HIGH | #D62728 (red) |
| Urgency MEDIUM | #FF7F0E (orange) |
| Urgency LOW | #2CA02C (green) |
| Coordination nodes | #9467BD (purple) |
| Mobile responders | #555555 (grey) |

### Output Figures

All saved to `outputs/figures/`:

```text
fig_delivery_rate_bars.png        fig_predictability_adaptive.png
fig_assignment_rate_bars.png      fig_predictability_baseline.png
fig_response_time_bars.png        fig_pred_heatmap_adaptive.png
fig_box_distributions.png         fig_pred_heatmap_baseline.png
fig_heatmap.png                   fig_pred_evolution_adaptive.png
fig_degradation_lines.png         fig_pred_evolution_baseline.png
                                  fig_journey_adaptive.png
                                  fig_journey_baseline.png
                                  fig_paths_adaptive.png
                                  fig_paths_baseline.png
```

---

## 13. Key Interfaces

These are the main integration points used by the visualization layer and
external scripts:

```python
# Run experiment
runner = ExperimentRunner(config, base_seed=42)
results = runner.run_all(progress_callback=fn)  # -> list[SimulationResults]

# Build analysis DataFrame
df = build_results_dataframe(results)  # -> DataFrame (180 rows typical)

# Compute stats and plot
summary = compute_summary_stats(df, "delivery_rate")
fig = plot_grouped_bars(summary, "delivery_rate")

# Statistical evaluation
evaluator = PerformanceEvaluator(results)
report = evaluator.generate_report(metrics=[MetricType.DELIVERY_RATE, ...])
ttest_df = build_ttest_table(report)
anova_df = build_anova_table(report)

# Diagnostics
frames_a, frames_b, fwd_a, fwd_b = run_paired_simulation(config, 0.20, seed=42)
journeys = find_message_journeys(fwd_a)
fig = plot_message_journey(msg_id, journeys[msg_id], frames_a, config)
```

### Protocols (dependency injection)

| Protocol | Methods | Used by |
| --- | --- | --- |
| `NetworkStateProvider` | `get_delivery_predictability(from, to)`, `get_last_encounter_time(from, to)` | Coordinator |
| `ResponderLocator` | `get_responder_position(id)`, `get_all_responder_ids()` | Coordinator |
| `NodePositionUpdater` | `update_node_position(id, x, y)` | MobilityManager |

`TopologyAdapter` (in `engine.py`) implements both `NetworkStateProvider` and
`ResponderLocator`, bridging the `NetworkTopology` and `CommunicationLayer` to
the coordination algorithms.
