# ERCS Project Guide

Emergency Response Coordination Simulator — MSc Dissertation, University of Liverpool (2026).

## 1. What This Project Does

ERCS is a discrete-event simulation that answers one research question:

> Can adaptive scheduling algorithms integrated with delay-tolerant communication
> architectures improve emergency resource coordination effectiveness when operating
> under intermittent connectivity conditions?

It does this by simulating 50 emergency responder nodes in a disaster zone, generating
tasks via Poisson arrivals, routing assignment messages through a PRoPHET-inspired
delay-tolerant network, and comparing two coordination algorithms across three
connectivity degradation levels — running the whole thing 180 times for statistical
rigour.

---

## 2. Architecture Overview

The system is built in 6 phases, each depending on the ones below it:

```
Phase 6  VISUALIZATION       plots.py, dashboard.py, notebook, animation.py, diagnostics.py
Phase 5  SIMULATION ENGINE   engine.py — event queue, orchestration
Phase 4  COORDINATION        algorithms.py — Adaptive vs Baseline
Phase 3  SCENARIOS           generator.py — Poisson task arrivals
Phase 2  COMMUNICATION       prophet.py — PRoPHET DTN, buffers, predictability
Phase 1  NETWORK             topology.py + mobility.py — 50 nodes, edges, movement
```

### File Map

| File | Phase | What it does |
|------|-------|-------------|
| `src/ercs/config/parameters.py` | 1-4 | Pydantic config models, `SimulationConfig` |
| `src/ercs/config/schemas.py` | 1-4 | YAML schema validation and loading |
| `src/ercs/network/topology.py` | 1 | `NetworkTopology`, `TopologyGenerator`, node placement, edge management |
| `src/ercs/network/mobility.py` | 1 | `MobilityManager`, Random Waypoint model, `calculate_encounters()` |
| `src/ercs/communication/prophet.py` | 2 | PRoPHET protocol, message buffers, predictability matrix |
| `src/ercs/scenario/generator.py` | 3 | `ScenarioGenerator`, Poisson task arrivals, `Task` dataclass |
| `src/ercs/coordination/algorithms.py` | 4 | `AdaptiveCoordinator`, `BaselineCoordinator`, `CoordinationManager` |
| `src/ercs/simulation/engine.py` | 5 | `SimulationEngine`, `ExperimentRunner`, event dispatch |
| `src/ercs/evaluation/metrics.py` | 6 | `PerformanceEvaluator`, `StatisticalAnalyzer`, t-tests, ANOVA, effect sizes |
| `src/ercs/visualization/plots.py` | 6 | Shared matplotlib figures (thesis-quality) |
| `src/ercs/visualization/animation.py` | 6 | Real-time side-by-side animation with frame capture |
| `src/ercs/visualization/diagnostics.py` | 6 | PRoPHET network graphs, message journey tracking |
| `app/dashboard.py` | 6 | Streamlit interactive dashboard (6 tabs) |
| `notebooks/experiment_report.ipynb` | 6 | Jupyter notebook for static thesis-quality analysis |
| `scripts/run_experiment.py` | — | CLI experiment runner (config validation) |
| `scripts/run_animation.py` | — | CLI with 6 visualization modes |
| `scripts/diagnose_coordination.py` | — | Single coordination cycle comparison |
| `scripts/diagnose_encounters.py` | — | Network encounter and connectivity diagnostics |
| `tests/conftest.py` | — | Centralised test constants (mirrors spec defaults) |
| `configs/default.yaml` | — | Default experiment configuration |

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
│   │   ├── parameters.py              # All Pydantic config models (528 lines)
│   │   └── schemas.py                 # YAML schema validation & loading
│   ├── network/
│   │   ├── topology.py                # NetworkTopology, TopologyGenerator
│   │   └── mobility.py                # MobilityManager, Random Waypoint
│   ├── communication/
│   │   └── prophet.py                 # CommunicationLayer, PRoPHET routing
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
│   ├── diagnose_coordination.py       # Coordination cycle inspector
│   └── diagnose_encounters.py         # Network encounter diagnostics
├── configs/
│   └── default.yaml                   # Default experiment configuration
├── docs/
│   └── GUIDE.md                       # This file
├── tests/                             # 281 passing tests
│   ├── conftest.py
│   ├── test_network.py
│   ├── test_communication.py
│   ├── test_scenario.py
│   ├── test_coordination.py
│   ├── test_simulation.py
│   ├── test_mobility.py
│   └── test_evaluation.py
└── outputs/
    └── figures/                       # Saved plots and visualizations
```

---

## 4. The Event Cascade

This is the core of the simulation — what triggers what and in what order.

### 4.1 Initialization (before any events fire)

When `SimulationEngine.run()` is called:

```
_initialize_components():
  1. generate_topology()              → 50 nodes (2 coordination + 48 mobile)
                                        Coordination nodes placed in grid within
                                        coordination zone (50×50m at 800,300).
                                        Mobile responders placed randomly in
                                        incident zone (700×600m at 0,450).
                                        Edges between nodes within 100m radio range.

  2. CommunicationLayer()             → Per-node message buffers (25 MB each)
                                        Delivery predictability matrix (all zeros)
                                        PRoPHET parameters: P_init=0.75, β=0.25, γ=0.98

  3. _initialize_coordination_         → Seed P(coord, mobile) = 0.75 for all pairs.
     predictability()                   Coordination nodes have backhaul infrastructure,
                                        so they start with nonzero predictability to
                                        every mobile node. Without this, the Adaptive
                                        algorithm could never assign tasks (P = 0 everywhere).

  4. create_coordinator()             → AdaptiveCoordinator or BaselineCoordinator
     CoordinationManager()              Manages pending task queue, 30-min cycle timer.

  5. ScenarioGenerator.generate()     → ~200 tasks via Poisson process (2/min, ~100 min)
     manager.add_tasks()                All tasks queued for future coordination cycles.
                                        Each task has urgency (20% H, 50% M, 30% L)
                                        and a location within the incident zone.

  6. _initialize_mobility()           → Random Waypoint states for 48 mobile nodes.
                                        Speed 1–20 m/s, pause 0–30s at each waypoint.
                                        Movement covers the full simulation area (3000×1500m),
                                        enabling transit between incident and coordination zones.
```

### 4.2 Event Scheduling

All events are pre-scheduled into a priority queue sorted by timestamp.
The first `COORDINATION_CYCLE` fires at t=0 (assigns any tasks created at
t=0), then every 30 minutes thereafter. `TASK_CREATED` events are drawn from
the Poisson process and land at random times throughout the simulation.

| Event Type | Frequency | Count (~100 min run) |
|---|---|---|
| `MOBILITY_UPDATE` | Every 1 second | ~6,000 |
| `NODE_ENCOUNTER` | Every 10 seconds | ~600 |
| `TASK_CREATED` | Poisson (~2/min) | ~200 |
| `COORDINATION_CYCLE` | Every 30 minutes | 3–4 |

### 4.3 Main Loop

```python
while event_queue is not empty:
    event = pop earliest event
    if event.timestamp > simulation_duration: break
    process_event(event)
```

### 4.4 Event Processing — What Each Event Does

#### TASK_CREATED (Poisson arrivals, ~2/min)

```
Task arrives → increment total_tasks counter → record urgency
→ task sits in pending queue, waiting for next coordination cycle
```

No cascade. Tasks just wait.

#### COORDINATION_CYCLE (every 30 minutes)

```
Get pending tasks (created before now, status = PENDING)
  │
  ├─ ADAPTIVE: sort by urgency (HIGH > MED > LOW), then creation time
  │            for each task, score all reachable responders (P > threshold)
  │            using weighted formula: Score = 0.5 × P_norm + 0.5 × D_norm
  │            pick responder with highest combined score
  │
  └─ BASELINE: sort by creation time (FCFS)
               for each task, pick nearest responder by distance (any)
  │
  ▼
For each assignment:
  ├─ Record response_time = now − task.creation_time
  ├─ Create COORDINATION message (contains task_id, location, urgency)
  ├─ Store message in coordination node's buffer (round-robin across coord nodes)
  └─ Message now waits for encounters to be forwarded/delivered
```

#### MOBILITY_UPDATE (every 1 second) — physical movement

This handles node movement and detects **new** physical connections:

```
MobilityManager.step()
  └─ Each mobile node moves toward its Random Waypoint destination
     (speed 1–20 m/s, pause 0–30s at each waypoint)
     When a waypoint is reached, the node pauses then picks a new
     random destination anywhere in the simulation area (3000×1500m).
     │
     ▼
topology.update_node_position() for each moved node
topology.update_edges_from_positions()
  └─ Detects NEW connections (nodes just came within 100m)
     │
     ▼
For each new connection:
  _is_link_available() — deterministic infrastructure damage check
     │
     ▼ (if link available)
  communication_layer.process_encounter(node_a, node_b)
    ├─ Age predictabilities: P *= γ^k  where k = elapsed / update_interval
    ├─ Update P(a,b) — encounter:    P(a,b) += (1 − P(a,b)) × P_init
    ├─ Update P(a,c) — transitivity: P(a,c) += (1 − P(a,c)) × P(a,b) × P(b,c) × β
    └─ Exchange messages:
        ├─ Direct delivery if destination node present → MESSAGE_DELIVERED
        └─ Forward copy if other node has higher P to destination
```

#### NODE_ENCOUNTER (every 10 seconds) — sustained link reliability

This models **ongoing communication** over existing edges, subject to
infrastructure degradation (Karaman et al., 2026):

```
For each existing edge in topology:
  └─ Deterministic check: _is_link_available(node_a, node_b)
     Uses hash(pair, seed) to produce a consistent per-pair decision.
     The same pair always succeeds or fails for a given seed and
     connectivity level — this models permanent infrastructure damage,
     not random interference.
     ├─ 0.75 → 75% of pairs have a working link (mild degradation)
     ├─ 0.40 → 40% of pairs work (moderate, 24–48h post-earthquake)
     └─ 0.20 → 20% of pairs work (severe, early disaster phase)
     │
     ▼ (if link available)
  communication_layer.process_encounter()
    └─ Same cascade: aging → predictability update → message exchange

Also: expire_all_messages() removes messages past their 300-minute TTL
```

Together, MOBILITY_UPDATE and NODE_ENCOUNTER create two pathways for
message delivery: new connections as nodes physically move into range,
and repeated communication attempts over existing links filtered by the
connectivity level.

### 4.5 The Critical Path (how a task assignment actually reaches a responder)

```
Task created (Poisson)
  │ waits up to 30 min for next coordination cycle
  ▼
Coordination cycle assigns it → message created in coord node buffer
  │ message sits in buffer, waiting for encounters
  ▼
Encounter with intermediate node → PRoPHET forwards if better path
  │ may hop through multiple intermediate nodes
  ▼
Encounter with assigned responder → message delivered
  │
  ▼
MESSAGE_DELIVERED recorded in results
```

The connectivity level controls how hard the delivery step is. At 75%, messages
flow relatively easily. At 20%, many messages expire (TTL = 300 min = 18,000s)
before they can reach their destination — this is where Adaptive should
outperform Baseline because it considers reachability when assigning tasks.

### 4.6 PRoPHET Predictability Equations (Kumar et al., 2023)

Three equations govern the delivery predictability matrix `P`:

**Encounter** — when nodes A and B come into contact:

```
P(A,B) = P(A,B)_old + (1 − P(A,B)_old) × P_init
```

Predictability increases toward 1.0 with each encounter. Nodes that meet
frequently develop high mutual predictability. Applied symmetrically for
both directions.

**Aging** — time decay applied before each encounter:

```
P(A,B) = P(A,B)_old × γ^k    where k = elapsed_seconds / update_interval
```

Predictability decays over time if nodes do not meet. With γ = 0.98 and a
30-second aging interval, predictability halves after ~34 intervals (~17 min).
Values below 0.001 are pruned to save memory.

**Transitivity** — when A encounters B, update A's predictability to C
through B:

```
P(A,C) = P(A,C)_old + (1 − P(A,C)_old) × P(A,B) × P(B,C) × β
```

If B frequently meets C, then A encountering B indirectly improves A's
path to C. The `(1 − P(A,C)_old)` damping factor prevents the value from
exceeding 1.0. This enables multi-hop forwarding decisions.

These equations are applied inside `process_encounter()`, which is called
by both MOBILITY_UPDATE (new connections) and NODE_ENCOUNTER (sustained links).

### 4.7 Message Transfer Logic

When two nodes encounter each other, message exchange follows this logic
for each message in the sender's buffer:

1. Skip if expired (TTL exceeded)
2. Skip if recipient already delivered it (prevents duplicates)
3. Skip if recipient already has a copy in its buffer
4. **Direct delivery** if the recipient IS the message destination —
   mark as DELIVERED, remove from sender, record delivery time
5. **Forwarding** if the recipient has higher P to the destination than
   the sender — create a copy, increment hop count, store in recipient's
   buffer (sender keeps its copy for other encounters)
6. If recipient's buffer is full, apply drop policy (DROP_OLDEST by default)

Transmission time = `message_size_bytes × 8 / transmit_speed_bps`
(512 KB at 2 Mbps = ~2.048 seconds).

---

## 5. The Two Algorithms

### Adaptive (urgency-first, network-aware)

1. **Task ordering**: Sort by urgency (HIGH > MEDIUM > LOW), then by creation time
   within the same urgency tier.
2. **Responder filtering**: Only consider responders where
   `P(coordination_node, responder) > threshold` (default threshold = 0.0,
   meaning any nonzero predictability). This excludes unreachable responders.
3. **Weighted scoring**: For each candidate responder, compute a combined score:
   ```
   Score = α × P_norm + β × D_norm
   ```
   where:
   - `α = 0.5` (predictability weight)
   - `β = 0.5` (proximity weight)
   - `P_norm = predictability / max_predictability` (normalised 0–1)
   - `D_norm = 1.0 − (distance / max_distance)` (inverted: closer = higher)
4. **Selection**: Pick the responder with the highest combined score.
5. **Trade-off**: May fail to assign a task if no responder has P > 0, but
   assignments that are made balance reachability and proximity.

### Baseline (FCFS, proximity-only)

1. **Task ordering**: Sort by creation time (first-come, first-served).
   No urgency consideration.
2. **Responder selection**: Pick the nearest responder by Euclidean distance,
   regardless of whether a communication path exists.
3. **Trade-off**: Always assigns all tasks (if responders exist), but assignment
   messages may never reach unreachable responders.

### Key Difference

Adaptive is **conservative and network-aware**: it prioritises urgent tasks and
balances proximity with communication reachability using a weighted score.
Baseline is **optimistic and proximity-only**: it always assigns to the nearest
responder and hopes the message gets through. The research question is whether
the network-aware approach yields better real-world outcomes (delivery rate,
response time) under degraded connectivity.

---

## 6. The Experiment Matrix

```
2 algorithms × 3 connectivity levels × 30 runs = 180 total simulation runs
```

| | Connectivity 75% | 40% | 20% |
|---|---|---|---|
| **Adaptive** | 30 runs | 30 runs | 30 runs |
| **Baseline** | 30 runs | 30 runs | 30 runs |

Each run uses a deterministic seed: `base_seed (42) + run_offset`.

The 30 runs per configuration satisfy the Central Limit Theorem requirement
for statistical inference (Law, 2015).

---

## 7. Configuration Reference

All configuration is defined in Pydantic models in `src/ercs/config/parameters.py`.
The root model is `SimulationConfig`, composed of four parameter groups.

### Network (Phase 1)

| Parameter | Default | Source |
|---|---|---|
| Total nodes | 50 (2 coordination + 48 mobile) | Ullah & Qayyum (2022) |
| Simulation area | 3000 × 1500 m | Ullah & Qayyum (2022) |
| Incident zone | 700 × 600 m, origin (0, 450) | Ullah & Qayyum (2022) |
| Coordination zone | 50 × 50 m, origin (800, 300) | Design decision |
| Radio range | 100 m | Ullah & Qayyum (2022) |
| Buffer size | 25 MB per node (26,214,400 bytes) | Ullah & Qayyum (2022) |
| Message size | 500 KB (512,000 bytes) | Kumar et al. (2023) |
| Connectivity scenarios | [0.75, 0.40, 0.20] | Karaman et al. (2026) |
| Mobility model | Random Waypoint | Ullah & Qayyum (2022) |
| Speed range | 0–20 m/s (min 1 m/s enforced) | Ullah & Qayyum (2022) |
| Sensitivity node counts | [30, 50, 70] | Design decision |

### Communication / PRoPHET (Phase 2)

| Parameter | Default | Source |
|---|---|---|
| P_init | 0.75 | Kumar et al. (2023) |
| Beta (transitivity) | 0.25 | Kumar et al. (2023) |
| Gamma (aging) | 0.98 | Kumar et al. (2023) |
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
| Urgency distribution | 20% HIGH, 50% MEDIUM, 30% LOW | Li et al. (2025) |
| Runs per config | 30 | Law (2015) |

### Coordination (Phase 4)

| Parameter | Default | Source |
|---|---|---|
| Update interval | 1,800 seconds (30 min) | Kaji et al. (2025) |
| Priority levels | 3 | Rosas et al. (2023) |
| Path threshold | 0.0 (any P > 0) | Ullah & Qayyum (2022) |
| Proximity method | Euclidean | Keykhaei et al. (2024) |
| Adaptive task order | urgency_first | Kaji et al. (2025) |
| Baseline task order | FCFS | Design decision |
| Predictability weight (α) | 0.5 | Kumar et al. (2023) |
| Proximity weight (β) | 0.5 | Kumar et al. (2023) |

### Root Config

| Parameter | Default | Description |
|---|---|---|
| `experiment_name` | "ercs_experiment" | Label for outputs |
| `random_seed` | None | Base seed for reproducibility |
| `output_directory` | "./outputs" | Results directory |
| `log_level` | "INFO" | Logging verbosity |

**Computed properties** on `SimulationConfig`:
- `total_experimental_runs` → 2 × 3 × 30 = 180
- `total_nodes` → coordination_node_count + mobile_responder_count = 50
- `get_message_transmission_time_seconds()` → ~2.048 seconds

### Connectivity Scenarios

| Level | Meaning |
|---|---|
| 0.75 (75%) | Mild degradation — near-normal operation |
| 0.40 (40%) | Moderate — 24–48 hours post-earthquake |
| 0.20 (20%) | Severe — early disaster, critical paths only |

---

## 8. What Gets Measured

Each run produces a `SimulationResults` with these metrics:

| Metric | Formula | What it means |
|---|---|---|
| `delivery_rate` | messages_delivered / messages_created | Did the message reach the responder? |
| `assignment_rate` | tasks_assigned / total_tasks | Did the algorithm assign the task? |
| `avg_response_time` | mean(assignment_time − creation_time) | How long did the task wait for assignment? |
| `avg_delivery_time` | mean(delivery_time − creation_time) | How long until the message physically arrived? |

Additional raw counts are tracked: `total_tasks`, `tasks_assigned`,
`tasks_by_urgency`, `messages_created`, `messages_delivered`, `messages_expired`,
plus per-task `response_times` and `delivery_times` lists.

The `PerformanceEvaluator` (in `metrics.py`) then computes:

- **Descriptive stats**: n, mean, std, median, min, max, 95% CI per group
  (CI uses t-distribution: `mean ± t_crit(α/2, df=n-1) × SE`)
- **Independent t-tests** (Welch's): Adaptive vs Baseline at each connectivity level
  - Welch-Satterthwaite degrees of freedom (unequal variances assumed)
  - Cohen's d effect size with pooled standard deviation
  - Interpretation: negligible (<0.2), small (<0.5), medium (<0.8), large (≥0.8)
- **One-way ANOVA**: effect of connectivity within each algorithm
  - Eta-squared (η²) effect size
  - Interpretation: negligible (<0.01), small (<0.06), medium (<0.14), large (≥0.14)
- **Improvement percentage**: `(adaptive_mean − baseline_mean) / baseline_mean × 100`

All tests use α = 0.05 significance level.

---

## 9. How to Run

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
print(f"Avg response time: {results.average_response_time:.1f}s")
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
1. **Parameters** — 4 tables showing all config values with literature sources
2. **Run Experiment** — Start button, live progress bar with ETA, summary stats
3. **Visualizations** — Grouped bars, box plots, heatmaps, degradation lines
4. **Network Diagnostics** — PRoPHET graphs, predictability heatmaps, message journeys
5. **Statistical Analysis** — Welch's t-test table, ANOVA table, effect sizes
6. **Key Findings** — Headline metrics, largest advantage, significance summary

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

# Predictability heatmap (coord→mobile matrix)
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
```

### CLI experiment runner

```bash
# Validate configuration without running
python scripts/run_experiment.py --config configs/default.yaml --dry-run
```

### Tests

```bash
pytest                      # All 281 tests
pytest --cov=ercs           # With coverage report
pytest -m "not slow"        # Exclude slow tests
```

---

## 10. Spatial Layout

```
 0        700  850                                                    3000
 ┌─────────┬───┬─────────────────────────────────────────────────────┐
 │         │   │                                                     │ 1500
 │         │   │                                                     │
 │ INCIDENT│   │                                                     │
 │  ZONE   │   │            SIMULATION AREA                          │
 │ 700×600 │   │            (3000 × 1500m)                           │
 │ y:450–  │   │                                                     │
 │   1050  │   │            Mobile nodes traverse                    │
 │         │┌──┤            the full area via                        │
 │ Tasks   ││CO│            Random Waypoint.                         │
 │ appear  ││RD│ ← ~100m gap                                        │
 │ here.   ││50│   between zones                                    │
 │         │└──┤   (800, 300)                                        │
 ├─────────┘   │                                                     │
 │             │                                                     │
 └─────────────┴─────────────────────────────────────────────────────┘
 Radio range: 100m — nodes must be this close to form an edge
```

The coordination zone at (800, 300) sits ~100m from the incident zone
boundary. This separation requires relay hops for message delivery,
exercising the PRoPHET routing protocol. Mobile responders start in the
incident zone but move throughout the full simulation area via Random
Waypoint, creating encounter opportunities with coordination nodes and
other responders.

---

## 11. Visualization Reference

### Shared Plotting Functions (`src/ercs/visualization/plots.py`)

All figures use a thesis style: serif font, 12pt labels, 16pt titles, no
top/right spines, light grid, 300 dpi for saved files.

| Function | What it produces |
|---|---|
| `build_results_dataframe(results)` | 180-row DataFrame from SimulationResults list |
| `compute_summary_stats(df, metric)` | Mean + 95% CI per algorithm × connectivity |
| `build_parameter_tables(config)` | 4 DataFrames with values and literature sources |
| `build_ttest_table(report)` | Formatted Welch's t-test comparison table |
| `build_anova_table(report)` | Formatted one-way ANOVA results table |
| `plot_grouped_bars(summary, metric)` | Adaptive vs Baseline bars per connectivity |
| `plot_box_distributions(df)` | Box plots showing full distributions |
| `plot_heatmap(df)` | Algorithm × connectivity colour matrix |
| `plot_degradation_lines(summaries)` | Lines with CI bands showing degradation |
| `save_figure(fig, name, dir)` | Export to PNG at 300 dpi |

### Diagnostic Visualizations (`src/ercs/visualization/diagnostics.py`)

| Function | What it shows |
|---|---|
| `plot_predictability_graph(frame)` | Network graph with edges coloured by P value |
| `plot_predictability_heatmap(frame)` | Coord→mobile P matrix as colour grid |
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
|---|---|
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

```
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

## 12. Key Interfaces

These are the main integration points used by the visualization layer and
external scripts:

```python
# Run experiment
runner = ExperimentRunner(config, base_seed=42)
results = runner.run_all(progress_callback=fn)  # → list[SimulationResults]

# Build analysis DataFrame
df = build_results_dataframe(results)  # → DataFrame (180 rows typical)

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
|---|---|---|
| `NetworkStateProvider` | `get_delivery_predictability(from, to)` | Coordinator |
| `ResponderLocator` | `get_responder_position(id)`, `get_all_responder_ids()` | Coordinator |
| `NodePositionUpdater` | `update_node_position(id, x, y)` | MobilityManager |

`TopologyAdapter` (in `engine.py`) implements both `NetworkStateProvider` and
`ResponderLocator`, bridging the `NetworkTopology` and `CommunicationLayer` to
the coordination algorithms.
