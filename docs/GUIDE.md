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
Phase 6  VISUALIZATION       plots.py, dashboard.py, notebook, animation.py
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
| `src/ercs/network/topology.py` | 1 | `NetworkTopology`, node placement, edge management |
| `src/ercs/network/mobility.py` | 1 | `MobilityManager`, Random Waypoint model |
| `src/ercs/communication/prophet.py` | 2 | PRoPHET protocol, message buffers, predictability matrix |
| `src/ercs/scenario/generator.py` | 3 | `ScenarioGenerator`, Poisson task arrivals, `Task` dataclass |
| `src/ercs/coordination/algorithms.py` | 4 | `AdaptiveCoordinator`, `BaselineCoordinator`, `CoordinationManager` |
| `src/ercs/simulation/engine.py` | 5 | `SimulationEngine`, `ExperimentRunner`, event dispatch |
| `src/ercs/evaluation/metrics.py` | 6 | `PerformanceEvaluator`, t-tests, ANOVA, effect sizes |
| `src/ercs/visualization/plots.py` | 6 | Shared matplotlib figures (thesis-quality) |
| `src/ercs/visualization/animation.py` | 6 | Real-time side-by-side animation |
| `app/dashboard.py` | 6 | Streamlit interactive dashboard |
| `notebooks/experiment_report.ipynb` | 6 | Jupyter notebook for static analysis |
| `scripts/run_animation.py` | — | CLI to launch the animation |
| `tests/conftest.py` | — | Centralised test constants (mirrors spec defaults) |
| `configs/default.yaml` | — | Default experiment configuration |

---

## 3. The Event Cascade

This is the core of the simulation — what triggers what and in what order.

### 3.1 Initialization (before any events fire)

When `SimulationEngine.run()` is called:

```
_initialize_components():
  1. generate_topology()              → 50 nodes (2 coordination + 48 mobile)
                                        Edges between nodes within 100m radio range
  2. CommunicationLayer()             → Per-node message buffers (5 MB each)
                                        Delivery predictability matrix (all zeros)
  3. _initialize_coordination_         → Seed P(coord, mobile) = P_init for all pairs
     predictability()                   Coordination nodes have backhaul infrastructure,
                                        so they start with nonzero predictability to
                                        every mobile node. Without this, the Adaptive
                                        algorithm could never assign tasks (P = 0 everywhere).
  4. create_coordinator()             → AdaptiveCoordinator or BaselineCoordinator
     CoordinationManager()              Manages pending task queue
  5. ScenarioGenerator.generate()     → ~200 tasks via Poisson process (2/min, ~100 min)
     manager.add_tasks()                All tasks queued for future coordination cycles
  6. _initialize_mobility()           → Random Waypoint states for 48 mobile nodes
                                        Each starts at its topology position in the
                                        incident zone and picks a random destination.
```

### 3.2 Event Scheduling

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

### 3.3 Main Loop

```python
while event_queue is not empty:
    event = pop earliest event
    if event.timestamp > simulation_duration: break
    process_event(event)
```

### 3.4 Event Processing — What Each Event Does

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
  │            for each task, pick nearest responder WITH P > 0
  │
  └─ BASELINE: sort by creation time (FCFS)
               for each task, pick nearest responder (any)
  │
  ▼
For each assignment:
  ├─ Record response_time = now − task.creation_time
  ├─ Create COORDINATION message (contains task_id, location, urgency)
  ├─ Store message in coordination node's buffer
  └─ Message now waits for encounters to be forwarded/delivered
```

#### MOBILITY_UPDATE (every 1 second) — physical movement

This handles node movement and detects **new** physical connections:

```
MobilityManager.step()
  └─ Each mobile node moves toward its Random Waypoint destination
     (speed 1–20 m/s, pause 0–30s at each waypoint)
     │
     ▼
topology.update_node_position() for each moved node
topology.update_edges_from_positions()
  └─ Detects NEW connections (nodes just came within 100m)
     │
     ▼
For each new connection (deterministic — no connectivity filter):
  communication_layer.process_encounter(node_a, node_b)
    ├─ Age predictabilities: P *= γ^k  where k = elapsed / 30s
    ├─ Update P(a,b) — encounter:    P(a,b) += (1 − P(a,b)) × P_init
    ├─ Update P(a,c) — transitivity: P(a,c) += P(a,b) × P(b,c) × β
    └─ Exchange messages:
        ├─ Direct delivery if destination node present → MESSAGE_DELIVERED
        └─ Forward copy if other node has higher P to destination
```

New connections always trigger an encounter — they represent two radios
that just entered range of each other. The connectivity degradation filter
does not apply here; it only applies to sustained links (see below).

#### NODE_ENCOUNTER (every 10 seconds) — sustained link reliability

This models **ongoing communication** over existing edges, subject to
infrastructure degradation (Karaman et al., 2026):

```
For each existing edge in topology:
  └─ Probabilistic check: random() < connectivity_level
     ├─ 0.75 → 75% of attempts succeed (mild degradation)
     ├─ 0.40 → 40% succeed (moderate, 24–48h post-earthquake)
     └─ 0.20 → 20% succeed (severe, early disaster phase)
     │
     ▼ (if encounter succeeds)
  communication_layer.process_encounter()
    └─ Same cascade: aging → predictability update → message exchange

Also: expire_all_messages() removes messages past their 300-minute TTL
```

Together, MOBILITY_UPDATE and NODE_ENCOUNTER create two pathways for
message delivery: new connections as nodes physically move into range,
and repeated communication attempts over existing links filtered by the
connectivity level.

### 3.5 The Critical Path (how a task assignment actually reaches a responder)

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
outperform Baseline because it avoids assigning tasks to responders it cannot
reach (P > 0 filter).

### 3.6 PRoPHET Predictability Equations (Kumar et al., 2023)

Three equations govern the delivery predictability matrix `P`:

**Encounter** — when nodes A and B come into contact:

```
P(A,B) = P(A,B)_old + (1 − P(A,B)_old) × P_init
```

Predictability increases toward 1.0 with each encounter. Nodes that meet
frequently develop high mutual predictability.

**Aging** — time decay applied before each encounter:

```
P(A,B) = P(A,B)_old × γ^k    where k = elapsed_seconds / 30
```

Predictability decays over time if nodes do not meet. With γ = 0.98 and a
30-second aging interval, predictability halves after ~34 intervals (~17 min).

**Transitivity** — when A encounters B, update A's predictability to C
through B:

```
P(A,C) = P(A,C)_old + P(A,B) × P(B,C) × β
```

If B frequently meets C, then A encountering B indirectly improves A's
path to C. This enables multi-hop forwarding decisions.

These equations are applied inside `process_encounter()`, which is called
by both MOBILITY_UPDATE (new connections) and NODE_ENCOUNTER (sustained links).

---

## 4. The Two Algorithms

### Adaptive (urgency-first, network-aware)

1. **Task ordering**: Sort by urgency (HIGH first), then by creation time
2. **Responder selection**: For each task, consider only responders where
   `P(coordination_node, responder) > 0` (the PRoPHET predictability is nonzero,
   meaning a communication path exists). Among those, pick the nearest by
   Euclidean distance.
3. **Trade-off**: May fail to assign a task if no responder has P > 0, but
   assignments that are made have a higher chance of being delivered.

### Baseline (FCFS, proximity-only)

1. **Task ordering**: Sort by creation time (first-come, first-served)
2. **Responder selection**: Pick the nearest responder by Euclidean distance,
   regardless of whether a communication path exists.
3. **Trade-off**: Always assigns all tasks (if responders exist), but assignment
   messages may never reach unreachable responders.

### Key Difference

Adaptive is **conservative**: it only assigns when it believes the message can be
delivered. Baseline is **optimistic**: it always assigns and hopes for the best.
The research question is whether the conservative approach yields better real-world
outcomes (delivery rate, response time) under degraded connectivity.

---

## 5. The Experiment Matrix

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

## 6. Configuration Reference

### Network (Phase 1)

| Parameter | Default | Notes |
|---|---|---|
| Total nodes | 50 | 2 coordination + 48 mobile |
| Simulation area | 3000 × 1500 m | Full operational theatre |
| Incident zone | 700 × 600 m | Where tasks appear, responders start |
| Coordination zone | 50 × 50 m | Fixed infrastructure, origin (2900, 725) |
| Radio range | 100 m | Nodes must be this close to communicate |
| Buffer size | 5 MB per node | Message storage capacity |
| Message size | 512 KB | Per coordination message |
| Mobility model | Random Waypoint | Speed 1–20 m/s, pause 0–30s |

### Communication / PRoPHET (Phase 2)

| Parameter | Default | Notes |
|---|---|---|
| P_init | 0.75 | Initial predictability on encounter |
| Beta (transitivity) | 0.25 | Transitivity weight |
| Gamma (aging) | 0.98 | Time decay factor |
| Message TTL | 300 minutes | Expiry time |
| Transmit speed | 2 Mbps | Transfer rate between nodes |
| Aging interval | 30 seconds | Time unit for γ^k decay |
| Drop policy | DROP_OLDEST | When buffer is full |

### Scenario (Phase 3)

| Parameter | Default | Notes |
|---|---|---|
| Task arrival rate | 2 per minute | Poisson process |
| Simulation duration | 6,000 seconds | ~100 minutes (Ullah & Qayyum, 2022) |
| Urgency distribution | 20% H, 50% M, 30% L | Sampled per task |
| Runs per config | 30 | For statistical significance |

### Coordination (Phase 4)

| Parameter | Default | Notes |
|---|---|---|
| Update interval | 1,800 seconds | 30-minute coordination cycles |
| Path threshold | P > 0 | Adaptive only: minimum predictability |

### Connectivity Scenarios

| Level | Meaning |
|---|---|
| 0.75 (75%) | Mild degradation — near-normal operation |
| 0.40 (40%) | Moderate — 24–48 hours post-earthquake |
| 0.20 (20%) | Severe — early disaster, critical paths only |

---

## 7. What Gets Measured

Each run produces a `SimulationResults` with these metrics:

| Metric | Formula | What it means |
|---|---|---|
| `delivery_rate` | messages_delivered / messages_created | Did the message reach the responder? |
| `assignment_rate` | tasks_assigned / total_tasks | Did the algorithm assign the task? |
| `avg_response_time` | mean(assignment_time − creation_time) | How long did the task wait? |
| `avg_delivery_time` | mean(delivery_time − creation_time) | How long until message arrived? |

The `PerformanceEvaluator` then computes:
- **Descriptive stats**: mean, std, median, 95% CI per algorithm × connectivity group
- **Independent t-tests** (Welch's): Adaptive vs Baseline at each connectivity level
- **One-way ANOVA**: effect of connectivity within each algorithm
- **Effect sizes**: Cohen's d (t-tests), eta-squared (ANOVA)

---

## 8. How to Run

### Single simulation (Python API)

```python
from ercs.simulation import run_simulation

results = run_simulation("adaptive", connectivity_level=0.75, random_seed=42)
print(f"Delivery rate: {results.delivery_rate:.2%}")
print(f"Assignment rate: {results.assignment_rate:.2%}")
```

### Full experiment (180 runs)

```python
from ercs.simulation import run_experiment

results = run_experiment(runs_per_config=30, base_seed=42)
```

### Streamlit dashboard (interactive)

```bash
pip install -e ".[viz]"
streamlit run app/dashboard.py
```

### Jupyter notebook (static, thesis-quality figures)

```bash
jupyter notebook notebooks/experiment_report.ipynb
```

### Side-by-side animation (visual verification)

```bash
# Quick test (10-minute simulation, ~1 min to run)
python scripts/run_animation.py --duration 600 --sample-interval 10

# Full run with specific connectivity
python scripts/run_animation.py --connectivity 0.40 --seed 42

# Save as GIF
python scripts/run_animation.py --duration 600 --save outputs/animation.gif
```

### CLI experiment runner

```bash
python scripts/run_experiment.py --config configs/default.yaml --dry-run
```

---

## 9. Spatial Layout

```
 0                  700                                               3000
 ┌───────────────────┬────────────────────────────────────────────────┐
 │                   │                                                │ 1500
 │                   │                                                │
 │                   │               SIMULATION AREA                  │
 │   INCIDENT ZONE   │               (3000 × 1500m)                  │
 │   (700 × 600m)    │                                                │
 │   y: 450–1050     │               Mobile nodes traverse           │
 │   Tasks appear     │               the full area via     ┌──────┐  │
 │   here.            │               Random Waypoint.      │COORD │  │ ~y=750
 │   Responders       │                                     │ZONE  │  │
 │   start here.      │                                     │50×50 │  │
 │                   │                                     └──────┘  │
 ├───────────────────┘                                   (2900, 725)  │
 │                                                                    │
 │                                                                    │
 └────────────────────────────────────────────────────────────────────┘
 Radio range: 100m — nodes must be this close to form an edge
```

Both zones are vertically centred at ~y=750. Coordination nodes are ~2200m
horizontally from the incident zone. With a 100m radio range, messages must
relay through intermediate mobile nodes to reach responders.
