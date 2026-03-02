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
  3. _initialize_coordination_         → Seed P(coord, mobile) = 0.75 for all pairs
     predictability()                   (simulates pre-established backhaul)
  4. create_coordinator()             → AdaptiveCoordinator or BaselineCoordinator
     CoordinationManager()              Manages pending task queue
  5. ScenarioGenerator.generate()     → ~1440 tasks via Poisson process (2/min, 12h)
     manager.add_tasks()                All tasks queued for future coordination cycles
  6. _initialize_mobility()           → Random Waypoint states for 48 mobile nodes
```

### 3.2 Event Scheduling

All events are pre-scheduled into a priority queue sorted by timestamp:

| Event Type | Frequency | Count (12h run) |
|---|---|---|
| `MOBILITY_UPDATE` | Every 1 second | ~43,200 |
| `NODE_ENCOUNTER` | Every 10 seconds | ~4,320 |
| `TASK_CREATED` | Poisson (~2/min) | ~1,440 |
| `COORDINATION_CYCLE` | Every 30 minutes | 24 |

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

#### MOBILITY_UPDATE (every 1 second)

This is the main driver of the simulation dynamics:

```
MobilityManager.step()
  └─ Each mobile node moves toward its Random Waypoint destination
     │
     ▼
topology.update_node_position() for each moved node
topology.update_edges_from_positions()
  └─ Detects NEW connections (nodes just came within 100m of each other)
     │
     ▼
For each new connection:
  communication_layer.process_encounter(node_a, node_b)
    ├─ Age predictabilities (time decay: P *= 0.98^k)
    ├─ Update P(a,b) — encounter equation: P += (1-P) × 0.75
    ├─ Update P(a,c) — transitivity through b
    └─ Exchange messages:
        ├─ Direct delivery if destination node present → MESSAGE_DELIVERED
        └─ Forward copy if other node has higher P to destination
```

#### NODE_ENCOUNTER (every 10 seconds)

```
For each existing edge in topology:
  └─ Probabilistic check: random() < connectivity_level
     ├─ 0.75 → 75% of edges succeed (mild degradation)
     ├─ 0.40 → 40% succeed (moderate, post-earthquake)
     └─ 0.20 → 20% succeed (severe, early disaster)
     │
     ▼ (if encounter succeeds)
  communication_layer.process_encounter()
    └─ Same cascade: predictability update → message exchange → delivery

Also: expire_all_messages() removes messages past their 300-minute TTL
```

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
flow relatively easily. At 20%, many messages expire (TTL = 300 min) before they
can reach their destination — this is where Adaptive should outperform Baseline
because it avoids assigning tasks to responders it cannot reach (P > 0 filter).

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
| Coordination zone | 50 × 50 m | Fixed infrastructure, near (2900, 1400) |
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
| Drop policy | DROP_OLDEST | When buffer is full |

### Scenario (Phase 3)

| Parameter | Default | Notes |
|---|---|---|
| Task arrival rate | 2 per minute | Poisson process |
| Simulation duration | 43,200 seconds | 12 hours |
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
 │   INCIDENT ZONE   │                                                │
 │   (700 × 600m)    │               SIMULATION AREA                  │
 │   Tasks appear     │               (3000 × 1500m)                  │
 │   here.            │                                                │
 │   Responders       │               Mobile nodes traverse           │
 │   start here.      │               the full area via               │
 │                   │               Random Waypoint.                 │
 ├───────────────────┘                                                │
 │                                                                    │
 │                                                      ┌──────┐      │
 │                                                      │COORD │      │
 │                                                      │ZONE  │      │
 │                                                      │50×50 │      │
 │                                                      └──────┘      │
 │                                                    ~(2900, 1400)   │
 └────────────────────────────────────────────────────────────────────┘
 Radio range: 100m — nodes must be this close to form an edge
```

Coordination nodes are ~2200m from the incident zone. With a 100m radio range,
messages must relay through intermediate mobile nodes to reach responders.

---

## 10. Known Limitations

### 10.1 Task Duration is Never Used (Critical)

The `estimated_duration` field on `Task` is generated based on urgency level
(HIGH = 3 min, MEDIUM = 5 min, LOW = 7 min, with +/-30% variation) but is
**never consumed by the simulation engine**. There is:

- No `TASK_STARTED` or `TASK_COMPLETED` event type
- No responder "busy" state — responders can receive unlimited simultaneous assignments
- No task execution modelling — once assigned, a task's only remaining lifecycle is
  whether its assignment *message* gets delivered

**Impact**: Assignment rates may be artificially high because there is no capacity
constraint. Both algorithms see all 48 responders as available for every task.
The differentiation between Adaptive and Baseline may be weaker than it should be,
since resource contention (responders being occupied) is not modelled.

The `Task` class has `start()`, `complete()`, and `fail()` methods, and a
`completion_duration` property — they exist but are never called.

### 10.2 Coordination Nodes are Geographically Isolated

Coordination nodes are positioned at ~(2900, 1400), roughly 2200m from the
incident zone at (0–700, 450–1050). With a 100m radio range, no direct link
can ever exist between coordination and incident zones. All message delivery
depends on mobile nodes physically traversing the gap via Random Waypoint
mobility. This is by design (simulating degraded infrastructure) but means
delivery times include substantial physical transit time.

### 10.3 No Warm-up Period

`warmup_period_seconds` is set to 0 in the default configuration. This means
results include the initial transient phase when PRoPHET predictability values
are still building up from zero (or from the seeded 0.75 for coord→mobile pairs).
Early coordination cycles may behave differently from steady-state cycles.

### 10.4 All Tasks Assigned in First Eligible Cycle

Because responders are never "busy", all pending tasks at each 30-minute
coordination cycle are assigned immediately. The Adaptive algorithm may leave
some unassigned if P = 0 for all responders (no reachable path), but the
Baseline algorithm always assigns everything. Most response times cluster
around multiples of 1800 seconds (the cycle interval).

### 10.5 Encounter Probability Applies to All Edges Equally

The `connectivity_level` parameter (0.75, 0.40, 0.20) is applied uniformly
to all topology edges during `NODE_ENCOUNTER` events — `random() < connectivity_level`.
There is no spatial variation in connectivity (e.g., closer to infrastructure = better
signal). All links degrade equally.
