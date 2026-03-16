# APPENDIX D: TECHNICAL REFERENCE

This appendix consolidates all final simulation parameters, protocol equations, algorithm specifications, and experimental design values used in the ERCS evaluation. All values correspond to the final configuration employed in the 180-run experiment (`configs/default.yaml`). Literature sources for each parameter are documented in the corresponding chapter sections.

## D.1 Network Topology Parameters

| Parameter | Value |
|-----------|-------|
| Node count | 50 (2 coordination + 48 mobile) |
| Simulation area | 3000 x 1500 m (4.5 km) |
| Incident zone | 700 x 600 m (origin at (0, 450)) |
| Coordination zone | 50 x 50 m (origin at (800, 300)) |
| Inter-zone separation | ~100-141 m (100 m edge-to-edge, ~141 m corner-to-corner) |
| Radio range | 100 m |
| Message size | 500 kB (512,000 bytes) |
| Buffer size | 25 MB (26,214,400 bytes) |
| Connectivity model | Link existence (symmetric, per-pair) |
| Connectivity scenarios | 75%, 40%, 20% link availability |
| Link determination | Deterministic CRC32 hash of (node_i, node_j, seed) |
| Mobility update interval | 1.0 s |
| Encounter check interval | 10.0 s |

*Inter-zone geometry: The incident zone is vertically centred within the simulation area, spanning x = [0, 700] and y = [450, 1050]. The coordination zone sits below and to the right, spanning x = [800, 850] and y = [300, 350]. This vertical offset places the coordination zone outside the incident zone's footprint, reflecting the operational practice of locating command posts away from the disaster area. Horizontally, the gap is 800 - 700 = 100 m edge-to-edge. The closest corner-to-corner path -- incident (700, 450) to coordination (800, 350) -- is sqrt(100^2 + 100^2) ~ 141 m. This separation exceeds the 100 m radio range, ensuring that no single-hop link can bridge the two zones and that mobile intermediaries (transport units and liaison agents) are required for multi-hop relay between coordination and incident operations.*

## D.2 Role-Based Mobility Model

The 48 mobile responders are assigned deterministically to three roles by node index. Role assignment is fixed at simulation initialisation and consistent across paired algorithm runs.

| Role | Count (%) | Zone constraint | Speed range | Pause range | Function |
|------|-----------|-----------------|-------------|-------------|----------|
| Rescue workers | ~29 (60%) | Incident zone only | 1-5 m/s | 10-60 s | Operate within disaster area |
| Transport units | ~12 (25%) | Shuttle: incident <-> coordination | 5-20 m/s | 30-120 s | Bridge disconnected zones (data mules) |
| Liaison agents | ~7 (15%) | Full simulation area | 1-10 m/s | 0-30 s | Cross-zone coordination |

## D.3 PRoPHETv2 Protocol Parameters and Equations

### D.3.1 Protocol Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Max encounter predictability | P_enc_max | 0.5 |
| Typical inter-encounter interval | I_typ | 1800 s |
| Transitivity constant | beta | 0.9 |
| Aging constant (per second) | gamma | 0.999885791 |
| Aging time unit | -- | 30 s |
| Transmit speed | -- | 2 Mbps |
| Message TTL | -- | 18,000 s (300 min) |
| Buffer drop policy | -- | Drop Oldest |
| Encounter trigger | -- | Connection-up only |
| Initialisation | P(a,b)_0 | 0 (cold-start) |
| Min predictability threshold | -- | 0.001 |

### D.3.2 PRoPHETv2 Equations

**Encounter update (time-scaled):**

P(a,b) = P(a,b)_old + (1 - P(a,b)_old) x P_enc

where:

P_enc = P_enc_max x min(delta_t / I_typ, 1.0)

*delta_t = time elapsed since last encounter with this specific node. P_enc is capped at P_enc_max = 0.5.*

**Aging (continuous-time):**

P(a,b) = P(a,b)_old x gamma^k

*where k = number of time units elapsed (1 time unit = 30 s). gamma = 0.999885791 per second ~ 0.98 per 30-second unit.*

**Transitivity (MAX-based, replacing original additive formula):**

P(a,c) = max( P(a,c)_old, P(a,b) x P(b,c) x beta )

*Replaces the original additive transitivity P(a,c) = P(a,c) + (1 - P(a,c)) x P(a,b) x P(b,c) x beta, which caused unbounded predictability accumulation.*

**Forwarding criterion (GRTR):**

Forward message for destination d to node b iff P(b,d) >= P(a,d) and P(b,d) > 0

*The >= operator allows forwarding when predictabilities are equal, increasing delivery opportunities. The P(b,d) > 0 guard prevents degenerate forwarding when neither node has encountered the destination.*

## D.4 Coordination Algorithm Specifications

### D.4.1 Shared Infrastructure

| Parameter | Value |
|-----------|-------|
| Coordination nodes | 2 (fixed, in coordination zone) |
| Coordination cycle interval | 1800 s (30 min); cycles at t = 0, 1800, 3600, 5400 |
| Task queue | Shared Poisson-generated queue (identical per seed) |
| Message delivery | PRoPHETv2 store-carry-forward |
| Multi-node P query | max(P) across all coordination nodes |

### D.4.2 Baseline Algorithm

| Property | Specification |
|----------|---------------|
| Task ordering | First-Come-First-Served (by creation time) |
| Responder selection | Nearest available by Euclidean distance |
| Network state usage | None |
| Eligibility filter | None (all responders eligible) |
| Workload management | None |

### D.4.3 Adaptive Algorithm

**Task ordering:** Urgency-first (High > Medium > Low), creation time as tiebreaker.

**Eligibility filter:** P(coord, responder) > 0.3

**Scoring function:**

**Score = alpha x P_abs + gamma_r x R_norm + beta x D_norm - lambda x W_inter**

| Component | Symbol | Definition | Weight |
|-----------|--------|------------|--------|
| Absolute predictability | P_abs | P(coord, responder) from PRoPHETv2 layer | alpha = 0.2 |
| Encounter recency | R_norm | 1 - min(delta_t / T_REF, 1.0); T_REF = 1800 s | gamma_r = 0.2 |
| Normalised proximity | D_norm | 1 - (distance / diagonal) | beta = 0.6 |
| Workload penalty | W_inter | 1.0 if assigned in previous cycle, else 0.0 | lambda = 0.2 |

*Proximity-dominant weighting (beta = 0.6). Combined network-awareness weight (alpha + gamma_r = 0.4). When P-values are homogeneous, the recency term introduces variance; when P-values vary, both network terms reinforce the preference for well-connected candidates.*

### D.4.4 Adaptive Algorithm -- Pseudocode

```
ALGORITHM: AdaptiveCoordination
INPUT: pending_tasks, responders, coord_nodes, prophet_layer
OUTPUT: assignments[]

sort pending_tasks by (urgency DESC, creation_time ASC)

for each task in pending_tasks:
    candidates <- []
    for each responder r:
        P_max <- max over all coord_nodes c: prophet_layer.get_P(c, r)
        if P_max > 0.3:
            candidates.append(r, P_max)

    if candidates is empty:
        skip task  // no reachable responder

    for each (r, P_max) in candidates:
        R_norm <- 1 - min(delta_t_last_encounter(r) / 1800, 1.0)
        D_norm <- 1 - (distance(r, task) / area_diagonal)
        W_inter <- 1.0 if r assigned in previous cycle, else 0.0
        score <- 0.2 x P_max + 0.2 x R_norm + 0.6 x D_norm - 0.2 x W_inter

    best <- candidate with highest score
    assign task to best
    send assignment via PRoPHETv2 from coord_node to best
```

## D.5 Experimental Design Summary

| Parameter | Value |
|-----------|-------|
| Design type | 2 x 3 fully crossed factorial |
| Factor 1: Algorithm | Adaptive, Baseline |
| Factor 2: Connectivity | 75%, 40%, 20% |
| Replications per cell | 30 pre-screened seeds |
| Total runs | 180 |
| Simulation duration | 6000 s (100 min) |
| Warm-up period | None (cold-start) |
| Coordination cycles | 4 (t = 0, 1800, 3600, 5400 s) |
| Paired comparison | Both algorithms on same seed |
| Seed exclusion | Seeds with structurally disconnected topologies removed |

### D.5.1 Task Generation

| Parameter | Value |
|-----------|-------|
| Arrival process | Poisson (lambda = 2 tasks/min) |
| Task location | Uniform random within incident zone |
| Urgency distribution | 20% High / 50% Medium / 30% Low |
| Expected tasks per run | ~200 (100 min x 2 tasks/min) |
| Unassigned tail | ~10% (tasks arriving after t = 5400 s) |

## D.6 Statistical Analysis Framework

| Analysis | Method |
|----------|--------|
| Between-algorithm comparison | Welch's t-test (n = 30 per level) |
| Effect size (between) | Cohen's d |
| Within-algorithm connectivity effect | One-way ANOVA (df = 2, 87) |
| Effect size (within) | Eta-squared |
| Robustness check | Exclude seed 6 at 20% (n = 29) |

### D.6.1 Effect Size Benchmarks (Cohen, 1988)

| Measure | Small | Medium | Large |
|---------|-------|--------|-------|
| Cohen's d | 0.2 | 0.5 | 0.8 |
| Eta-squared | 0.01 | 0.06 | 0.14 |

## D.7 Performance Metrics

| Metric | Definition | Role |
|--------|------------|------|
| Avg. delivery time | Mean elapsed time: task creation -> assignment message delivery | Primary |
| Delivery rate | Proportion of assigned tasks whose messages were delivered | Secondary |
| R(600s) | Proportion of tasks delivered within 600 s of assignment | Diagnostic |
| R(1800s) | Proportion of tasks delivered within 1800 s (one cycle) | Diagnostic |
| Assignment rate | Proportion of tasks assigned by any coordination cycle | Control |
| Decision time | Mean time from task creation to assignment decision | Control |

## D.8 Connectivity Scenario Mapping

| Scenario | Link availability | Empirical basis | Severity |
|----------|-------------------|-----------------|----------|
| Mild degradation | 75% | ~73% base stations operational post-earthquake | Immediate post-disaster |
| Moderate degradation | 40% | ~40% capacity restored within 48 hours | Sustained disruption phase |
| Severe degradation | 20% | Near-total failure in worst-affected provinces | Critical infrastructure collapse |

## D.9 Simulation Engine: Determinism and Reproducibility

All stochastic elements in the simulation are controlled through seeded random number generators:

- **Topology generation:** NetworkX graph construction uses the experiment seed
- **Mobility:** Random Waypoint movement uses per-node seeded RNG
- **Task generation:** Poisson arrivals use a seeded RNG
- **Link availability:** Deterministic CRC32 hash of (node_i, node_j, seed)

**Deterministic encounter processing order:** The simulation engine's `_handle_node_encounters()` method processes newly formed and existing network links. Because Python set iteration order depends on `PYTHONHASHSEED`, the implementation uses `sorted()` on link tuples to impose deterministic lexicographic ordering. This ensures that encounter updates, message transfers, and predictability calculations produce identical results across processes, regardless of the Python hash seed.

**Seed pre-screening:** All 200 candidate seeds (1-200) are validated by `scripts/validate_seeds.py` to ensure that at each connectivity level, at least one transport or liaison node can encounter a coordination node through an available link during a 300-second warm-up period. Seeds producing structurally disconnected topologies are excluded. The validated seed manifest is stored in `configs/valid_seeds.json`.
