# Emergency Response Coordination Simulator (ERCS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A discrete-event simulation framework for evaluating adaptive scheduling algorithms in emergency response coordination under intermittent connectivity. Integrates PRoPHETv2 delay-tolerant networking (DTN) with multi-agent task assignment and role-based Random Waypoint mobility.

## Research Questions

**Main Research Question (MRQ):** Can adaptive scheduling algorithms integrated with delay-tolerant communication architectures improve emergency resource coordination effectiveness when operating under intermittent connectivity conditions?

**Sub-question 1 (SQ1):** How do distributed communication strategies combined with adaptive scheduling algorithms impact resource allocation effectiveness during varying levels of network disruption?

**Sub-question 2 (SQ2):** What are the optimal trade-offs between system complexity, resilience, and performance when adapting centralised emergency coordination approaches for decentralised, low-connectivity environments?

## Architecture

```
Phase 6  VISUALIZATION       plots, dashboard, notebook, animation, diagnostics
Phase 5  SIMULATION ENGINE   discrete-event queue, orchestration
Phase 4  COORDINATION        Adaptive (urgency-first, weighted scoring) vs Baseline (FCFS)
Phase 3  SCENARIOS           Poisson task arrivals, urgency distribution
Phase 2  COMMUNICATION       PRoPHETv2 store-and-forward, message buffers, predictability
Phase 1  NETWORK             two-zone topology, role-based Random Waypoint mobility, connectivity model
```

## Project Structure

```
src/ercs/
├── config/          # Pydantic parameter models and YAML validation
├── network/         # Topology generation (NetworkX) and role-based Random Waypoint mobility
├── communication/   # PRoPHETv2 DTN protocol, message buffers, predictability matrix
├── scenario/        # Emergency task generation (Poisson arrivals)
├── coordination/    # Adaptive and Baseline scheduling algorithms
├── simulation/      # Discrete-event engine and experiment runner
├── evaluation/      # Statistical analysis (t-tests, ANOVA, effect sizes)
└── visualization/   # Matplotlib plots, side-by-side animation, PRoPHET diagnostics

app/dashboard.py             # Streamlit interactive dashboard (6 tabs)
notebooks/experiment_report.ipynb  # Thesis-quality static analysis notebook
scripts/                     # CLI tools: experiment, animation, diagnostics
configs/default.yaml         # Default experiment configuration
```

## Installation

**Requirements:** Python 3.10+ | Tested on macOS (Apple Silicon)

```bash
git clone https://github.com/dianasnfonseca/resilient-emergency-response.git
cd resilient-emergency-response
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

For exact dependency versions used in development:
```bash
pip install -r requirements.txt
```

To install with visualization dependencies (Streamlit dashboard + Jupyter notebook):
```bash
pip install -e ".[viz]"
```

## Usage

```bash
# Validate configuration
python scripts/run_experiment.py --config configs/default.yaml --dry-run

# Quick test via Python
from ercs.simulation import run_simulation
results = run_simulation("adaptive", connectivity_level=0.75, random_seed=42)
print(f"Delivery rate: {results.delivery_rate:.2%}")
```

### Full Experiment (180 runs)

```python
from ercs.simulation import ExperimentRunner
from ercs.config.parameters import SimulationConfig

runner = ExperimentRunner(SimulationConfig(), base_seed=42)
results = runner.run_all(
    progress_callback=lambda current, total: print(f"{current}/{total}")
)
```

### Quick Start

```bash
# Interactive dashboard with research context and statistical analysis
streamlit run app/dashboard.py

# Static thesis-quality notebook with interpretive guidance
jupyter notebook notebooks/experiment_report.ipynb
```

### Streamlit Dashboard

Interactive web dashboard for running experiments and exploring results.

```bash
streamlit run app/dashboard.py
```

Features:

- Research question framing and metric-to-question mapping
- Sidebar with all experiment parameters and quick test mode (5 runs/config)
- Live progress bar with ETA during experiment execution
- Interactive visualizations: grouped bar charts, box plots, heatmaps, degradation lines
- Network diagnostics: PRoPHET predictability graphs, message journey tracking
- Statistical analysis tables (t-tests, ANOVA) with interpretive guidance
- Results interpretation guide tied to research questions

### Jupyter Notebook

Notebook with publication-quality figures and interpretive markdown for each visualisation.

```bash
jupyter notebook notebooks/experiment_report.ipynb
```

Run all cells top-to-bottom. To do a quick test, change `RUNS = runs_per_config` to `RUNS = 5` in the execution cell. All figures are saved to `outputs/figures/`.

### Animation and Diagnostics

```bash
# Side-by-side Adaptive vs Baseline animation
python scripts/run_animation.py --duration 3600 --sample-interval 30

# PRoPHET predictability network graph at a specific time
python scripts/run_animation.py --mode predictability --duration 3600 --time 1800

# Message journey tracking (spatial path + hop timeline)
python scripts/run_animation.py --mode journey --duration 3600

# All message paths overview
python scripts/run_animation.py --mode paths --connectivity 0.20

# Coordination cycle comparison (Adaptive vs Baseline decisions)
python scripts/diagnose_coordination.py

# Network encounter frequency analysis
python scripts/diagnose_encounters.py --connectivity 0.20
```

## Parameters

### Network Topology

| Parameter | Value | Source |
|-----------|-------|--------|
| Node count | 50 (2 coordination + 48 mobile) | Ullah & Qayyum (2022) |
| Simulation area | 3000 x 1500 m | Ullah & Qayyum (2022) |
| Incident zone | 700 x 600 m, origin (0, 450) | Ullah & Qayyum (2022) |
| Coordination zone | 50 x 50 m, origin (800, 300) | Design decision |
| Radio range | 100 m | Ullah & Qayyum (2022) |
| Buffer size | 25 MB (26,214,400 bytes) | Ullah & Qayyum (2022) |
| Message size | 500 kB (512,000 bytes) | Kumar et al. (2023) |
| Connectivity scenarios | 75%, 40%, 20% | Karaman et al. (2026) |
| Mobility model | Role-based Random Waypoint | Ullah & Qayyum (2022); Aschenbruck et al. (2009) |

### Role-Based Mobility

| Role | Proportion | Zone Constraint | Speed Range |
|------|-----------|-----------------|-------------|
| RESCUE | ~60% | Incident zone only | 1-5 m/s |
| TRANSPORT | ~25% | Shuttle incident <-> coordination | 5-20 m/s |
| LIAISON | ~15% | Full simulation area | 1-10 m/s |

### PRoPHETv2 Protocol

| Parameter | Value | Source |
|-----------|-------|--------|
| P_enc_max | 0.5 | Grasic et al. (2011) |
| I_typ (inter-encounter interval) | 1800 s | Grasic et al. (2011) |
| beta (transitivity) | 0.9 | Grasic et al. (2011) |
| gamma (aging) | 0.999885791 | Grasic et al. (2011) |
| Aging interval | 30 s | Kumar et al. (2023) |
| Message TTL | 300 min (18,000 s) | Ullah & Qayyum (2022) |
| Transmit speed | 2 Mbps | Ullah & Qayyum (2022) |
| Buffer drop policy | Drop oldest | Ullah & Qayyum (2022) |

### Scenario Generation

| Parameter | Value | Source |
|-----------|-------|--------|
| Task arrival | Poisson process | Pu et al. (2025) |
| Message rate | 2 msgs/min | Kumar et al. (2023) |
| Urgency distribution | 20% High, 50% Medium, 30% Low | Li et al. (2025) |
| Simulation duration | 6000 s (~100 min) | Ullah & Qayyum (2022) |
| Warm-up period | 0 s (cold-start, configurable) | Grassmann (2008) |

### Coordination

| Parameter | Value | Source |
|-----------|-------|--------|
| Coordination interval | 30 min (1800 s) | Kaji et al. (2025) |
| Priority levels | 3 | Rosas et al. (2023) |
| Path threshold (Adaptive) | P > 0.3 | Ullah & Qayyum (2022) |
| Scoring weights (Adaptive) | alpha=0.2 predict., gamma_r=0.2 recency, beta=0.6 proximity | Boondirek et al. (2014); Nelson et al. (2009) |
| Workload penalty (Adaptive) | lambda=0.2 | Cui et al. (2022) |

### Experimental Design

| Parameter | Value | Source |
|-----------|-------|--------|
| Runs per configuration | 30 | Law (2015) |
| Total configurations | 6 (2 algorithms x 3 connectivity) | -- |
| Total experimental runs | 180 | -- |

## Algorithms

**Adaptive Coordinator** (urgency-first, network-aware): Prioritises tasks by urgency (High > Medium > Low). Filters responders by PRoPHETv2 delivery predictability (P > 0.3), applies a hard capacity bound (k_max), then selects using a weighted score: `Score = 0.2 x P_abs + 0.2 x R_norm + 0.6 x D_norm - 0.2 x W_penalty`, balancing delivery predictability, encounter recency, and physical proximity while penalising already-assigned responders.

**Baseline Coordinator** (FCFS, proximity-only): FCFS task ordering, assigns to nearest responder by Euclidean distance regardless of network connectivity or link quality. No network state.

## Experiment

2 algorithms x 3 connectivity levels x 30 runs = 180 total simulation runs.

Methodology: PRoPHETv2 routing (Grasic et al., 2011), 30 runs per configuration (Law, 2015), P > 0.3 reachability threshold (Ullah & Qayyum, 2022), k_max capacity bound (Bhatti et al., 2021), connectivity levels 75%/40%/20% (Karaman et al., 2024), urgency distribution 20% High / 50% Medium / 30% Low (Li et al., 2025).

## Evaluation Metrics

| Metric | Research Question | What It Measures |
| ------ | ----------------- | ---------------- |
| avg_delivery_time | MRQ, SQ1 | Whether adaptive coordination improves coordination speed |
| delivery_rate | SQ2 | The reliability-over-coverage trade-off from the P > 0.3 filter |
| assignment_rate | Diagnostic | Experimental parity check -- expected identical across algorithms |
| avg_response_time | Diagnostic | Internal processing overhead -- expected identical across algorithms |

Statistical analysis includes Welch's independent t-tests, one-way ANOVA, Cohen's d effect sizes, and eta-squared with 95% confidence intervals.

## Testing

```bash
pytest                    # Run all 411 tests
pytest --cov=ercs         # With coverage
pytest -m "not slow"      # Skip slow tests
```

## Documentation

- [`docs/GUIDE.md`](docs/GUIDE.md) — Complete architecture guide, event cascade details, PRoPHETv2 equations, and visualisation reference.
- [`docs/Appendix_D_Technical_Reference_v1.md`](docs/Appendix_D_Technical_Reference_v1.md) — Formal technical reference appendix with all parameters, equations, algorithm pseudocode, and statistical framework.
- [`configs/default.yaml`](configs/default.yaml) — Single source of truth for all runtime parameters (mirrored in Pydantic defaults).

## Academic Context

**Resilient and Adaptive Scheduling Systems for Emergency Response in Low Connectivity Environments**

MSc Computer Science
University of Liverpool, 2026

## License

MIT
