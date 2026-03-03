# Emergency Response Coordination Simulator (ERCS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A discrete-event simulation framework for evaluating adaptive scheduling algorithms in emergency response coordination under intermittent connectivity. Integrates PRoPHET-inspired delay-tolerant networking (DTN) with multi-agent task assignment and Random Waypoint mobility.

## Research Questions

**Main Question:** Can adaptive scheduling algorithms integrated with delay-tolerant communication architectures improve emergency resource coordination effectiveness when operating under intermittent connectivity conditions?

**Sub-Questions:**
1. How do distributed communication strategies combined with adaptive scheduling algorithms impact resource allocation effectiveness during varying levels of network disruption?
2. What are the optimal trade-offs between system complexity, resilience, and performance when adapting centralized emergency coordination approaches for decentralized, low-connectivity environments?

## Architecture

```
Phase 6  VISUALIZATION       plots, dashboard, notebook, animation, diagnostics
Phase 5  SIMULATION ENGINE   discrete-event queue, orchestration
Phase 4  COORDINATION        Adaptive (urgency-first, weighted scoring) vs Baseline (FCFS)
Phase 3  SCENARIOS           Poisson task arrivals, urgency distribution
Phase 2  COMMUNICATION       PRoPHET store-and-forward, message buffers, predictability
Phase 1  NETWORK             two-zone topology, Random Waypoint mobility, connectivity model
```

## Project Structure

```
src/ercs/
├── config/          # Pydantic parameter models and YAML validation
├── network/         # Topology generation (NetworkX) and Random Waypoint mobility
├── communication/   # PRoPHET DTN protocol, message buffers, predictability matrix
├── scenario/        # Emergency task generation (Poisson arrivals)
├── coordination/    # Adaptive and Baseline scheduling algorithms
├── simulation/      # Discrete-event engine and experiment runner
├── evaluation/      # Statistical analysis (t-tests, ANOVA, effect sizes)
└── visualization/   # Matplotlib plots, side-by-side animation, PRoPHET diagnostics

app/dashboard.py             # Streamlit interactive dashboard (6 tabs)
notebooks/experiment_report.ipynb  # Thesis-quality static analysis notebook
scripts/                     # CLI tools: experiment, animation (6 modes), diagnostics
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

### Streamlit Dashboard

Interactive web dashboard for running experiments and exploring results visually.

```bash
streamlit run app/dashboard.py
```

Features:

- Sidebar with all experiment parameters and quick test mode (5 runs/config)
- Live progress bar with ETA during experiment execution
- Interactive visualizations: grouped bar charts, box plots, heatmaps, degradation lines
- Network diagnostics: PRoPHET predictability graphs, message journey tracking
- Statistical analysis tables (t-tests, ANOVA) with significance highlighting
- Auto-generated key findings summary

### Jupyter Notebook

Static notebook suitable for thesis appendix, with publication-quality figures.

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
| Simulation area | 3000 × 1500 m | Ullah & Qayyum (2022) |
| Incident zone | 700 × 600 m, origin (0, 450) | Ullah & Qayyum (2022) |
| Coordination zone | 50 × 50 m, origin (800, 300) | Design decision |
| Radio range | 100 m | Ullah & Qayyum (2022) |
| Buffer size | 25 MB (26,214,400 bytes) | Ullah & Qayyum (2022) |
| Message size | 500 kB (512,000 bytes) | Kumar et al. (2023) |
| Connectivity scenarios | 75%, 40%, 20% | Karaman et al. (2026) |
| Mobility model | Random Waypoint | Ullah & Qayyum (2022) |
| Speed range | 0–20 m/s (min 1 m/s enforced) | Ullah & Qayyum (2022) |

### PRoPHET Protocol

| Parameter | Value | Source |
|-----------|-------|--------|
| P_init | 0.75 | Kumar et al. (2023) |
| β (transitivity) | 0.25 | Kumar et al. (2023) |
| γ (aging) | 0.98 | Kumar et al. (2023) |
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

### Coordination

| Parameter | Value | Source |
|-----------|-------|--------|
| Coordination interval | 30 min (1800 s) | Kaji et al. (2025) |
| Priority levels | 3 | Rosas et al. (2023) |
| Path threshold (Adaptive) | P > 0 | Ullah & Qayyum (2022) |
| Scoring weights (Adaptive) | α=0.5 predictability, β=0.5 proximity | Kumar et al. (2023) |

### Experimental Design

| Parameter | Value | Source |
|-----------|-------|--------|
| Runs per configuration | 30 | Law (2015) |
| Total configurations | 6 (2 algorithms × 3 connectivity) | — |
| Total experimental runs | 180 | — |

## Algorithms

**Adaptive Coordinator**: Prioritises tasks by urgency (High > Medium > Low). Filters responders by communication reachability (P > 0), then selects using a weighted score: `Score = 0.5 × P_norm + 0.5 × D_norm`, balancing delivery predictability with physical proximity.

**Baseline Coordinator**: FCFS task ordering, assigns to nearest responder by Euclidean distance regardless of network connectivity.

## Evaluation Metrics

- **Delivery Rate**: Messages successfully delivered / messages created
- **Assignment Rate**: Tasks assigned / total tasks
- **Response Time**: Time from task creation to assignment
- **Delivery Time**: Time from task creation to message delivery

Statistical analysis includes Welch's independent t-tests, one-way ANOVA, Cohen's d effect sizes, and eta-squared (η²) with 95% confidence intervals.

## Testing

```bash
pytest                    # Run all 281 tests
pytest --cov=ercs         # With coverage
pytest -m "not slow"      # Skip slow tests
```

## Documentation

See [`docs/GUIDE.md`](docs/GUIDE.md) for the complete architecture guide, event cascade details, PRoPHET equations, and visualization reference.

## Academic Context

**Resilient and Adaptive Scheduling Systems for Emergency Response in Low Connectivity Environments**

MSc Computer Science
University of Liverpool, 2026

## License

MIT
