# Emergency Response Coordination Simulator (ERCS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-411%20passing-brightgreen.svg)]()

A discrete-event simulation framework for evaluating adaptive scheduling algorithms in emergency response coordination under intermittent connectivity. Integrates PRoPHETv2 delay-tolerant networking (DTN) with multi-agent task assignment and role-based Random Waypoint mobility.

**MSc Computer Science Dissertation** -- University of Liverpool, 2026

*Resilient and Adaptive Scheduling Systems for Emergency Response in Low Connectivity Environments*

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

app/dashboard.py                  # Streamlit interactive dashboard
notebooks/experiment_report.ipynb # Thesis-quality static analysis notebook
scripts/                          # CLI tools: experiment, animation, diagnostics
configs/default.yaml              # Single source of truth for all runtime parameters
configs/valid_seeds.json          # Pre-screened seeds for topological validity
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

For visualization tools (Streamlit dashboard + Jupyter notebook):
```bash
pip install -e ".[viz]"
```

## Usage

### Quick Start

```bash
# Validate configuration
python scripts/run_experiment.py --config configs/default.yaml --dry-run

# Interactive dashboard with live experiment execution
streamlit run app/dashboard.py

# Static thesis-quality notebook
jupyter notebook notebooks/experiment_report.ipynb
```

### Running Experiments

```python
from ercs.config.parameters import SimulationConfig
from ercs.simulation.engine import ExperimentRunner

# Quick test (30 runs)
runner = ExperimentRunner(SimulationConfig(), base_seed=42)
results = runner.run_all(
    runs_per_config=5,
    progress_callback=lambda current, total: print(f"{current}/{total}")
)

# Full experiment (180 runs)
results = runner.run_all()
```

### Streamlit Dashboard

Interactive web dashboard for running experiments and exploring results:

```bash
streamlit run app/dashboard.py
```

Features:
- Research question framing with project overview
- Complete parameter tables with literature rationale
- Network topology visualization
- Live experiment execution with progress bar and ETA
- Interactive visualizations: grouped bars, box plots, heatmaps, degradation lines
- Network diagnostics: PRoPHET predictability graphs, heatmaps, evolution, message journeys
- Statistical analysis tables (t-tests, ANOVA) with effect size interpretation
- Key findings tab with research question answers

### Animation and Diagnostics

```bash
# Side-by-side Adaptive vs Baseline animation
python scripts/run_animation.py --duration 3600 --sample-interval 30

# PRoPHET predictability network graph
python scripts/run_animation.py --mode predictability --duration 3600 --time 1800

# Message journey tracking
python scripts/run_animation.py --mode journey --duration 3600

# All message paths overview
python scripts/run_animation.py --mode paths --connectivity 0.20
```

## Parameters

All parameters are defined in `configs/default.yaml` and loaded via Pydantic models in `src/ercs/config/parameters.py`. See `TECHNICAL_REFERENCE.md` for complete parameter tables, equations, and algorithm pseudocode.

### Key Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Node count | 50 (2 coordination + 48 mobile) | Ullah & Qayyum (2022) |
| Simulation area | 3000 x 1500 m | Ullah & Qayyum (2022) |
| Radio range | 100 m | Ullah & Qayyum (2022) |
| Connectivity scenarios | 75%, 40%, 20% | Karaman et al. (2026) |
| Mobility model | Role-based Random Waypoint | Ullah & Qayyum (2022); Aschenbruck et al. (2009) |
| PRoPHETv2 P_enc_max | 0.5 | Grasic et al. (2011) |
| Task arrival | Poisson, 2 msgs/min | Kumar et al. (2023) |
| Simulation duration | 6000 s (~100 min) | Ullah & Qayyum (2022) |
| Runs per configuration | 30 | Law (2015) |
| Total experimental runs | 180 | -- |

### Algorithms

**Adaptive Coordinator** (urgency-first, network-aware): Prioritises tasks by urgency (High > Medium > Low). Filters responders by PRoPHETv2 delivery predictability (P > 0.3), applies a hard capacity bound (k_max), then selects using a weighted score: `Score = 0.2 x P_abs + 0.2 x R_norm + 0.6 x D_norm - 0.2 x W_penalty`, balancing delivery predictability, encounter recency, and physical proximity while penalising already-assigned responders.

**Baseline Coordinator** (FCFS, proximity-only): FCFS task ordering, assigns to nearest responder by Euclidean distance regardless of network connectivity or link quality. No network state.

### Evaluation Metrics

| Metric | Research Question | What It Measures |
| ------ | ----------------- | ---------------- |
| avg_delivery_time | MRQ, SQ1 | Whether adaptive coordination improves coordination speed |
| delivery_rate | SQ2 | The reliability-over-coverage trade-off from the P > 0.3 filter |
| assignment_rate | Diagnostic | Experimental parity check -- expected identical across algorithms |
| avg_decision_time | Diagnostic | Internal processing overhead -- expected identical across algorithms |

Statistical analysis includes Welch's independent t-tests, one-way ANOVA, Cohen's d effect sizes, and eta-squared with 95% confidence intervals.

## Reproducibility

All stochastic elements in the simulation are controlled through seeded random number generators:

- **Topology and mobility:** NetworkX graph construction and Random Waypoint movement use per-seed, per-node RNG instances
- **Task generation:** Poisson arrivals use a seeded RNG
- **Link availability:** Deterministic CRC32 hash of (node_i, node_j, seed) -- no randomness involved
- **Seed pre-screening:** All candidate seeds are validated for topological connectivity at every scenario level (`configs/valid_seeds.json`)
- **Encounter processing order:** The simulation engine processes node encounters using `sorted()` on link tuples, imposing deterministic lexicographic ordering. This ensures identical results across processes regardless of `PYTHONHASHSEED`, because Python set iteration is hash-seed-dependent

The dissertation results in Chapter 5 were generated in a single session and are internally consistent. The deterministic encounter ordering guarantees that future runs are also cross-process reproducible.

## Testing

```bash
pytest                    # Run all 411 tests
pytest --cov=ercs         # With coverage (>80% threshold)
pytest -m "not slow"      # Skip slow tests
```

## Documentation

- `TECHNICAL_REFERENCE.md` -- Complete parameter tables, PRoPHETv2 equations, algorithm pseudocode, experimental design, and statistical framework (maps to dissertation Appendix D)
- `configs/default.yaml` -- Single source of truth for all runtime parameters
- `scripts/README.md` -- Guide to CLI tools and diagnostic scripts

## License

MIT
