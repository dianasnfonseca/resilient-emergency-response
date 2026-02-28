# Emergency Response Coordination Simulator (ERCS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A discrete-event simulation framework for evaluating adaptive scheduling algorithms in emergency response coordination under intermittent connectivity. Integrates PRoPHET-inspired delay-tolerant networking (DTN) with multi-agent task assignment.

## Research Questions

**Main Question:** Can adaptive scheduling algorithms integrated with delay-tolerant communication architectures improve emergency resource coordination effectiveness when operating under intermittent connectivity conditions?

**Sub-Questions:**
1. How do distributed communication strategies combined with adaptive scheduling algorithms impact resource allocation effectiveness during varying levels of network disruption?
2. What are the optimal trade-offs between system complexity, resilience, and performance when adapting centralized emergency coordination approaches for decentralized, low-connectivity environments?

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Coordination Layer                        │
│         (Adaptive & Baseline Task Assignment)                │
├─────────────────────────────────────────────────────────────┤
│                   Communication Layer                        │
│            (PRoPHET Store-and-Forward Protocol)              │
├─────────────────────────────────────────────────────────────┤
│                      Network Layer                           │
│           (Two-Zone Topology, Connectivity Model)            │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/ercs/
├── config/          # Parameter schemas and validation
├── network/         # Topology generation (NetworkX)
├── communication/   # PRoPHET DTN protocol
├── scenario/        # Emergency task generation
├── coordination/    # Scheduling algorithms
├── simulation/      # Discrete-event engine
└── evaluation/      # Statistical analysis
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

## Usage

```bash
# Validate configuration
python scripts/run_experiment.py --config configs/default.yaml --dry-run

# Run full experiment (180 runs)
python scripts/run_experiment.py --config configs/default.yaml

# Quick test via Python
from ercs.simulation import run_simulation
results = run_simulation("adaptive", connectivity_level=0.75, random_seed=42)
print(f"Delivery rate: {results.delivery_rate:.2%}")
```

## Parameters

### Network Topology

| Parameter | Value | Source |
|-----------|-------|--------|
| Node count | 50 (2 coordination + 48 mobile) | Ullah & Qayyum (2022) |
| Simulation area | 3000 × 1500 m² | Ullah & Qayyum (2022) |
| Incident zone | 700 × 600 m² | Ullah & Qayyum (2022) |
| Radio range | 100 m | Ullah & Qayyum (2022) |
| Buffer size | 5 MB | Ullah & Qayyum (2022) |
| Message size | 500 kB | Kumar et al. (2023) |
| Connectivity scenarios | 75%, 40%, 20% | Karaman et al. (2026) |
| Mobility model | Random Waypoint | Ullah & Qayyum (2022) |
| Speed range | 0–20 m/s | Ullah & Qayyum (2022) |

### PRoPHET Protocol

| Parameter | Value | Source |
|-----------|-------|--------|
| P_init | 0.75 | Kumar et al. (2023) |
| β (transitivity) | 0.25 | Kumar et al. (2023) |
| γ (aging) | 0.98 | Kumar et al. (2023) |
| Message TTL | 300 min | Ullah & Qayyum (2022) |
| Transmit speed | 2 Mbps | Ullah & Qayyum (2022) |
| Buffer drop policy | Drop oldest | Ullah & Qayyum (2022) |

### Scenario Generation

| Parameter | Value | Source |
|-----------|-------|--------|
| Task arrival | Poisson process | Pu et al. (2025) |
| Message rate | 2 msgs/min | Kumar et al. (2023) |
| Urgency distribution | 20% High, 50% Medium, 30% Low | Li et al. (2025) |
| Coordination interval | 30 min | Kaji et al. (2025) |

### Experimental Design

| Parameter | Value | Source |
|-----------|-------|--------|
| Simulation duration | 6000 s | Ullah & Qayyum (2022) |
| Runs per configuration | 30 | Law (2015) |
| Total configurations | 6 (2 algorithms × 3 connectivity) | — |
| Total experimental runs | 180 | — |

## Algorithms

**Adaptive Coordinator**: Prioritises tasks by urgency (High → Medium → Low), assigns only to responders with available communication paths (P > 0).

**Baseline Coordinator**: FCFS task ordering, assigns to nearest responder regardless of network connectivity.

## Evaluation Metrics

- **Delivery Rate**: Messages successfully delivered / messages created
- **Assignment Rate**: Tasks assigned / total tasks
- **Response Time**: Time from task creation to assignment

Statistical analysis includes independent t-tests, one-way ANOVA, and Cohen's d effect sizes.

## Testing

```bash
pytest                    # Run all tests
pytest --cov=ercs         # With coverage
pytest -m "not slow"      # Skip slow tests
```

## Academic Context

**Resilient and Adaptive Scheduling Systems for Emergency Response in Low Connectivity Environments**

MSc Computer Science  
University of Liverpool, 2026

## License

MIT