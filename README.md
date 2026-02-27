# Emergency Response Coordination Simulator (ERCS)

A Python-based simulation framework for evaluating adaptive scheduling algorithms in emergency response coordination under intermittent connectivity conditions.

## Research Question

> Can adaptive scheduling algorithms integrated with delay-tolerant communication architectures improve emergency resource coordination effectiveness when operating under intermittent connectivity conditions?

## Installation

```bash
# Clone repository
git clone https://github.com/dianasnfonseca/resilient-emergency-response.git
cd resilient-emergency-response

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```bash
# Validate configuration (dry run)
python scripts/run_experiment.py --config configs/default.yaml --dry-run

# Run experiment
python scripts/run_experiment.py --config configs/default.yaml
```

## Project Structure

```
src/ercs/
├── config/          # Configuration parameters (Phases 1-4)
├── network/         # Network topology (Phase 1)
├── communication/   # DTN/PRoPHET (Phase 2)
├── scenario/        # Scenario generation (Phase 3)
├── coordination/    # Algorithms (Phase 4)
├── simulation/      # Integration (Phase 5)
└── evaluation/      # Metrics (Phase 6)
```

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Node count | 50 | Ullah & Qayyum (2022) |
| Radio range | 100m | Ullah & Qayyum (2022) |
| Connectivity | 75%, 40%, 20% | Karaman et al. (2026) |
| PRoPHET P_init | 0.75 | Kumar et al. (2023) |
| Duration | 6000s | Ullah & Qayyum (2022) |
| Runs/config | 30 | Law (2015) |

## Running Tests

```bash
pytest
pytest --cov=ercs
```

## License

MIT
