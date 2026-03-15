"""
Simulation Engine (Phase 5).

This module provides the discrete event simulation engine that integrates
all components from Phases 1-4 into a complete experimental framework.

Classes:
    SimulationEngine: Core discrete event simulation
    ExperimentRunner: Manages multi-run experiments
    SimulationResults: Results container with metrics
    SimulationEvent: Individual simulation event
    TopologyAdapter: Bridges topology/communication with coordination

Enums:
    SimulationEventType: Types of simulation events

Factory Functions:
    run_simulation: Run a single simulation
    run_experiment: Run complete experiment (180 runs)

"""

from ercs.simulation.engine import (
    ExperimentRunner,
    SimulationEngine,
    SimulationEvent,
    SimulationEventType,
    SimulationResults,
    TopologyAdapter,
    run_experiment,
    run_simulation,
)

__all__ = [
    "ExperimentRunner",
    "SimulationEngine",
    "SimulationEvent",
    "SimulationEventType",
    "SimulationResults",
    "TopologyAdapter",
    "run_experiment",
    "run_simulation",
]