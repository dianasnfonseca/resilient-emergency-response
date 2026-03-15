"""
Scenario Generation (Phase 3).

This module provides scenario generation for emergency response simulation,
implementing Poisson-distributed task arrivals with urgency-based classification.

Classes:
    Task: Individual emergency response task
    Scenario: Complete scenario with all tasks
    ScenarioGenerator: Generates scenarios with configurable parameters
    ExperimentConfiguration: Manages experimental design

Enums:
    TaskStatus: Status of task execution

Factory Functions:
    generate_scenario: Create a single scenario
    generate_experiment_scenarios: Create scenarios for full experiment

"""

from ercs.scenario.generator import (
    ExperimentConfiguration,
    Scenario,
    ScenarioGenerator,
    Task,
    TaskStatus,
    generate_experiment_scenarios,
    generate_scenario,
)

__all__ = [
    "ExperimentConfiguration",
    "Scenario",
    "ScenarioGenerator",
    "Task",
    "TaskStatus",
    "generate_experiment_scenarios",
    "generate_scenario",
]