"""
Scenario Generation (Phase 3).

This module generates emergency response scenarios for simulation,
implementing task generation with Poisson arrivals and urgency-based
prioritisation.

The scenario generator produces:
- Tasks with Poisson-distributed arrival times
- Urgency levels (H/M/L) based on configured distribution
- Source/destination assignments for coordination messages
- Complete scenario timelines for simulation execution

"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np

from ercs.config.parameters import (
    NetworkParameters,
    ScenarioParameters,
    UrgencyLevel,
)


class TaskStatus(str, Enum):
    """Status of a task in the simulation."""

    PENDING = "pending"  # Awaiting assignment
    ASSIGNED = "assigned"  # Assigned to responder


@dataclass
class Task:
    """
    Represents an emergency response task requiring coordination.

    Tasks are generated at specific times during simulation and require
    assignment to mobile responders. Each task has an urgency level
    that may influence prioritisation.

    Attributes:
        task_id: Unique identifier for the task
        creation_time: Simulation time when task was created (seconds)
        source_node: Node ID where task originated
        target_location_x: X coordinate of task location (metres)
        target_location_y: Y coordinate of task location (metres)
        urgency: Task urgency level (H/M/L)
        status: Current task status
        assigned_to: Node ID of assigned responder (if assigned)
        assignment_time: Time when task was assigned (if assigned)
    """

    task_id: str
    creation_time: float
    source_node: str
    target_location_x: float
    target_location_y: float
    urgency: UrgencyLevel
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str | None = None
    assignment_time: float | None = None

    def assign(self, responder_id: str, current_time: float) -> None:
        """Assign this task to a responder."""
        self.assigned_to = responder_id
        self.assignment_time = current_time
        self.status = TaskStatus.ASSIGNED

    @property
    def response_time(self) -> float | None:
        """Calculate response time (assignment - creation) if assigned."""
        if self.assignment_time is not None:
            return self.assignment_time - self.creation_time
        return None

    def age(self, current_time: float) -> float:
        """Calculate task age in seconds."""
        return current_time - self.creation_time

    def is_pending(self) -> bool:
        """Check if task is still pending assignment."""
        return self.status == TaskStatus.PENDING


@dataclass
class Scenario:
    """
    Complete emergency response scenario for simulation.

    Contains all tasks to be executed during a simulation run,
    along with scenario metadata and configuration.

    Attributes:
        scenario_id: Unique identifier for this scenario
        tasks: List of tasks in chronological order
        duration_seconds: Total scenario duration
        parameters: Scenario generation parameters used
        random_seed: Random seed used for reproducibility
        connectivity_level: Network connectivity level for this scenario
    """

    scenario_id: str
    tasks: list[Task]
    duration_seconds: int
    parameters: ScenarioParameters
    random_seed: int | None = None
    connectivity_level: float = 1.0

    # Computed statistics
    _tasks_by_urgency: dict[UrgencyLevel, list[Task]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        """Compute task classifications after initialization."""
        self._tasks_by_urgency = {level: [] for level in UrgencyLevel}
        for task in self.tasks:
            self._tasks_by_urgency[task.urgency].append(task)

    @property
    def total_tasks(self) -> int:
        """Total number of tasks in scenario."""
        return len(self.tasks)

    @property
    def high_urgency_count(self) -> int:
        """Number of high-urgency tasks."""
        return len(self._tasks_by_urgency[UrgencyLevel.HIGH])

    @property
    def medium_urgency_count(self) -> int:
        """Number of medium-urgency tasks."""
        return len(self._tasks_by_urgency[UrgencyLevel.MEDIUM])

    @property
    def low_urgency_count(self) -> int:
        """Number of low-urgency tasks."""
        return len(self._tasks_by_urgency[UrgencyLevel.LOW])

    def get_tasks_by_urgency(self, urgency: UrgencyLevel) -> list[Task]:
        """Get all tasks of a specific urgency level."""
        return self._tasks_by_urgency.get(urgency, [])

    def get_tasks_in_window(self, start_time: float, end_time: float) -> list[Task]:
        """Get tasks created within a time window."""
        return [
            task for task in self.tasks if start_time <= task.creation_time < end_time
        ]

    def get_pending_tasks(self, current_time: float) -> list[Task]:
        """Get all pending tasks that have been created by current_time."""
        return [
            task
            for task in self.tasks
            if task.creation_time <= current_time and task.is_pending()
        ]

    def tasks_iterator(self) -> Iterator[Task]:
        """Iterate over tasks in chronological order."""
        return iter(sorted(self.tasks, key=lambda t: t.creation_time))

    def summary(self) -> dict:
        """Generate scenario summary statistics."""
        return {
            "scenario_id": self.scenario_id,
            "total_tasks": self.total_tasks,
            "duration_seconds": self.duration_seconds,
            "connectivity_level": self.connectivity_level,
            "urgency_distribution": {
                "high": self.high_urgency_count,
                "medium": self.medium_urgency_count,
                "low": self.low_urgency_count,
            },
            "tasks_per_minute": self.total_tasks / (self.duration_seconds / 60),
        }


class ScenarioGenerator:
    """
    Generates emergency response scenarios for simulation.

    Implements Poisson process task generation with urgency-based
    classification following Phase 3 specification.

    The generator produces realistic emergency scenarios with:
    - Tasks arriving according to Poisson process
    - Three-level urgency classification
    - Configurable duration and connectivity levels

    Attributes:
        scenario_params: Scenario generation parameters
        network_params: Network topology parameters
        random_seed: Base random seed for reproducibility
    """

    def __init__(
        self,
        scenario_params: ScenarioParameters | None = None,
        network_params: NetworkParameters | None = None,
        random_seed: int | None = None,
    ):
        """
        Initialize the scenario generator.

        Args:
            scenario_params: Scenario generation parameters (uses defaults if None)
            network_params: Network parameters for location generation
            random_seed: Base random seed for reproducibility
        """
        self.scenario_params = scenario_params or ScenarioParameters()
        self.network_params = network_params or NetworkParameters()
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        # Pre-compute urgency probabilities
        urgency = self.scenario_params.urgency_distribution
        self._urgency_probs = [urgency.high, urgency.medium, urgency.low]
        self._urgency_levels = [
            UrgencyLevel.HIGH,
            UrgencyLevel.MEDIUM,
            UrgencyLevel.LOW,
        ]

    def generate(
        self,
        scenario_id: str | None = None,
        connectivity_level: float = 1.0,
        coordination_nodes: list[str] | None = None,
    ) -> Scenario:
        """
        Generate a complete emergency response scenario.

        Args:
            scenario_id: Optional scenario identifier (auto-generated if None)
            connectivity_level: Network connectivity level (0-1)
            coordination_nodes: List of coordination node IDs for message destinations

        Returns:
            Complete Scenario object with generated tasks
        """
        if scenario_id is None:
            scenario_id = f"scenario_{self._rng.integers(100000, 999999)}"

        if coordination_nodes is None:
            coordination_nodes = [
                f"coord_{i}" for i in range(self.network_params.coordination_node_count)
            ]

        # Generate task arrival times using Poisson process
        arrival_times = self._generate_arrival_times()

        # Generate tasks
        tasks = []
        for i, arrival_time in enumerate(arrival_times):
            task = self._generate_task(
                task_index=i,
                creation_time=arrival_time,
                coordination_nodes=coordination_nodes,
            )
            tasks.append(task)

        return Scenario(
            scenario_id=scenario_id,
            tasks=tasks,
            duration_seconds=self.scenario_params.simulation_duration_seconds,
            parameters=self.scenario_params,
            random_seed=self.random_seed,
            connectivity_level=connectivity_level,
        )

    def _generate_arrival_times(self) -> list[float]:
        """
        Generate task arrival times using Poisson process.

        The Poisson process generates inter-arrival times from an
        exponential distribution with rate λ = messages per second.

        Returns:
            List of arrival times in seconds, sorted chronologically
        """
        duration = self.scenario_params.simulation_duration_seconds
        rate_per_minute = self.scenario_params.message_rate_per_minute
        rate_per_second = rate_per_minute / 60.0

        # Generate inter-arrival times from exponential distribution
        # Mean inter-arrival time = 1/λ
        mean_interval = 1.0 / rate_per_second

        arrival_times = []
        current_time = 0.0

        while current_time < duration:
            # Generate next inter-arrival time
            interval = self._rng.exponential(mean_interval)
            current_time += interval

            if current_time < duration:
                arrival_times.append(current_time)

        return arrival_times

    def _generate_task(
        self,
        task_index: int,
        creation_time: float,
        coordination_nodes: list[str],
    ) -> Task:
        """
        Generate a single task with random attributes.

        Args:
            task_index: Index of the task (for ID generation)
            creation_time: When the task is created
            coordination_nodes: Possible source nodes

        Returns:
            Generated Task object
        """
        # Generate task ID
        task_id = f"task_{task_index:04d}"

        # Select source node (coordination nodes generate tasks)
        source_node = self._rng.choice(coordination_nodes)

        # Generate target location within incident zone
        zone = self.network_params.incident_zone
        target_x = self._rng.uniform(zone.origin_x, zone.origin_x + zone.width_m)
        target_y = self._rng.uniform(zone.origin_y, zone.origin_y + zone.height_m)

        # Assign urgency level based on distribution
        urgency = self._select_urgency()

        return Task(
            task_id=task_id,
            creation_time=creation_time,
            source_node=source_node,
            target_location_x=target_x,
            target_location_y=target_y,
            urgency=urgency,
        )

    def _select_urgency(self) -> UrgencyLevel:
        """
        Select urgency level based on configured distribution.

        Distribution:
        - High: 20%
        - Medium: 50%
        - Low: 30%

        Returns:
            Selected UrgencyLevel
        """
        idx = self._rng.choice(
            len(self._urgency_levels),
            p=self._urgency_probs,
        )
        return self._urgency_levels[idx]

    def generate_batch(
        self,
        count: int,
        connectivity_levels: list[float] | None = None,
        base_seed: int | None = None,
    ) -> list[Scenario]:
        """
        Generate multiple scenarios for experimental runs.

        Args:
            count: Number of scenarios per connectivity level
            connectivity_levels: List of connectivity levels (uses defaults if None)
            base_seed: Base seed for reproducible generation

        Returns:
            List of generated scenarios
        """
        if connectivity_levels is None:
            connectivity_levels = self.network_params.connectivity_scenarios

        scenarios = []
        scenario_index = 0

        for conn_level in connectivity_levels:
            for run in range(count):
                # Use deterministic seed for reproducibility
                if base_seed is not None:
                    run_seed = base_seed + scenario_index
                    self._rng = np.random.default_rng(run_seed)
                else:
                    run_seed = None

                scenario_id = f"scenario_c{int(conn_level*100)}_r{run:02d}"
                scenario = self.generate(
                    scenario_id=scenario_id,
                    connectivity_level=conn_level,
                )
                scenarios.append(scenario)
                scenario_index += 1

        return scenarios


class ExperimentConfiguration:
    """
    Manages experimental configuration for multiple simulation runs.

    Implements the statistical design from Law (2015):
    - 30 runs per configuration for statistical significance
    - 2 algorithms × 3 connectivity levels = 6 configurations
    - 180 total experimental runs

    Attributes:
        scenario_params: Scenario generation parameters
        network_params: Network topology parameters
        connectivity_levels: Connectivity levels to test
        algorithms: Algorithm types to compare
        runs_per_config: Number of runs per configuration
    """

    def __init__(
        self,
        scenario_params: ScenarioParameters | None = None,
        network_params: NetworkParameters | None = None,
        connectivity_levels: list[float] | None = None,
        algorithms: list[Literal["adaptive", "baseline"]] | None = None,
        runs_per_config: int | None = None,
    ):
        """
        Initialize experiment configuration.

        Args:
            scenario_params: Scenario parameters (uses defaults if None)
            network_params: Network parameters (uses defaults if None)
            connectivity_levels: Connectivity levels (uses defaults if None)
            algorithms: Algorithms to compare (uses both if None)
            runs_per_config: Runs per configuration (uses default 30 if None)
        """
        self.scenario_params = scenario_params or ScenarioParameters()
        self.network_params = network_params or NetworkParameters()
        self.connectivity_levels = (
            connectivity_levels or self.network_params.connectivity_scenarios
        )
        self.algorithms = algorithms or ["adaptive", "baseline"]
        self.runs_per_config = (
            runs_per_config or self.scenario_params.runs_per_configuration
        )

    @property
    def total_configurations(self) -> int:
        """Total number of unique configurations."""
        return len(self.algorithms) * len(self.connectivity_levels)

    @property
    def total_runs(self) -> int:
        """Total number of experimental runs."""
        return self.total_configurations * self.runs_per_config

    def get_configuration_matrix(self) -> list[dict]:
        """
        Generate the full configuration matrix.

        Returns:
            List of configuration dictionaries
        """
        configurations = []
        config_id = 0

        for algorithm in self.algorithms:
            for connectivity in self.connectivity_levels:
                for run in range(self.runs_per_config):
                    configurations.append(
                        {
                            "config_id": config_id,
                            "algorithm": algorithm,
                            "connectivity_level": connectivity,
                            "run_number": run,
                            "random_seed": config_id * 1000 + run,
                        }
                    )
                config_id += 1

        return configurations

    def get_configurations_for_algorithm(self, algorithm: str) -> list[dict]:
        """Get all configurations for a specific algorithm."""
        return [
            cfg
            for cfg in self.get_configuration_matrix()
            if cfg["algorithm"] == algorithm
        ]

    def get_configurations_for_connectivity(
        self, connectivity_level: float
    ) -> list[dict]:
        """Get all configurations for a specific connectivity level."""
        return [
            cfg
            for cfg in self.get_configuration_matrix()
            if cfg["connectivity_level"] == connectivity_level
        ]

    def summary(self) -> dict:
        """Generate experiment summary."""
        return {
            "algorithms": self.algorithms,
            "connectivity_levels": self.connectivity_levels,
            "runs_per_configuration": self.runs_per_config,
            "total_configurations": self.total_configurations,
            "total_runs": self.total_runs,
            "simulation_duration_seconds": (
                self.scenario_params.simulation_duration_seconds
            ),
            "message_rate_per_minute": (self.scenario_params.message_rate_per_minute),
        }


def generate_scenario(
    scenario_params: ScenarioParameters | None = None,
    network_params: NetworkParameters | None = None,
    connectivity_level: float = 1.0,
    random_seed: int | None = None,
) -> Scenario:
    """
    Convenience function to generate a single scenario.

    Args:
        scenario_params: Scenario parameters (uses defaults if None)
        network_params: Network parameters (uses defaults if None)
        connectivity_level: Network connectivity level (0-1)
        random_seed: Random seed for reproducibility

    Returns:
        Generated Scenario

    Example:
        >>> from ercs.scenario import generate_scenario
        >>> scenario = generate_scenario(connectivity_level=0.75, random_seed=42)
        >>> print(f"Tasks: {scenario.total_tasks}")
    """
    generator = ScenarioGenerator(
        scenario_params=scenario_params,
        network_params=network_params,
        random_seed=random_seed,
    )
    return generator.generate(connectivity_level=connectivity_level)


def generate_experiment_scenarios(
    runs_per_connectivity: int = 30,
    connectivity_levels: list[float] | None = None,
    base_seed: int = 42,
) -> list[Scenario]:
    """
    Generate all scenarios for a complete experiment.

    Args:
        runs_per_connectivity: Number of runs per connectivity level
        connectivity_levels: Connectivity levels (uses defaults if None)
        base_seed: Base random seed for reproducibility

    Returns:
        List of scenarios for all experimental runs

    Example:
        >>> from ercs.scenario import generate_experiment_scenarios
        >>> scenarios = generate_experiment_scenarios(runs_per_connectivity=5)
        >>> print(f"Total scenarios: {len(scenarios)}")
    """
    generator = ScenarioGenerator(random_seed=base_seed)
    return generator.generate_batch(
        count=runs_per_connectivity,
        connectivity_levels=connectivity_levels,
        base_seed=base_seed,
    )
