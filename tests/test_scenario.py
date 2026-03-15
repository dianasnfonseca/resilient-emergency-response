"""
Tests for Scenario Generation (Phase 3).

These tests verify that the scenario generator correctly implements
the specifications from the Phase 3 documentation:
- Poisson process task arrivals with rate 2/minute
- Urgency distribution: 20% H, 50% M, 30% L
- Simulation duration: 6000 seconds
- 30 runs per configuration
"""

import numpy as np
import pytest

from ercs.config.parameters import (
    NetworkParameters,
    ScenarioParameters,
    UrgencyDistribution,
    UrgencyLevel,
)
from conftest import (
    ALGORITHMS,
    CONNECTIVITY_MILD,
    CONNECTIVITY_MODERATE,
    CONNECTIVITY_SCENARIOS,
    CONNECTIVITY_SEVERE,
    MESSAGE_RATE_PER_MIN,
    RUNS_PER_CONFIG,
    SIMULATION_DURATION_S,
    TOTAL_EXPERIMENTAL_RUNS,
    URGENCY_HIGH_PROP,
    URGENCY_LOW_PROP,
    URGENCY_MEDIUM_PROP,
    WARMUP_PERIOD_S,
)
from ercs.scenario import (
    ExperimentConfiguration,
    Scenario,
    ScenarioGenerator,
    Task,
    TaskStatus,
    generate_experiment_scenarios,
    generate_scenario,
)

# =============================================================================
# Test Task Class
# =============================================================================


class TestTask:
    """Tests for the Task class."""

    def test_task_creation(self):
        """Test basic task creation with all attributes."""
        task = Task(
            task_id="task_0001",
            creation_time=100.0,
            source_node="coord_0",
            target_location_x=350.0,
            target_location_y=750.0,
            urgency=UrgencyLevel.HIGH,
        )

        assert task.task_id == "task_0001"
        assert task.creation_time == 100.0
        assert task.source_node == "coord_0"
        assert task.target_location_x == 350.0
        assert task.target_location_y == 750.0
        assert task.urgency == UrgencyLevel.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.assigned_to is None

    def test_task_assignment(self):
        """Test task assignment to responder."""
        task = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=100.0,
            urgency=UrgencyLevel.MEDIUM,
        )

        task.assign("mobile_5", current_time=50.0)

        assert task.status == TaskStatus.ASSIGNED
        assert task.assigned_to == "mobile_5"
        assert task.assignment_time == 50.0

    def test_response_time_calculation(self):
        """Test response time calculation (assignment - creation)."""
        task = Task(
            task_id="task_0001",
            creation_time=100.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=100.0,
            urgency=UrgencyLevel.HIGH,
        )

        # Before assignment
        assert task.response_time is None

        # After assignment
        task.assign("mobile_1", current_time=150.0)
        assert task.response_time == 50.0  # 150 - 100

    def test_task_age(self):
        """Test task age calculation."""
        task = Task(
            task_id="task_0001",
            creation_time=100.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=100.0,
            urgency=UrgencyLevel.LOW,
        )

        assert task.age(100.0) == 0.0
        assert task.age(200.0) == 100.0
        assert task.age(500.0) == 400.0

    def test_is_pending(self):
        """Test pending status check."""
        task = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=100.0,
            urgency=UrgencyLevel.MEDIUM,
        )

        assert task.is_pending() is True

        task.assign("mobile_1", current_time=10.0)
        assert task.is_pending() is False


# =============================================================================
# Test Scenario Class
# =============================================================================


class TestScenario:
    """Tests for the Scenario class."""

    @pytest.fixture
    def sample_tasks(self) -> list[Task]:
        """Create sample tasks for testing."""
        tasks = []
        urgencies = [
            UrgencyLevel.HIGH,
            UrgencyLevel.MEDIUM,
            UrgencyLevel.MEDIUM,
            UrgencyLevel.LOW,
            UrgencyLevel.HIGH,
        ]

        for i, urgency in enumerate(urgencies):
            tasks.append(
                Task(
                    task_id=f"task_{i:04d}",
                    creation_time=float(i * 30),
                    source_node="coord_0",
                    target_location_x=100.0 + i * 50,
                    target_location_y=500.0,
                    urgency=urgency,
                )
            )
        return tasks

    @pytest.fixture
    def scenario(self, sample_tasks: list[Task]) -> Scenario:
        """Create a sample scenario."""
        return Scenario(
            scenario_id="test_scenario",
            tasks=sample_tasks,
            duration_seconds=SIMULATION_DURATION_S,
            parameters=ScenarioParameters(),
            random_seed=42,
            connectivity_level=CONNECTIVITY_MILD,
        )

    def test_scenario_creation(self, scenario: Scenario):
        """Test scenario creation."""
        assert scenario.scenario_id == "test_scenario"
        assert scenario.total_tasks == 5
        assert scenario.duration_seconds == SIMULATION_DURATION_S
        assert scenario.connectivity_level == CONNECTIVITY_MILD

    def test_urgency_counts(self, scenario: Scenario):
        """Test urgency level counts."""
        assert scenario.high_urgency_count == 2
        assert scenario.medium_urgency_count == 2
        assert scenario.low_urgency_count == 1

    def test_get_tasks_by_urgency(self, scenario: Scenario):
        """Test filtering tasks by urgency."""
        high_tasks = scenario.get_tasks_by_urgency(UrgencyLevel.HIGH)
        assert len(high_tasks) == 2
        for task in high_tasks:
            assert task.urgency == UrgencyLevel.HIGH

    def test_get_tasks_in_window(self, scenario: Scenario):
        """Test getting tasks within a time window."""
        # Tasks at times 0, 30, 60, 90, 120
        tasks = scenario.get_tasks_in_window(25.0, 95.0)
        # Should include tasks at 30, 60, 90
        assert len(tasks) == 3

    def test_get_pending_tasks(self, scenario: Scenario):
        """Test getting pending tasks."""
        # All tasks should be pending initially
        pending = scenario.get_pending_tasks(current_time=1000.0)
        assert len(pending) == 5

        # Assign one task
        scenario.tasks[0].assign("mobile_1", current_time=10.0)
        pending = scenario.get_pending_tasks(current_time=1000.0)
        assert len(pending) == 4

    def test_tasks_iterator(self, scenario: Scenario):
        """Test iterating over tasks in order."""
        times = [task.creation_time for task in scenario.tasks_iterator()]
        assert times == sorted(times)

    def test_scenario_summary(self, scenario: Scenario):
        """Test scenario summary generation."""
        summary = scenario.summary()

        assert summary["scenario_id"] == "test_scenario"
        assert summary["total_tasks"] == 5
        assert summary["duration_seconds"] == SIMULATION_DURATION_S
        assert summary["connectivity_level"] == CONNECTIVITY_MILD
        assert summary["urgency_distribution"]["high"] == 2
        assert summary["urgency_distribution"]["medium"] == 2
        assert summary["urgency_distribution"]["low"] == 1


# =============================================================================
# Test ScenarioGenerator
# =============================================================================


class TestScenarioGenerator:
    """Tests for the ScenarioGenerator class."""

    @pytest.fixture
    def generator(self) -> ScenarioGenerator:
        """Create a generator with default parameters and fixed seed."""
        return ScenarioGenerator(random_seed=42)

    def test_generator_creation(self, generator: ScenarioGenerator):
        """Test generator initialization."""
        assert generator.scenario_params is not None
        assert generator.network_params is not None
        assert generator.random_seed == 42

    def test_generate_scenario(self, generator: ScenarioGenerator):
        """Test basic scenario generation."""
        scenario = generator.generate(
            scenario_id="test_001",
            connectivity_level=CONNECTIVITY_MILD,
        )

        assert scenario.scenario_id == "test_001"
        assert scenario.connectivity_level == CONNECTIVITY_MILD
        assert scenario.total_tasks > 0
        assert scenario.duration_seconds == SIMULATION_DURATION_S

    def test_poisson_arrival_rate(self, generator: ScenarioGenerator):
        """Test that arrival rate approximately matches 2/minute.

        With rate = 2/min and duration = 6000s = 100 min,
        expected tasks ≈ 200 (with Poisson variation).
        """
        # Generate multiple scenarios to average
        total_tasks = 0
        num_scenarios = 10

        for i in range(num_scenarios):
            gen = ScenarioGenerator(random_seed=i * 100)
            scenario = gen.generate()
            total_tasks += scenario.total_tasks

        avg_tasks = total_tasks / num_scenarios
        expected_tasks = MESSAGE_RATE_PER_MIN * (SIMULATION_DURATION_S / 60)  # 2/min * 100 min = 200

        # Should be within 20% of expected
        assert abs(avg_tasks - expected_tasks) / expected_tasks < 0.2

    def test_arrival_times_are_sorted(self, generator: ScenarioGenerator):
        """Test that task arrival times are in chronological order."""
        scenario = generator.generate()

        times = [task.creation_time for task in scenario.tasks]
        assert times == sorted(times)

    def test_arrival_times_within_duration(self, generator: ScenarioGenerator):
        """Test that all tasks arrive before simulation ends."""
        scenario = generator.generate()

        for task in scenario.tasks:
            assert task.creation_time < scenario.duration_seconds

    def test_urgency_distribution(self, generator: ScenarioGenerator):
        """Test that urgency distribution approximately matches 20/50/30.

        This is a statistical test over multiple scenarios.
        """
        total_high = 0
        total_medium = 0
        total_low = 0
        total_tasks = 0

        num_scenarios = 20
        for i in range(num_scenarios):
            gen = ScenarioGenerator(random_seed=i * 100)
            scenario = gen.generate()

            total_high += scenario.high_urgency_count
            total_medium += scenario.medium_urgency_count
            total_low += scenario.low_urgency_count
            total_tasks += scenario.total_tasks

        # Calculate actual proportions
        high_prop = total_high / total_tasks
        medium_prop = total_medium / total_tasks
        low_prop = total_low / total_tasks

        # Should be within 5% of expected (20%, 50%, 30%)
        assert abs(high_prop - 0.20) < 0.05
        assert abs(medium_prop - 0.50) < 0.05
        assert abs(low_prop - 0.30) < 0.05

    def test_task_locations_in_incident_zone(self, generator: ScenarioGenerator):
        """Test that task locations are within incident zone."""
        scenario = generator.generate()
        zone = generator.network_params.incident_zone

        for task in scenario.tasks:
            assert (
                zone.origin_x <= task.target_location_x <= zone.origin_x + zone.width_m
            )
            assert (
                zone.origin_y <= task.target_location_y <= zone.origin_y + zone.height_m
            )

    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical scenarios."""
        gen1 = ScenarioGenerator(random_seed=12345)
        gen2 = ScenarioGenerator(random_seed=12345)

        scenario1 = gen1.generate(scenario_id="test")
        scenario2 = gen2.generate(scenario_id="test")

        assert scenario1.total_tasks == scenario2.total_tasks

        for t1, t2 in zip(scenario1.tasks, scenario2.tasks):
            assert t1.creation_time == t2.creation_time
            assert t1.target_location_x == t2.target_location_x
            assert t1.urgency == t2.urgency

    def test_different_seeds_produce_different_scenarios(self):
        """Test that different seeds produce different scenarios."""
        gen1 = ScenarioGenerator(random_seed=111)
        gen2 = ScenarioGenerator(random_seed=222)

        scenario1 = gen1.generate()
        scenario2 = gen2.generate()

        # At least some tasks should have different creation times
        differences = sum(
            1
            for t1, t2 in zip(scenario1.tasks, scenario2.tasks)
            if t1.creation_time != t2.creation_time
        )
        assert differences > 0

    def test_generate_batch(self, generator: ScenarioGenerator):
        """Test batch scenario generation."""
        scenarios = generator.generate_batch(
            count=5,
            connectivity_levels=[CONNECTIVITY_MILD, CONNECTIVITY_MODERATE],
            base_seed=42,
        )

        # Should have 5 * 2 = 10 scenarios
        assert len(scenarios) == 10

        # Check connectivity levels
        conn_75 = [s for s in scenarios if s.connectivity_level == CONNECTIVITY_MILD]
        conn_40 = [s for s in scenarios if s.connectivity_level == CONNECTIVITY_MODERATE]

        assert len(conn_75) == 5
        assert len(conn_40) == 5

    def test_coordination_nodes_as_sources(self, generator: ScenarioGenerator):
        """Test that tasks originate from coordination nodes."""
        scenario = generator.generate(coordination_nodes=["coord_0", "coord_1"])

        for task in scenario.tasks:
            assert task.source_node in ["coord_0", "coord_1"]


# =============================================================================
# Test ExperimentConfiguration
# =============================================================================


class TestExperimentConfiguration:
    """Tests for the ExperimentConfiguration class."""

    @pytest.fixture
    def config(self) -> ExperimentConfiguration:
        """Create default experiment configuration."""
        return ExperimentConfiguration()

    def test_default_configuration(self, config: ExperimentConfiguration):
        """Test default configuration matches spec."""
        assert config.algorithms == ALGORITHMS
        assert config.connectivity_levels == CONNECTIVITY_SCENARIOS
        assert config.runs_per_config == RUNS_PER_CONFIG

    def test_total_configurations(self, config: ExperimentConfiguration):
        """Test total configuration count (2 algorithms × 3 connectivity)."""
        assert config.total_configurations == 6

    def test_total_runs(self, config: ExperimentConfiguration):
        """Test total runs (6 configs × 30 runs = 180)."""
        assert config.total_runs == TOTAL_EXPERIMENTAL_RUNS

    def test_configuration_matrix(self, config: ExperimentConfiguration):
        """Test configuration matrix generation."""
        matrix = config.get_configuration_matrix()

        assert len(matrix) == TOTAL_EXPERIMENTAL_RUNS

        # Check all combinations exist
        algorithms_seen = set()
        connectivities_seen = set()

        for cfg in matrix:
            algorithms_seen.add(cfg["algorithm"])
            connectivities_seen.add(cfg["connectivity_level"])
            assert "random_seed" in cfg
            assert "run_number" in cfg

        assert algorithms_seen == set(ALGORITHMS)
        assert connectivities_seen == set(CONNECTIVITY_SCENARIOS)

    def test_filter_by_algorithm(self, config: ExperimentConfiguration):
        """Test filtering configurations by algorithm."""
        adaptive_configs = config.get_configurations_for_algorithm("adaptive")
        baseline_configs = config.get_configurations_for_algorithm("baseline")

        assert len(adaptive_configs) == len(CONNECTIVITY_SCENARIOS) * RUNS_PER_CONFIG
        assert len(baseline_configs) == len(CONNECTIVITY_SCENARIOS) * RUNS_PER_CONFIG

        for cfg in adaptive_configs:
            assert cfg["algorithm"] == "adaptive"

    def test_filter_by_connectivity(self, config: ExperimentConfiguration):
        """Test filtering configurations by connectivity."""
        configs_75 = config.get_configurations_for_connectivity(0.75)
        configs_40 = config.get_configurations_for_connectivity(0.40)
        configs_20 = config.get_configurations_for_connectivity(0.20)

        assert len(configs_75) == len(ALGORITHMS) * RUNS_PER_CONFIG
        assert len(configs_40) == len(ALGORITHMS) * RUNS_PER_CONFIG
        assert len(configs_20) == len(ALGORITHMS) * RUNS_PER_CONFIG

    def test_custom_configuration(self):
        """Test custom experiment configuration."""
        config = ExperimentConfiguration(
            algorithms=["adaptive"],
            connectivity_levels=[0.50],
            runs_per_config=10,
        )

        assert config.total_configurations == 1
        assert config.total_runs == 10

    def test_summary(self, config: ExperimentConfiguration):
        """Test configuration summary."""
        summary = config.summary()

        assert summary["algorithms"] == ALGORITHMS
        assert summary["connectivity_levels"] == CONNECTIVITY_SCENARIOS
        assert summary["runs_per_configuration"] == RUNS_PER_CONFIG
        assert summary["total_configurations"] == 6
        assert summary["total_runs"] == TOTAL_EXPERIMENTAL_RUNS
        assert summary["simulation_duration_seconds"] == SIMULATION_DURATION_S
        assert summary["message_rate_per_minute"] == MESSAGE_RATE_PER_MIN


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_scenario_function(self):
        """Test generate_scenario convenience function."""
        scenario = generate_scenario(
            connectivity_level=CONNECTIVITY_MILD,
            random_seed=42,
        )

        assert scenario.total_tasks > 0
        assert scenario.connectivity_level == CONNECTIVITY_MILD
        assert scenario.duration_seconds == SIMULATION_DURATION_S

    def test_generate_experiment_scenarios_function(self):
        """Test generate_experiment_scenarios convenience function."""
        scenarios = generate_experiment_scenarios(
            runs_per_connectivity=5,
            connectivity_levels=[CONNECTIVITY_MILD, CONNECTIVITY_MODERATE],
            base_seed=42,
        )

        assert len(scenarios) == 10  # 5 × 2 connectivity levels

        # Verify connectivity distribution
        conn_75 = [s for s in scenarios if s.connectivity_level == CONNECTIVITY_MILD]
        conn_40 = [s for s in scenarios if s.connectivity_level == CONNECTIVITY_MODERATE]

        assert len(conn_75) == 5
        assert len(conn_40) == 5


# =============================================================================
# Test Parameter Verification (Phase 3 Spec)
# =============================================================================


class TestPhase3Parameters:
    """Verify Phase 3 parameters match specification."""

    def test_message_rate(self):
        """Verify message rate = 2/minute."""
        params = ScenarioParameters()
        assert params.message_rate_per_minute == MESSAGE_RATE_PER_MIN

    def test_simulation_duration(self):
        """Verify duration = 6000 seconds."""
        params = ScenarioParameters()
        assert params.simulation_duration_seconds == SIMULATION_DURATION_S

    def test_warmup_period(self):
        """Verify warm-up period = 0s (cold-start)."""
        params = ScenarioParameters()
        assert params.warmup_period_seconds == WARMUP_PERIOD_S

    def test_runs_per_configuration(self):
        """Verify 30 runs per configuration."""
        params = ScenarioParameters()
        assert params.runs_per_configuration == RUNS_PER_CONFIG

    def test_urgency_distribution_high(self):
        """Verify high urgency = 20%."""
        dist = UrgencyDistribution()
        assert dist.high == URGENCY_HIGH_PROP

    def test_urgency_distribution_medium(self):
        """Verify medium urgency = 50%."""
        dist = UrgencyDistribution()
        assert dist.medium == URGENCY_MEDIUM_PROP

    def test_urgency_distribution_low(self):
        """Verify low urgency = 30%."""
        dist = UrgencyDistribution()
        assert dist.low == URGENCY_LOW_PROP

    def test_urgency_distribution_sums_to_one(self):
        """Verify urgency proportions sum to 1.0."""
        dist = UrgencyDistribution()
        total = dist.high + dist.medium + dist.low
        assert total == pytest.approx(1.0)

    def test_poisson_generation_model(self):
        """Verify Poisson process model."""
        params = ScenarioParameters()
        assert params.message_generation_model == "poisson"

    def test_generic_emergency_scenario(self):
        """Verify generic emergency framework."""
        params = ScenarioParameters()
        assert params.scenario_type == "generic_emergency"


# =============================================================================
# Statistical Tests for Poisson Process
# =============================================================================


class TestPoissonStatistics:
    """Statistical tests for Poisson arrival process."""

    def test_inter_arrival_times_exponential(self):
        """Test that inter-arrival times follow exponential distribution.

        For a Poisson process with rate λ, inter-arrival times should
        be exponentially distributed with mean 1/λ.
        """
        generator = ScenarioGenerator(random_seed=42)

        # Collect many inter-arrival times
        all_intervals = []
        for i in range(50):
            gen = ScenarioGenerator(random_seed=i)
            scenario = gen.generate()

            times = sorted([t.creation_time for t in scenario.tasks])
            intervals = [times[j + 1] - times[j] for j in range(len(times) - 1)]
            all_intervals.extend(intervals)

        # Expected mean = 60/rate seconds (for rate of 2/min)
        expected_mean = 60.0 / MESSAGE_RATE_PER_MIN
        actual_mean = np.mean(all_intervals)

        # Should be within 10% of expected
        assert abs(actual_mean - expected_mean) / expected_mean < 0.1

    def test_task_count_variance(self):
        """Test that task count variance approximately equals mean.

        For a Poisson distribution, variance equals mean.
        """
        task_counts = []

        for i in range(100):
            gen = ScenarioGenerator(random_seed=i)
            scenario = gen.generate()
            task_counts.append(scenario.total_tasks)

        mean_count = np.mean(task_counts)
        var_count = np.var(task_counts)

        # Variance should be close to mean (within 30% for sample)
        assert abs(var_count - mean_count) / mean_count < 0.3
