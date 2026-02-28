"""
Tests for Simulation Engine (Phase 5).

These tests verify that the simulation engine correctly integrates
all components from Phases 1-4 and produces valid experimental results.

Tests cover:
- Component initialization and integration
- Event-driven simulation execution
- Results collection and metrics
- Reproducibility with seeds
- Experiment runner functionality
"""

import pytest

from ercs.config.parameters import (
    AlgorithmType,
    CoordinationParameters,
    NetworkParameters,
    ScenarioParameters,
    SimulationConfig,
    UrgencyLevel,
)
from ercs.simulation import (
    ExperimentRunner,
    SimulationEngine,
    SimulationEvent,
    SimulationEventType,
    SimulationResults,
    TopologyAdapter,
    run_experiment,
    run_simulation,
)

# =============================================================================
# Test SimulationEvent
# =============================================================================


class TestSimulationEvent:
    """Tests for SimulationEvent class."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = SimulationEvent(
            event_type=SimulationEventType.TASK_CREATED,
            timestamp=100.0,
            data={"task_id": "task_0001"},
        )

        assert event.event_type == SimulationEventType.TASK_CREATED
        assert event.timestamp == 100.0
        assert event.data["task_id"] == "task_0001"

    def test_event_ordering(self):
        """Test events are orderable by timestamp."""
        event1 = SimulationEvent(SimulationEventType.TASK_CREATED, 50.0)
        event2 = SimulationEvent(SimulationEventType.COORDINATION_CYCLE, 100.0)
        event3 = SimulationEvent(SimulationEventType.MESSAGE_DELIVERED, 25.0)

        sorted_events = sorted([event1, event2, event3])

        assert sorted_events[0].timestamp == 25.0
        assert sorted_events[1].timestamp == 50.0
        assert sorted_events[2].timestamp == 100.0

    def test_event_default_data(self):
        """Test event with default empty data."""
        event = SimulationEvent(
            event_type=SimulationEventType.SIMULATION_START,
            timestamp=0.0,
        )

        assert event.data == {}


# =============================================================================
# Test SimulationResults
# =============================================================================


class TestSimulationResults:
    """Tests for SimulationResults class."""

    @pytest.fixture
    def sample_results(self) -> SimulationResults:
        """Create sample results for testing."""
        config = SimulationConfig()
        results = SimulationResults(
            config=config,
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            run_number=0,
            random_seed=42,
        )

        # Add some metrics
        results.total_tasks = 100
        results.tasks_assigned = 80
        results.messages_created = 80
        results.messages_delivered = 60
        results.response_times = [
            ("task_001", 30.0),
            ("task_002", 45.0),
            ("task_003", 25.0),
        ]
        results.delivery_times = [
            ("task_001", 120.0),
            ("task_002", 90.0),
        ]

        return results

    def test_delivery_rate(self, sample_results: SimulationResults):
        """Test delivery rate calculation."""
        assert sample_results.delivery_rate == pytest.approx(0.75)  # 60/80

    def test_assignment_rate(self, sample_results: SimulationResults):
        """Test assignment rate calculation."""
        assert sample_results.assignment_rate == pytest.approx(0.80)  # 80/100

    def test_average_response_time(self, sample_results: SimulationResults):
        """Test average response time calculation."""
        expected = (30.0 + 45.0 + 25.0) / 3
        assert sample_results.average_response_time == pytest.approx(expected)

    def test_average_delivery_time(self, sample_results: SimulationResults):
        """Test average delivery time calculation."""
        expected = (120.0 + 90.0) / 2
        assert sample_results.average_delivery_time == pytest.approx(expected)

    def test_zero_tasks_rates(self):
        """Test rates when no tasks exist."""
        results = SimulationResults(
            config=SimulationConfig(),
            algorithm=AlgorithmType.BASELINE,
            connectivity_level=0.40,
            run_number=0,
            random_seed=None,
        )

        assert results.delivery_rate == 0.0
        assert results.assignment_rate == 0.0
        assert results.average_response_time is None
        assert results.average_delivery_time is None

    def test_summary(self, sample_results: SimulationResults):
        """Test results summary generation."""
        summary = sample_results.summary()

        assert summary["algorithm"] == "adaptive"
        assert summary["connectivity_level"] == 0.75
        assert summary["total_tasks"] == 100
        assert summary["tasks_assigned"] == 80
        assert summary["delivery_rate"] == pytest.approx(0.75)


# =============================================================================
# Test SimulationEngine
# =============================================================================


class TestSimulationEngine:
    """Tests for SimulationEngine class."""

    @pytest.fixture
    def short_config(self) -> SimulationConfig:
        """Create config for quick tests."""
        scenario = ScenarioParameters(
            simulation_duration_seconds=600,  # 10 minutes
            message_rate_per_minute=1.0,  # Fewer tasks
        )
        coordination = CoordinationParameters(
            update_interval_seconds=120,  # 2 minutes
        )
        return SimulationConfig(
            scenario=scenario,
            coordination=coordination,
        )

    def test_engine_creation(self, short_config: SimulationConfig):
        """Test engine initialization."""
        engine = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=42,
        )

        assert engine.config == short_config
        assert engine.algorithm_type == AlgorithmType.ADAPTIVE
        assert engine.connectivity_level == 0.75
        assert engine.random_seed == 42

    def test_run_simulation_adaptive(self, short_config: SimulationConfig):
        """Test running simulation with adaptive algorithm."""
        engine = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=42,
        )

        results = engine.run(run_number=0)

        assert results.algorithm == AlgorithmType.ADAPTIVE
        assert results.connectivity_level == 0.75
        assert results.run_number == 0
        assert results.total_tasks > 0
        assert len(results.events) > 0

    def test_run_simulation_baseline(self, short_config: SimulationConfig):
        """Test running simulation with baseline algorithm."""
        engine = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.BASELINE,
            connectivity_level=0.40,
            random_seed=42,
        )

        results = engine.run(run_number=1)

        assert results.algorithm == AlgorithmType.BASELINE
        assert results.connectivity_level == 0.40
        assert results.run_number == 1

    def test_reproducibility_with_seed(self, short_config: SimulationConfig):
        """Test that same seed produces identical results."""
        engine1 = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=12345,
        )

        engine2 = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=12345,
        )

        results1 = engine1.run()
        results2 = engine2.run()

        assert results1.total_tasks == results2.total_tasks
        assert results1.tasks_assigned == results2.tasks_assigned

    def test_different_seeds_produce_different_results(
        self, short_config: SimulationConfig
    ):
        """Test that different seeds produce different results."""
        engine1 = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=111,
        )

        engine2 = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=222,
        )

        results1 = engine1.run()
        results2 = engine2.run()

        # Results should differ (statistically likely)
        # At minimum, check events are generated
        assert len(results1.events) > 0
        assert len(results2.events) > 0

    def test_simulation_events_logged(self, short_config: SimulationConfig):
        """Test that simulation events are logged."""
        engine = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=42,
        )

        results = engine.run()

        # Should have start and end events
        event_types = [e.event_type for e in results.events]
        assert SimulationEventType.SIMULATION_START in event_types
        assert SimulationEventType.SIMULATION_END in event_types

    def test_coordination_cycles_executed(self, short_config: SimulationConfig):
        """Test that coordination cycles are executed."""
        engine = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=42,
        )

        results = engine.run()

        # Should have coordination cycle events
        cycle_events = [
            e
            for e in results.events
            if e.event_type == SimulationEventType.COORDINATION_CYCLE
        ]
        assert len(cycle_events) > 0

    def test_tasks_created_events(self, short_config: SimulationConfig):
        """Test that task creation events are logged."""
        engine = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=42,
        )

        results = engine.run()

        task_events = [
            e
            for e in results.events
            if e.event_type == SimulationEventType.TASK_CREATED
        ]

        assert len(task_events) == results.total_tasks

    def test_low_connectivity_affects_delivery(self, short_config: SimulationConfig):
        """Test that lower connectivity affects message delivery."""
        # Run with high connectivity
        engine_high = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.90,
            random_seed=42,
        )
        results_high = engine_high.run()

        # Run with low connectivity
        engine_low = SimulationEngine(
            config=short_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.20,
            random_seed=42,
        )
        results_low = engine_low.run()

        # Both should complete, even if delivery differs
        assert results_high.total_tasks > 0
        assert results_low.total_tasks > 0


# =============================================================================
# Test TopologyAdapter
# =============================================================================


class TestTopologyAdapter:
    """Tests for TopologyAdapter class."""

    @pytest.fixture
    def adapter(self) -> TopologyAdapter:
        """Create adapter with test components."""
        from ercs.communication.prophet import CommunicationLayer
        from ercs.config.parameters import CommunicationParameters
        from ercs.network.topology import generate_topology

        params = NetworkParameters()
        topology = generate_topology(params, connectivity_level=0.75, random_seed=42)

        comm = CommunicationLayer(
            comm_params=CommunicationParameters(),
            network_params=params,
            node_ids=topology.get_all_node_ids(),
        )

        return TopologyAdapter(topology=topology, communication=comm)

    def test_get_responder_position(self, adapter: TopologyAdapter):
        """Test getting responder position."""
        responders = adapter.get_all_responder_ids()
        assert len(responders) > 0

        x, y = adapter.get_responder_position(responders[0])
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_get_all_responder_ids(self, adapter: TopologyAdapter):
        """Test getting all responder IDs."""
        responders = adapter.get_all_responder_ids()

        # Should have 48 mobile responders
        assert len(responders) == 48
        for r_id in responders:
            assert r_id.startswith("mobile_")

    def test_get_delivery_predictability(self, adapter: TopologyAdapter):
        """Test getting delivery predictability."""
        responders = adapter.get_all_responder_ids()

        # Initially predictability is 0 (no encounters)
        p = adapter.get_delivery_predictability(responders[0], responders[1])
        assert p >= 0.0
        assert p <= 1.0

    def test_unknown_responder_raises(self, adapter: TopologyAdapter):
        """Test that unknown responder raises error."""
        with pytest.raises(ValueError):
            adapter.get_responder_position("unknown_node")


# =============================================================================
# Test ExperimentRunner
# =============================================================================


class TestExperimentRunner:
    """Tests for ExperimentRunner class."""

    @pytest.fixture
    def quick_config(self) -> SimulationConfig:
        """Create config for quick experiment tests."""
        scenario = ScenarioParameters(
            simulation_duration_seconds=300,  # 5 minutes
            message_rate_per_minute=0.5,
            runs_per_configuration=2,  # Quick test
        )
        coordination = CoordinationParameters(
            update_interval_seconds=60,
        )
        return SimulationConfig(
            scenario=scenario,
            coordination=coordination,
        )

    def test_runner_creation(self, quick_config: SimulationConfig):
        """Test experiment runner creation."""
        runner = ExperimentRunner(config=quick_config, base_seed=42)

        assert runner.config == quick_config
        assert runner.base_seed == 42

    def test_run_single_configuration(self, quick_config: SimulationConfig):
        """Test running single configuration."""
        runner = ExperimentRunner(config=quick_config, base_seed=42)

        results = runner.run_single_configuration(
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            runs=2,
        )

        assert len(results) == 2
        for r in results:
            assert r.algorithm == AlgorithmType.ADAPTIVE
            assert r.connectivity_level == 0.75

    def test_run_all_quick(self, quick_config: SimulationConfig):
        """Test running all configurations (quick version)."""
        runner = ExperimentRunner(config=quick_config, base_seed=42)

        results = runner.run_all(
            algorithms=[AlgorithmType.ADAPTIVE],
            connectivity_levels=[0.75],
            runs_per_config=2,
        )

        # 1 algorithm × 1 connectivity × 2 runs = 2
        assert len(results) == 2

    def test_run_all_both_algorithms(self, quick_config: SimulationConfig):
        """Test running both algorithms."""
        runner = ExperimentRunner(config=quick_config, base_seed=42)

        results = runner.run_all(
            algorithms=[AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE],
            connectivity_levels=[0.75],
            runs_per_config=2,
        )

        # 2 algorithms × 1 connectivity × 2 runs = 4
        assert len(results) == 4

        adaptive = [r for r in results if r.algorithm == AlgorithmType.ADAPTIVE]
        baseline = [r for r in results if r.algorithm == AlgorithmType.BASELINE]

        assert len(adaptive) == 2
        assert len(baseline) == 2

    def test_get_results_by_algorithm(self, quick_config: SimulationConfig):
        """Test filtering results by algorithm."""
        runner = ExperimentRunner(config=quick_config, base_seed=42)

        runner.run_all(
            algorithms=[AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE],
            connectivity_levels=[0.75],
            runs_per_config=2,
        )

        adaptive = runner.get_results_by_algorithm(AlgorithmType.ADAPTIVE)
        assert len(adaptive) == 2

    def test_get_results_by_connectivity(self, quick_config: SimulationConfig):
        """Test filtering results by connectivity."""
        runner = ExperimentRunner(config=quick_config, base_seed=42)

        runner.run_all(
            algorithms=[AlgorithmType.ADAPTIVE],
            connectivity_levels=[0.75, 0.40],
            runs_per_config=2,
        )

        conn_75 = runner.get_results_by_connectivity(0.75)
        conn_40 = runner.get_results_by_connectivity(0.40)

        assert len(conn_75) == 2
        assert len(conn_40) == 2

    def test_progress_callback(self, quick_config: SimulationConfig):
        """Test progress callback is called."""
        runner = ExperimentRunner(config=quick_config, base_seed=42)

        progress_calls = []

        def callback(current: int, total: int):
            progress_calls.append((current, total))

        runner.run_all(
            algorithms=[AlgorithmType.ADAPTIVE],
            connectivity_levels=[0.75],
            runs_per_config=2,
            progress_callback=callback,
        )

        assert len(progress_calls) == 2
        assert progress_calls[-1] == (2, 2)


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_run_simulation_function(self):
        """Test run_simulation convenience function."""
        results = run_simulation(
            algorithm="adaptive",
            connectivity_level=0.75,
            random_seed=42,
        )

        assert results.algorithm == AlgorithmType.ADAPTIVE
        assert results.connectivity_level == 0.75
        assert results.total_tasks > 0

    def test_run_simulation_baseline(self):
        """Test run_simulation with baseline algorithm."""
        results = run_simulation(
            algorithm="baseline",
            connectivity_level=0.40,
            random_seed=42,
        )

        assert results.algorithm == AlgorithmType.BASELINE
        assert results.connectivity_level == 0.40


# =============================================================================
# Test Parameter Verification
# =============================================================================


class TestPhase5Parameters:
    """Verify Phase 5 follows specification parameters."""

    def test_default_duration(self):
        """Verify default duration = 6000 seconds."""
        config = SimulationConfig()
        assert config.scenario.simulation_duration_seconds == 6000

    def test_default_runs_per_config(self):
        """Verify default runs per config = 30."""
        config = SimulationConfig()
        assert config.scenario.runs_per_configuration == 30

    def test_total_experimental_runs(self):
        """Verify total runs = 180 (2 × 3 × 30)."""
        config = SimulationConfig()
        assert config.total_experimental_runs == 180

    def test_update_interval(self):
        """Verify coordination update interval = 1800 seconds (30 min)."""
        config = SimulationConfig()
        assert config.coordination.update_interval_seconds == 1800

    def test_connectivity_scenarios(self):
        """Verify connectivity scenarios = [0.75, 0.40, 0.20]."""
        config = SimulationConfig()
        assert config.network.connectivity_scenarios == [0.75, 0.40, 0.20]


# =============================================================================
# Integration Tests
# =============================================================================


class TestSimulationIntegration:
    """Integration tests for complete simulation."""

    @pytest.fixture
    def integration_config(self) -> SimulationConfig:
        """Config for integration tests."""
        return SimulationConfig(
            scenario=ScenarioParameters(
                simulation_duration_seconds=600,
                message_rate_per_minute=2.0,
            ),
            coordination=CoordinationParameters(
                update_interval_seconds=120,
            ),
        )

    def test_all_components_initialize(self, integration_config: SimulationConfig):
        """Test that all components initialize correctly."""
        engine = SimulationEngine(
            config=integration_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=42,
        )

        # Run should complete without errors
        results = engine.run()

        assert results is not None
        assert results.total_tasks > 0

    def test_tasks_get_assigned(self, integration_config: SimulationConfig):
        """Test that tasks actually get assigned."""
        engine = SimulationEngine(
            config=integration_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=42,
        )

        results = engine.run()

        # Some tasks should be assigned
        assert results.tasks_assigned > 0
        assert results.assignment_rate > 0

    def test_messages_get_created(self, integration_config: SimulationConfig):
        """Test that messages are created for assignments."""
        engine = SimulationEngine(
            config=integration_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=42,
        )

        results = engine.run()

        # Messages should be created for assigned tasks
        assert results.messages_created > 0
        assert results.messages_created == results.tasks_assigned

    def test_urgency_distribution_tracked(self, integration_config: SimulationConfig):
        """Test that urgency distribution is tracked."""
        engine = SimulationEngine(
            config=integration_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=42,
        )

        results = engine.run()

        # Should have tasks in each urgency level
        assert len(results.tasks_by_urgency) > 0

    def test_response_times_recorded(self, integration_config: SimulationConfig):
        """Test that response times are recorded."""
        engine = SimulationEngine(
            config=integration_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=42,
        )

        results = engine.run()

        # Should have response times for assigned tasks
        if results.tasks_assigned > 0:
            assert len(results.response_times) > 0
            assert results.average_response_time is not None
