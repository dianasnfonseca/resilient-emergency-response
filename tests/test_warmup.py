"""
Tests for PRoPHET warm-up period implementation.

These tests verify that:
1. Predictability starts at zero for all pairs
2. Predictability builds through encounters during warm-up
3. Tasks are not generated during warm-up
4. Coordination cycles only occur after warm-up
5. WARMUP_END event is logged with statistics
"""

import pytest

from ercs.config.parameters import (
    AlgorithmType,
    CoordinationParameters,
    ScenarioParameters,
    SimulationConfig,
)
from ercs.simulation.engine import (
    SimulationEngine,
    SimulationEventType,
)

from conftest import CONNECTIVITY_MILD


class TestWarmupPeriod:
    """Tests for warm-up period functionality."""

    @pytest.fixture
    def warmup_config(self) -> SimulationConfig:
        """Config with standard warm-up period."""
        return SimulationConfig(
            scenario=ScenarioParameters(
                warmup_period_seconds=300,  # 5 min for faster tests
                simulation_duration_seconds=600,
                message_rate_per_minute=2.0,
            ),
            coordination=CoordinationParameters(
                update_interval_seconds=120,
            ),
        )

    def test_initial_predictability_is_zero(
        self, warmup_config: SimulationConfig
    ):
        """Test that all predictabilities start at zero."""
        engine = SimulationEngine(
            config=warmup_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=CONNECTIVITY_MILD,
            random_seed=42,
        )

        # Initialize components but don't run
        engine._initialize_components()

        # Check all coord->mobile pairs have P=0
        coord_nodes = engine._topology.get_coordination_node_ids()
        mobile_nodes = engine._topology.get_mobile_responder_ids()

        for coord_id in coord_nodes:
            for mobile_id in mobile_nodes:
                p = engine._communication.get_delivery_predictability(
                    coord_id, mobile_id
                )
                assert p == 0.0, (
                    f"Expected P=0 for ({coord_id}, {mobile_id}), got {p}"
                )

    def test_predictability_builds_during_warmup(
        self, warmup_config: SimulationConfig
    ):
        """Test that encounters during warm-up build predictability."""
        engine = SimulationEngine(
            config=warmup_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=CONNECTIVITY_MILD,
            random_seed=42,
        )

        engine.run()

        # After warm-up + simulation, some pairs should have nonzero P
        coord_nodes = engine._topology.get_coordination_node_ids()
        mobile_nodes = engine._topology.get_mobile_responder_ids()

        nonzero_count = 0
        for coord_id in coord_nodes:
            for mobile_id in mobile_nodes:
                p = engine._communication.get_delivery_predictability(
                    coord_id, mobile_id
                )
                if p > 0:
                    nonzero_count += 1

        assert nonzero_count > 0, "No predictability built during warm-up"

    def test_no_tasks_during_warmup(self, warmup_config: SimulationConfig):
        """Test that no tasks are created during warm-up period."""
        engine = SimulationEngine(
            config=warmup_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=CONNECTIVITY_MILD,
            random_seed=42,
        )

        results = engine.run()

        warmup_end = warmup_config.scenario.warmup_period_seconds

        task_events = [
            e
            for e in results.events
            if e.event_type == SimulationEventType.TASK_CREATED
        ]

        for event in task_events:
            assert event.timestamp >= warmup_end, (
                f"Task created at {event.timestamp} before "
                f"warm-up end {warmup_end}"
            )

    def test_coordination_only_after_warmup(
        self, warmup_config: SimulationConfig
    ):
        """Test that coordination cycles only occur after warm-up."""
        engine = SimulationEngine(
            config=warmup_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=CONNECTIVITY_MILD,
            random_seed=42,
        )

        results = engine.run()

        warmup_end = warmup_config.scenario.warmup_period_seconds

        coord_events = [
            e
            for e in results.events
            if e.event_type == SimulationEventType.COORDINATION_CYCLE
        ]

        for event in coord_events:
            assert event.timestamp >= warmup_end, (
                f"Coordination at {event.timestamp} before "
                f"warm-up end {warmup_end}"
            )

    def test_warmup_end_event_logged(self, warmup_config: SimulationConfig):
        """Test that WARMUP_END event is logged with statistics."""
        engine = SimulationEngine(
            config=warmup_config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=CONNECTIVITY_MILD,
            random_seed=42,
        )

        results = engine.run()

        warmup_events = [
            e
            for e in results.events
            if e.event_type == SimulationEventType.WARMUP_END
        ]

        assert len(warmup_events) == 1
        assert warmup_events[0].timestamp == (
            warmup_config.scenario.warmup_period_seconds
        )

        # Should contain predictability statistics
        data = warmup_events[0].data
        assert "nonzero_predictabilities" in data
        assert "total_pairs" in data
        assert "coverage_pct" in data

    def test_no_warmup_event_when_zero(self):
        """Test that no WARMUP_END event fires when warmup is 0."""
        config = SimulationConfig(
            scenario=ScenarioParameters(
                warmup_period_seconds=0,
                simulation_duration_seconds=300,
                message_rate_per_minute=1.0,
            ),
            coordination=CoordinationParameters(
                update_interval_seconds=120,
            ),
        )

        engine = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.BASELINE,
            connectivity_level=CONNECTIVITY_MILD,
            random_seed=42,
        )

        results = engine.run()

        warmup_events = [
            e
            for e in results.events
            if e.event_type == SimulationEventType.WARMUP_END
        ]

        assert len(warmup_events) == 0

    def test_total_simulation_duration_property(self):
        """Test total_simulation_duration includes warm-up."""
        config = SimulationConfig(
            scenario=ScenarioParameters(
                warmup_period_seconds=1800,
                simulation_duration_seconds=6000,
            ),
        )
        assert config.total_simulation_duration == 7800

    def test_total_simulation_duration_no_warmup(self):
        """Test total_simulation_duration with no warm-up."""
        config = SimulationConfig(
            scenario=ScenarioParameters(
                warmup_period_seconds=0,
                simulation_duration_seconds=6000,
            ),
        )
        assert config.total_simulation_duration == 6000
