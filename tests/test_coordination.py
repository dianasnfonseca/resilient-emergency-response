"""
Tests for Coordination Layer (Phase 4).

These tests verify that the coordination algorithms correctly implement
the specifications from the Phase 4 documentation:
- Adaptive: urgency-first ordering, network-aware assignment (P > 0)
- Baseline: FCFS ordering, proximity-only assignment
- 30-minute update interval
- Euclidean distance for proximity
"""

import pytest
from conftest import (
    COORDINATION_INTERVAL_S,
    P_INIT,
    PATH_THRESHOLD,
    PRIORITY_LEVELS,
)

from ercs.config.parameters import (
    AlgorithmType,
    CoordinationParameters,
    UrgencyLevel,
)
from ercs.coordination import (
    AdaptiveCoordinator,
    Assignment,
    BaselineCoordinator,
    CoordinationManager,
    EventType,
    create_coordinator,
)
from ercs.scenario.generator import Task

# =============================================================================
# Mock Implementations for Testing
# =============================================================================


class MockResponderLocator:
    """Mock responder locator for testing."""

    def __init__(self, responders: dict[str, tuple[float, float]]):
        """
        Initialize with responder positions.

        Args:
            responders: Dict mapping responder_id to (x, y) position
        """
        self._responders = responders

    def get_responder_position(self, responder_id: str) -> tuple[float, float]:
        """Get responder position."""
        return self._responders[responder_id]

    def get_all_responder_ids(self) -> list[str]:
        """Get all responder IDs."""
        return list(self._responders.keys())


class MockNetworkState:
    """Mock network state provider for testing."""

    def __init__(self, predictabilities: dict[tuple[str, str], float] | None = None):
        """
        Initialize with predictability values.

        Args:
            predictabilities: Dict mapping (from, to) to P value
        """
        self._predictabilities = predictabilities or {}

    def get_delivery_predictability(self, from_node: str, to_node: str) -> float:
        """Get delivery predictability."""
        return self._predictabilities.get((from_node, to_node), 0.0)

    def get_last_encounter_time(self, from_node: str, to_node: str) -> float:
        """Get time of last direct encounter (0.0 if never)."""
        return 0.0

    def set_predictability(self, from_node: str, to_node: str, value: float) -> None:
        """Set a predictability value."""
        self._predictabilities[(from_node, to_node)] = value


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_responders() -> MockResponderLocator:
    """Create sample responders at known positions."""
    return MockResponderLocator(
        {
            "mobile_0": (100.0, 500.0),
            "mobile_1": (200.0, 600.0),
            "mobile_2": (300.0, 700.0),
            "mobile_3": (400.0, 800.0),
            "mobile_4": (500.0, 500.0),
        }
    )


@pytest.fixture
def full_connectivity_network() -> MockNetworkState:
    """Create network state with full connectivity (all P > 0)."""
    network = MockNetworkState()
    for i in range(5):
        network.set_predictability("coord_0", f"mobile_{i}", P_INIT)
    return network


@pytest.fixture
def partial_connectivity_network() -> MockNetworkState:
    """Create network state with partial connectivity."""
    network = MockNetworkState()
    # Only some responders reachable (P > 0.3 threshold)
    network.set_predictability("coord_0", "mobile_0", P_INIT)
    network.set_predictability("coord_0", "mobile_1", 0.50)
    network.set_predictability("coord_0", "mobile_2", 0.0)  # Not reachable
    network.set_predictability("coord_0", "mobile_3", 0.0)  # Not reachable
    network.set_predictability("coord_0", "mobile_4", 0.40)
    return network


@pytest.fixture
def sample_tasks() -> list[Task]:
    """Create sample tasks with different urgencies."""
    return [
        Task(
            task_id="task_0001",
            creation_time=10.0,
            source_node="coord_0",
            target_location_x=150.0,
            target_location_y=550.0,
            urgency=UrgencyLevel.LOW,
        ),
        Task(
            task_id="task_0002",
            creation_time=20.0,
            source_node="coord_0",
            target_location_x=250.0,
            target_location_y=650.0,
            urgency=UrgencyLevel.HIGH,
        ),
        Task(
            task_id="task_0003",
            creation_time=30.0,
            source_node="coord_0",
            target_location_x=350.0,
            target_location_y=750.0,
            urgency=UrgencyLevel.MEDIUM,
        ),
    ]


# =============================================================================
# Test Assignment Class
# =============================================================================


class TestAssignment:
    """Tests for the Assignment class."""

    def test_assignment_creation(self):
        """Test assignment creation."""
        task = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=100.0,
            urgency=UrgencyLevel.HIGH,
        )

        assignment = Assignment(
            task=task,
            responder_id="mobile_1",
            assignment_time=50.0,
            distance=141.4,
            predictability=0.75,
        )

        assert assignment.task_id == "task_0001"
        assert assignment.responder_id == "mobile_1"
        assert assignment.assignment_time == 50.0
        assert assignment.distance == 141.4
        assert assignment.predictability == 0.75


# =============================================================================
# Test AdaptiveCoordinator
# =============================================================================


class TestAdaptiveCoordinator:
    """Tests for the adaptive coordination algorithm."""

    @pytest.fixture
    def coordinator(self) -> AdaptiveCoordinator:
        """Create adaptive coordinator with default params."""
        return AdaptiveCoordinator()

    def test_coordinator_creation(self, coordinator: AdaptiveCoordinator):
        """Test coordinator initialization."""
        assert coordinator.algorithm_type == AlgorithmType.ADAPTIVE
        assert coordinator.params is not None

    def test_requires_network_state(
        self,
        coordinator: AdaptiveCoordinator,
        sample_tasks: list[Task],
        sample_responders: MockResponderLocator,
    ):
        """Test that adaptive algorithm requires network state."""
        with pytest.raises(ValueError, match="requires network_state"):
            coordinator.assign_tasks(
                tasks=sample_tasks,
                responder_locator=sample_responders,
                network_state=None,  # Missing!
                coordination_node="coord_0",
                current_time=100.0,
            )

    def test_urgency_first_ordering(
        self,
        coordinator: AdaptiveCoordinator,
        sample_tasks: list[Task],
        sample_responders: MockResponderLocator,
        full_connectivity_network: MockNetworkState,
    ):
        """Test that high-urgency tasks are assigned first."""
        assignments = coordinator.assign_tasks(
            tasks=sample_tasks,
            responder_locator=sample_responders,
            network_state=full_connectivity_network,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 3

        # Check order: HIGH first, then MEDIUM, then LOW
        [a.task.urgency for a in assignments]

        # First should be HIGH (task_0002)
        assert assignments[0].task.task_id == "task_0002"
        assert assignments[0].task.urgency == UrgencyLevel.HIGH

    def test_only_reachable_responders(
        self,
        coordinator: AdaptiveCoordinator,
        sample_tasks: list[Task],
        sample_responders: MockResponderLocator,
        partial_connectivity_network: MockNetworkState,
    ):
        """Test that only responders with P > 0 are considered."""
        # Use just one task near mobile_2 (which is unreachable)
        task = Task(
            task_id="task_near_mobile_2",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=300.0,  # Near mobile_2 at (300, 700)
            target_location_y=700.0,
            urgency=UrgencyLevel.HIGH,
        )

        assignments = coordinator.assign_tasks(
            tasks=[task],
            responder_locator=sample_responders,
            network_state=partial_connectivity_network,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 1
        # Should NOT be assigned to mobile_2 (P=0), even though it's nearest
        assert assignments[0].responder_id != "mobile_2"
        assert assignments[0].responder_id != "mobile_3"
        # Should be assigned to next nearest reachable responder

    def test_nearest_among_reachable(
        self,
        coordinator: AdaptiveCoordinator,
        sample_responders: MockResponderLocator,
        partial_connectivity_network: MockNetworkState,
    ):
        """Test that nearest reachable responder is selected."""
        # Task near mobile_0 (which is reachable)
        task = Task(
            task_id="task_near_mobile_0",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=500.0,
            urgency=UrgencyLevel.MEDIUM,
        )

        assignments = coordinator.assign_tasks(
            tasks=[task],
            responder_locator=sample_responders,
            network_state=partial_connectivity_network,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 1
        # mobile_0 is nearest and reachable
        assert assignments[0].responder_id == "mobile_0"
        assert assignments[0].distance == pytest.approx(0.0, abs=0.1)

    def test_logs_assignment_events(
        self,
        coordinator: AdaptiveCoordinator,
        sample_tasks: list[Task],
        sample_responders: MockResponderLocator,
        full_connectivity_network: MockNetworkState,
    ):
        """Test that assignment events are logged."""
        coordinator.assign_tasks(
            tasks=sample_tasks,
            responder_locator=sample_responders,
            network_state=full_connectivity_network,
            coordination_node="coord_0",
            current_time=100.0,
        )

        events = coordinator.get_events_by_type(EventType.TASK_ASSIGNED)
        assert len(events) == 3

        # Check event details
        for event in events:
            assert event.timestamp == 100.0
            assert event.task_id is not None
            assert event.responder_id is not None
            assert "urgency" in event.details
            assert "distance" in event.details

    def test_tracks_predictability(
        self,
        coordinator: AdaptiveCoordinator,
        sample_responders: MockResponderLocator,
        full_connectivity_network: MockNetworkState,
    ):
        """Test that predictability is tracked in assignments."""
        task = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=500.0,
            urgency=UrgencyLevel.HIGH,
        )

        assignments = coordinator.assign_tasks(
            tasks=[task],
            responder_locator=sample_responders,
            network_state=full_connectivity_network,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert assignments[0].predictability == P_INIT

    def test_no_assignment_when_all_unreachable(
        self,
        coordinator: AdaptiveCoordinator,
        sample_responders: MockResponderLocator,
    ):
        """Test handling when no responders are reachable."""
        # Network with no reachable responders
        network = MockNetworkState()  # All P = 0

        task = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=200.0,
            target_location_y=600.0,
            urgency=UrgencyLevel.HIGH,
        )

        assignments = coordinator.assign_tasks(
            tasks=[task],
            responder_locator=sample_responders,
            network_state=network,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 0

        # Should log failure event
        failures = coordinator.get_events_by_type(EventType.ASSIGNMENT_FAILED)
        assert len(failures) == 1
        assert failures[0].details["reason"] == "no_reachable_responder"


# =============================================================================
# Test BaselineCoordinator
# =============================================================================


class TestBaselineCoordinator:
    """Tests for the baseline coordination algorithm."""

    @pytest.fixture
    def coordinator(self) -> BaselineCoordinator:
        """Create baseline coordinator with default params."""
        return BaselineCoordinator()

    def test_coordinator_creation(self, coordinator: BaselineCoordinator):
        """Test coordinator initialization."""
        assert coordinator.algorithm_type == AlgorithmType.BASELINE
        assert coordinator.params is not None

    def test_fcfs_ordering(
        self,
        coordinator: BaselineCoordinator,
        sample_tasks: list[Task],
        sample_responders: MockResponderLocator,
    ):
        """Test that tasks are processed in creation order (FCFS)."""
        assignments = coordinator.assign_tasks(
            tasks=sample_tasks,
            responder_locator=sample_responders,
            network_state=None,  # Not required for baseline
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 3

        # Check order: by creation time regardless of urgency
        creation_times = [a.task.creation_time for a in assignments]
        assert creation_times == sorted(creation_times)

        # First should be task_0001 (created at 10.0, LOW urgency)
        assert assignments[0].task.task_id == "task_0001"

    def test_nearest_responder_selected(
        self,
        coordinator: BaselineCoordinator,
        sample_responders: MockResponderLocator,
    ):
        """Test that nearest responder is selected."""
        # Task exactly at mobile_0's position
        task = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=500.0,
            urgency=UrgencyLevel.MEDIUM,
        )

        assignments = coordinator.assign_tasks(
            tasks=[task],
            responder_locator=sample_responders,
            network_state=None,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 1
        assert assignments[0].responder_id == "mobile_0"
        assert assignments[0].distance == pytest.approx(0.0, abs=0.1)

    def test_ignores_network_state(
        self,
        coordinator: BaselineCoordinator,
        sample_responders: MockResponderLocator,
    ):
        """Test that baseline ignores network state (assigns to unreachable)."""
        # Network where mobile_0 is unreachable
        network = MockNetworkState()
        network.set_predictability("coord_0", "mobile_0", 0.0)  # Unreachable
        network.set_predictability("coord_0", "mobile_1", 0.75)

        # Task nearest to mobile_0
        task = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=500.0,
            urgency=UrgencyLevel.HIGH,
        )

        assignments = coordinator.assign_tasks(
            tasks=[task],
            responder_locator=sample_responders,
            network_state=network,  # Provided but should be ignored
            coordination_node="coord_0",
            current_time=100.0,
        )

        # Should still assign to mobile_0 (nearest) even though P=0
        assert assignments[0].responder_id == "mobile_0"

    def test_no_predictability_tracked(
        self,
        coordinator: BaselineCoordinator,
        sample_responders: MockResponderLocator,
    ):
        """Test that baseline doesn't track predictability."""
        task = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=500.0,
            urgency=UrgencyLevel.LOW,
        )

        assignments = coordinator.assign_tasks(
            tasks=[task],
            responder_locator=sample_responders,
            network_state=None,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert assignments[0].predictability is None

    def test_logs_events(
        self,
        coordinator: BaselineCoordinator,
        sample_tasks: list[Task],
        sample_responders: MockResponderLocator,
    ):
        """Test that baseline logs events."""
        coordinator.assign_tasks(
            tasks=sample_tasks,
            responder_locator=sample_responders,
            network_state=None,
            coordination_node="coord_0",
            current_time=100.0,
        )

        events = coordinator.get_events()
        assert len(events) > 0

        # Should have cycle event and assignment events
        cycles = coordinator.get_events_by_type(EventType.COORDINATION_CYCLE)
        assert len(cycles) == 1

        assignments = coordinator.get_events_by_type(EventType.TASK_ASSIGNED)
        assert len(assignments) == 3


# =============================================================================
# Test Algorithm Comparison
# =============================================================================


class TestAlgorithmComparison:
    """Tests comparing adaptive vs baseline algorithms."""

    def test_same_result_with_full_connectivity(
        self,
        sample_responders: MockResponderLocator,
        full_connectivity_network: MockNetworkState,
    ):
        """With full connectivity, both algorithms should assign to nearest."""
        adaptive = AdaptiveCoordinator()
        baseline = BaselineCoordinator()

        # Single task to eliminate ordering differences
        task = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=500.0,
            urgency=UrgencyLevel.MEDIUM,
        )

        adaptive_assignments = adaptive.assign_tasks(
            tasks=[task],
            responder_locator=sample_responders,
            network_state=full_connectivity_network,
            coordination_node="coord_0",
            current_time=100.0,
        )

        # Reset task for baseline
        task_copy = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=500.0,
            urgency=UrgencyLevel.MEDIUM,
        )

        baseline_assignments = baseline.assign_tasks(
            tasks=[task_copy],
            responder_locator=sample_responders,
            network_state=None,
            coordination_node="coord_0",
            current_time=100.0,
        )

        # Both should assign to mobile_0 (nearest)
        assert adaptive_assignments[0].responder_id == "mobile_0"
        assert baseline_assignments[0].responder_id == "mobile_0"

    def test_different_result_with_partial_connectivity(
        self,
        sample_responders: MockResponderLocator,
        partial_connectivity_network: MockNetworkState,
    ):
        """With partial connectivity, algorithms may choose differently."""
        adaptive = AdaptiveCoordinator()
        baseline = BaselineCoordinator()

        # Task nearest to mobile_2 (which is unreachable for adaptive)
        task_adaptive = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=300.0,
            target_location_y=700.0,
            urgency=UrgencyLevel.HIGH,
        )

        task_baseline = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=300.0,
            target_location_y=700.0,
            urgency=UrgencyLevel.HIGH,
        )

        adaptive_assignments = adaptive.assign_tasks(
            tasks=[task_adaptive],
            responder_locator=sample_responders,
            network_state=partial_connectivity_network,
            coordination_node="coord_0",
            current_time=100.0,
        )

        baseline_assignments = baseline.assign_tasks(
            tasks=[task_baseline],
            responder_locator=sample_responders,
            network_state=None,
            coordination_node="coord_0",
            current_time=100.0,
        )

        # Adaptive should NOT assign to mobile_2 (P=0)
        assert adaptive_assignments[0].responder_id != "mobile_2"

        # Baseline WILL assign to mobile_2 (nearest)
        assert baseline_assignments[0].responder_id == "mobile_2"


# =============================================================================
# Test CoordinationManager
# =============================================================================


class TestCoordinationManager:
    """Tests for the CoordinationManager class."""

    @pytest.fixture
    def manager(self) -> CoordinationManager:
        """Create manager with baseline coordinator."""
        coordinator = BaselineCoordinator()
        return CoordinationManager(coordinator)

    def test_manager_creation(self, manager: CoordinationManager):
        """Test manager initialization."""
        assert manager.coordinator is not None
        assert manager.get_pending_count() == 0

    def test_add_task(self, manager: CoordinationManager):
        """Test adding tasks to manager."""
        task = Task(
            task_id="task_0001",
            creation_time=0.0,
            source_node="coord_0",
            target_location_x=100.0,
            target_location_y=100.0,
            urgency=UrgencyLevel.HIGH,
        )

        manager.add_task(task)
        assert manager.get_pending_count() == 1

    def test_add_multiple_tasks(
        self, manager: CoordinationManager, sample_tasks: list[Task]
    ):
        """Test adding multiple tasks."""
        manager.add_tasks(sample_tasks)
        assert manager.get_pending_count() == 3

    def test_should_update_timing(self, manager: CoordinationManager):
        """Test update interval checking."""
        # Default interval is 1800 seconds (30 min)
        # At t=0 with last_update=0, should wait for interval
        assert manager.should_update(0.0) is False
        assert manager.should_update(1000.0) is False
        assert manager.should_update(float(COORDINATION_INTERVAL_S)) is True
        assert manager.should_update(2000.0) is True

    def test_run_coordination_cycle(
        self,
        manager: CoordinationManager,
        sample_tasks: list[Task],
        sample_responders: MockResponderLocator,
    ):
        """Test running a coordination cycle."""
        manager.add_tasks(sample_tasks)

        assignments = manager.run_coordination_cycle(
            responder_locator=sample_responders,
            network_state=None,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 3
        assert manager.get_pending_count() == 0

    def test_tasks_only_assigned_after_arrival(
        self,
        manager: CoordinationManager,
        sample_responders: MockResponderLocator,
    ):
        """Test that tasks aren't assigned before their creation time."""
        # Tasks at times 10, 20, 30
        tasks = [
            Task(
                task_id=f"task_{i}",
                creation_time=float(i * 10 + 10),
                source_node="coord_0",
                target_location_x=100.0,
                target_location_y=100.0,
                urgency=UrgencyLevel.MEDIUM,
            )
            for i in range(3)
        ]
        manager.add_tasks(tasks)

        # Run cycle at time 15 (only first task has arrived)
        assignments = manager.run_coordination_cycle(
            responder_locator=sample_responders,
            network_state=None,
            coordination_node="coord_0",
            current_time=15.0,
        )

        assert len(assignments) == 1
        assert assignments[0].task_id == "task_0"

    def test_statistics(
        self,
        manager: CoordinationManager,
        sample_tasks: list[Task],
        sample_responders: MockResponderLocator,
    ):
        """Test statistics collection."""
        manager.add_tasks(sample_tasks)
        manager.run_coordination_cycle(
            responder_locator=sample_responders,
            network_state=None,
            coordination_node="coord_0",
            current_time=100.0,
        )

        stats = manager.statistics
        assert stats["total_assignments_made"] == 3
        assert stats["cycles_completed"] == 1
        assert stats["pending_tasks"] == 0

    def test_reset(
        self,
        manager: CoordinationManager,
        sample_tasks: list[Task],
        sample_responders: MockResponderLocator,
    ):
        """Test resetting manager state."""
        manager.add_tasks(sample_tasks)
        manager.run_coordination_cycle(
            responder_locator=sample_responders,
            network_state=None,
            coordination_node="coord_0",
            current_time=100.0,
        )

        manager.reset()

        assert manager.get_pending_count() == 0
        assert len(manager.get_all_assignments()) == 0
        assert manager.statistics["cycles_completed"] == 0


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateCoordinator:
    """Tests for the create_coordinator factory function."""

    def test_create_adaptive(self):
        """Test creating adaptive coordinator."""
        coordinator = create_coordinator("adaptive")
        assert isinstance(coordinator, AdaptiveCoordinator)
        assert coordinator.algorithm_type == AlgorithmType.ADAPTIVE

    def test_create_baseline(self):
        """Test creating baseline coordinator."""
        coordinator = create_coordinator("baseline")
        assert isinstance(coordinator, BaselineCoordinator)
        assert coordinator.algorithm_type == AlgorithmType.BASELINE

    def test_create_with_enum(self):
        """Test creating with AlgorithmType enum."""
        coordinator = create_coordinator(AlgorithmType.ADAPTIVE)
        assert isinstance(coordinator, AdaptiveCoordinator)

    def test_create_with_params(self):
        """Test creating with custom parameters."""
        params = CoordinationParameters(update_interval_seconds=600)
        coordinator = create_coordinator("baseline", params=params)
        assert coordinator.params.update_interval_seconds == 600


# =============================================================================
# Test Distance Calculation
# =============================================================================


class TestDistanceCalculation:
    """Tests for Euclidean distance calculation."""

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        coordinator = BaselineCoordinator()

        # 3-4-5 triangle
        dist = coordinator._calculate_distance(0, 0, 3, 4)
        assert dist == pytest.approx(5.0)

    def test_same_point_zero_distance(self):
        """Test distance is zero for same point."""
        coordinator = BaselineCoordinator()

        dist = coordinator._calculate_distance(100, 200, 100, 200)
        assert dist == 0.0

    def test_symmetric_distance(self):
        """Test distance is symmetric."""
        coordinator = BaselineCoordinator()

        dist1 = coordinator._calculate_distance(0, 0, 100, 200)
        dist2 = coordinator._calculate_distance(100, 200, 0, 0)
        assert dist1 == dist2


# =============================================================================
# Test Parameter Verification (Phase 4 Spec)
# =============================================================================


class TestPhase4Parameters:
    """Verify Phase 4 parameters match specification."""

    def test_update_interval(self):
        """Verify update interval = 30 minutes."""
        params = CoordinationParameters()
        assert params.update_interval_seconds == COORDINATION_INTERVAL_S  # 30 * 60

    def test_priority_levels(self):
        """Verify 3 priority levels."""
        params = CoordinationParameters()
        assert params.priority_levels == PRIORITY_LEVELS

    def test_path_threshold(self):
        """Verify path threshold = 0.3."""
        params = CoordinationParameters()
        assert params.available_path_threshold == PATH_THRESHOLD

    def test_proximity_method(self):
        """Verify Euclidean distance."""
        params = CoordinationParameters()
        assert params.proximity_method == "euclidean"

    def test_adaptive_task_order(self):
        """Verify urgency-first ordering for adaptive."""
        params = CoordinationParameters()
        assert params.adaptive_task_order == "urgency_first"

    def test_baseline_task_order(self):
        """Verify FCFS ordering for baseline."""
        params = CoordinationParameters()
        assert params.baseline_task_order == "fcfs"


# =============================================================================
# Integration Tests
# =============================================================================


class TestCoordinationIntegration:
    """Integration tests for coordination scenarios."""

    def test_full_coordination_scenario(
        self,
        sample_responders: MockResponderLocator,
        full_connectivity_network: MockNetworkState,
    ):
        """Test a complete coordination scenario."""
        # Create tasks arriving over time (spread across cycles)
        tasks = [
            Task(
                task_id=f"task_{i:04d}",
                creation_time=float(i * 600),  # Every 10 minutes
                source_node="coord_0",
                target_location_x=150.0 + i * 50,
                target_location_y=550.0 + i * 30,
                urgency=[UrgencyLevel.HIGH, UrgencyLevel.MEDIUM, UrgencyLevel.LOW][
                    i % 3
                ],
            )
            for i in range(10)
        ]

        # Use adaptive coordinator
        coordinator = AdaptiveCoordinator()
        manager = CoordinationManager(coordinator)
        manager.add_tasks(tasks)

        # Run multiple coordination cycles (30min = 1800s intervals)
        # Tasks arrive at: 0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400
        all_assignments = []

        # Cycle 1 at t=1800: assigns tasks 0,1,2,3 (created at 0,600,1200,1800)
        assignments = manager.run_coordination_cycle(
            responder_locator=sample_responders,
            network_state=full_connectivity_network,
            coordination_node="coord_0",
            current_time=float(COORDINATION_INTERVAL_S),
        )
        all_assignments.extend(assignments)

        # Cycle 2 at t=3600: assigns tasks 4,5,6 (created at 2400,3000,3600)
        assignments = manager.run_coordination_cycle(
            responder_locator=sample_responders,
            network_state=full_connectivity_network,
            coordination_node="coord_0",
            current_time=float(2 * COORDINATION_INTERVAL_S),
        )
        all_assignments.extend(assignments)

        # Cycle 3 at t=5400: assigns tasks 7,8,9 (created at 4200,4800,5400)
        assignments = manager.run_coordination_cycle(
            responder_locator=sample_responders,
            network_state=full_connectivity_network,
            coordination_node="coord_0",
            current_time=float(3 * COORDINATION_INTERVAL_S),
        )
        all_assignments.extend(assignments)

        # All 10 tasks should be assigned
        assert len(all_assignments) == 10

        # Check statistics
        stats = manager.statistics
        assert stats["total_assignments_made"] == 10
        assert stats["cycles_completed"] == 3

    def test_degraded_connectivity_scenario(
        self,
        sample_responders: MockResponderLocator,
        partial_connectivity_network: MockNetworkState,
    ):
        """Test coordination under degraded connectivity."""
        tasks = [
            Task(
                task_id=f"task_{i:04d}",
                creation_time=0.0,
                source_node="coord_0",
                target_location_x=200.0 + i * 100,
                target_location_y=600.0,
                urgency=UrgencyLevel.HIGH,
            )
            for i in range(5)
        ]

        coordinator = AdaptiveCoordinator()
        assignments = coordinator.assign_tasks(
            tasks=tasks,
            responder_locator=sample_responders,
            network_state=partial_connectivity_network,
            coordination_node="coord_0",
            current_time=100.0,
        )

        # All tasks should be assigned
        assert len(assignments) == 5

        # Check that only reachable responders were used
        reachable = {"mobile_0", "mobile_1", "mobile_4"}
        for assignment in assignments:
            assert assignment.responder_id in reachable
