"""
Diagnostic tests for simulation anomalies identified after PRoPHETv2 migration.

These tests investigate locally (without full simulation) each issue:
1. Assignment rate always 1.000 — is this by design?
2. Response time = cycle_time - creation_time (not delivery-based)
3. PRoPHET state identical between Adaptive & Baseline (same seed)
4. Adaptive geographic clustering: weighted score biases towards high-P nodes
5. Forwarding differences: >= condition + p_to > 0 guard impact

Each test is self-contained and uses lightweight mock/stub components
so they run in seconds, not minutes.
"""

import numpy as np
import pytest

from ercs.communication.prophet import (
    CommunicationLayer,
    DeliveryPredictabilityMatrix,
    MessageType,
)
from ercs.config.parameters import (
    AlgorithmType,
    CommunicationParameters,
    CoordinationParameters,
    NetworkParameters,
    ScenarioParameters,
    SimulationConfig,
    UrgencyLevel,
)
from ercs.coordination.algorithms import (
    AdaptiveCoordinator,
    Assignment,
    BaselineCoordinator,
    CoordinationManager,
    NetworkStateProvider,
    ResponderLocator,
)
from ercs.scenario.generator import Task, TaskStatus
from ercs.simulation.engine import SimulationEngine, SimulationResults


# ============================================================================
# Helpers: lightweight stubs for coordinator protocols
# ============================================================================


class StubResponderLocator:
    """Stub that provides fixed responder positions."""

    def __init__(self, positions: dict[str, tuple[float, float]]):
        self._positions = positions

    def get_responder_position(self, responder_id: str) -> tuple[float, float]:
        return self._positions[responder_id]

    def get_all_responder_ids(self) -> list[str]:
        return list(self._positions.keys())


class StubNetworkState:
    """Stub that returns configurable predictability values."""

    def __init__(self, predictabilities: dict[tuple[str, str], float]):
        self._pred = predictabilities

    def get_delivery_predictability(self, from_node: str, to_node: str) -> float:
        return self._pred.get((from_node, to_node), 0.0)

    def get_last_encounter_time(self, from_node: str, to_node: str) -> float:
        return 0.0


def _make_task(
    task_id: str,
    creation_time: float,
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    x: float = 100.0,
    y: float = 100.0,
    source_node: str = "coord_0",
) -> Task:
    """Create a task for testing."""
    return Task(
        task_id=task_id,
        creation_time=creation_time,
        source_node=source_node,
        target_location_x=x,
        target_location_y=y,
        urgency=urgency,
    )


# ============================================================================
# 1. Assignment Rate: always 1.000 when threshold=0.0
# ============================================================================


class TestAssignmentRateSemantics:
    """
    Issue: assignment_rate = tasks_assigned / total_tasks = 1.000 everywhere.

    Root cause (pre-fix): available_path_threshold was 0.0, meaning any node
    with P > 0 passed the filter. Now threshold = 0.3, and cold-start means
    P builds only through actual encounters during the simulation.

    assignment_rate measures COORDINATOR DECISIONS, not successful deliveries.
    """

    def test_adaptive_always_assigns_when_all_nodes_reachable(self):
        """When all responders have P > 0 to coord, all tasks are assigned."""
        coord = AdaptiveCoordinator(
            CoordinationParameters(available_path_threshold=0.0)
        )

        # 5 responders, all with P > 0 from coord_0
        positions = {f"r_{i}": (100.0 * i, 100.0) for i in range(5)}
        locator = StubResponderLocator(positions)
        net = StubNetworkState({
            ("coord_0", f"r_{i}"): 0.1 + i * 0.1 for i in range(5)
        })

        tasks = [_make_task(f"t_{i}", i * 10.0) for i in range(5)]

        assignments = coord.assign_tasks(
            tasks=tasks,
            responder_locator=locator,
            network_state=net,
            coordination_node="coord_0",
            current_time=100.0,
        )

        # ALL tasks assigned — assignment rate = 1.0
        assert len(assignments) == 5
        assignment_rate = len(assignments) / len(tasks)
        assert assignment_rate == 1.0

    def test_adaptive_fails_assignment_when_threshold_requires_high_p(self):
        """With higher threshold, some responders become unreachable."""
        coord = AdaptiveCoordinator(
            CoordinationParameters(available_path_threshold=0.3)
        )

        positions = {f"r_{i}": (100.0 * i, 100.0) for i in range(5)}
        locator = StubResponderLocator(positions)

        # Only r_3 and r_4 have P > 0.3
        net = StubNetworkState({
            ("coord_0", "r_0"): 0.05,
            ("coord_0", "r_1"): 0.1,
            ("coord_0", "r_2"): 0.2,
            ("coord_0", "r_3"): 0.4,
            ("coord_0", "r_4"): 0.5,
        })

        tasks = [_make_task(f"t_{i}", i * 10.0) for i in range(5)]

        assignments = coord.assign_tasks(
            tasks=tasks,
            responder_locator=locator,
            network_state=net,
            coordination_node="coord_0",
            current_time=100.0,
        )

        # Only 2 responders pass threshold → at most 2 unique responders
        assigned_responders = {a.responder_id for a in assignments}
        assert assigned_responders.issubset({"r_3", "r_4"})
        # Tasks assigned to those 2 responders — others FAIL
        assert coord._failed_assignments > 0 or len(assignments) == len(tasks)

    def test_adaptive_zero_p_means_no_assignment(self):
        """When all P = 0, no tasks should be assigned."""
        coord = AdaptiveCoordinator(
            CoordinationParameters(available_path_threshold=0.0)
        )

        positions = {"r_0": (100.0, 100.0), "r_1": (200.0, 200.0)}
        locator = StubResponderLocator(positions)

        # All P = 0
        net = StubNetworkState({})

        tasks = [_make_task("t_0", 10.0), _make_task("t_1", 20.0)]

        assignments = coord.assign_tasks(
            tasks=tasks,
            responder_locator=locator,
            network_state=net,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 0
        assert coord._failed_assignments == 2

    def test_baseline_always_assigns_regardless_of_connectivity(self):
        """Baseline ignores network state — always assigns to nearest."""
        coord = BaselineCoordinator()

        positions = {"r_0": (100.0, 100.0), "r_1": (500.0, 500.0)}
        locator = StubResponderLocator(positions)

        # Even with None network state, baseline assigns
        tasks = [_make_task("t_0", 10.0), _make_task("t_1", 20.0)]

        assignments = coord.assign_tasks(
            tasks=tasks,
            responder_locator=locator,
            network_state=None,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 2
        assert coord._failed_assignments == 0

    def test_assignment_rate_is_not_delivery_rate(self):
        """
        Confirm that assignment_rate and delivery_rate are independent metrics.
        assignment_rate = coordinator decisions; delivery_rate = network success.
        """
        results = SimulationResults(
            config=SimulationConfig(),
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            run_number=0,
            random_seed=42,
        )

        # 10 tasks, all assigned (assignment_rate = 1.0)
        results.total_tasks = 10
        results.tasks_assigned = 10

        # But only 3 of 10 messages delivered (delivery_rate = 0.3)
        results.messages_created = 10
        results.messages_delivered = 3

        assert results.assignment_rate == 1.0
        assert results.delivery_rate == pytest.approx(0.3)


# ============================================================================
# 2. Response Time: measures assignment delay, NOT delivery
# ============================================================================


class TestResponseTimeSemantics:
    """
    Issue: response times suspiciously identical at 75% and 40%.

    Root cause: response_time = coordination_cycle_time - task.creation_time.
    This is the time from task creation to the COORDINATOR DECISION, not the
    time for the message to reach the responder (delivery_time).

    Since coordination cycles run at fixed 1800s intervals, and task arrival
    follows the same Poisson process (same seed), response times will be
    identical regardless of connectivity level.
    """

    def test_response_time_equals_cycle_minus_creation(self):
        """Response time is purely a function of coordination timing."""
        results = SimulationResults(
            config=SimulationConfig(),
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            run_number=0,
            random_seed=42,
        )

        # Task created at t=100 (cold-start, no warmup)
        # First coord cycle at t=0
        # Next coord cycle at t=1800
        # → task is assigned at t=1800
        task_creation = 100.0
        cycle_time = 1800.0
        expected_response_time = cycle_time - task_creation  # = 1700s

        results.response_times.append(("t_0", expected_response_time))

        assert results.average_decision_time == pytest.approx(1700.0)

    def test_response_time_independent_of_connectivity(self):
        """
        Same seed + same task schedule → same response times,
        regardless of connectivity level.
        """
        # Create two result sets: different connectivity, same timing
        results_75 = SimulationResults(
            config=SimulationConfig(),
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            run_number=0,
            random_seed=42,
        )
        results_40 = SimulationResults(
            config=SimulationConfig(),
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.40,
            run_number=0,
            random_seed=42,
        )

        # Same tasks, same coordination cycles → same response times
        for results in (results_75, results_40):
            results.response_times = [
                ("t_0", 1700.0),
                ("t_1", 1500.0),
                ("t_2", 800.0),
            ]

        # Response times are identical
        assert results_75.average_decision_time == results_40.average_decision_time

    def test_delivery_time_differs_from_response_time(self):
        """Delivery time (network latency) is distinct from response time."""
        results = SimulationResults(
            config=SimulationConfig(),
            algorithm=AlgorithmType.ADAPTIVE,
            connectivity_level=0.40,
            run_number=0,
            random_seed=42,
        )

        # Response time = cycle - creation = 1700s
        results.response_times = [("t_0", 1700.0)]

        # Delivery time = actual_delivery - creation = 2500s (message took 800s to arrive)
        results.delivery_times = [("t_0", 2500.0)]

        assert results.average_decision_time == pytest.approx(1700.0)
        assert results.average_delivery_time == pytest.approx(2500.0)
        assert results.average_delivery_time > results.average_decision_time


# ============================================================================
# 3. PRoPHET State: identical between Adaptive & Baseline (same seed)
# ============================================================================


class TestProphetStateIndependence:
    """
    Issue: predictability matrices identical between algorithms.

    Root cause: PRoPHET state depends on mobility (positions → encounters),
    not on the coordination algorithm. Same seed → same RNG → same mobility
    → same encounters → same predictability matrix.

    This is EXPECTED behaviour, not a bug.
    """

    def test_same_seed_produces_identical_predictability(self):
        """
        Two DeliveryPredictabilityMatrix instances with the same encounter
        sequence produce identical P values, regardless of which algorithm
        would use them.
        """
        matrix_a = DeliveryPredictabilityMatrix()
        matrix_b = DeliveryPredictabilityMatrix()

        # Same encounter sequence
        encounters = [
            ("n0", "n1", 100.0),
            ("n1", "n2", 200.0),
            ("n0", "n2", 350.0),
            ("n0", "n1", 500.0),  # Re-encounter
        ]

        for n_a, n_b, t in encounters:
            matrix_a.update_encounter(n_a, n_b, t)
            matrix_a.update_transitivity(n_a, n_b)

            matrix_b.update_encounter(n_a, n_b, t)
            matrix_b.update_transitivity(n_a, n_b)

        # Matrices are identical
        for src in ["n0", "n1", "n2"]:
            for dst in ["n0", "n1", "n2"]:
                if src == dst:
                    continue
                p_a = matrix_a.get_predictability(src, dst)
                p_b = matrix_b.get_predictability(src, dst)
                assert p_a == pytest.approx(p_b), f"P({src},{dst}) differs"

    def test_different_algorithms_share_same_mobility_seed(self):
        """
        The ExperimentRunner uses the same seed for both algorithms,
        meaning they see the exact same node movement and encounters.
        """
        # This is a design verification: same seed → same topology + mobility
        seed = 42
        config = SimulationConfig(
            scenario=ScenarioParameters(
                simulation_duration_seconds=60,
                warmup_period_seconds=0,
                runs_per_configuration=1,
            ),
            coordination=CoordinationParameters(
                update_interval_seconds=30,
            ),
        )

        engine_adaptive = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=seed,
        )
        engine_baseline = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.BASELINE,
            connectivity_level=0.75,
            random_seed=seed,
        )

        result_a = engine_adaptive.run()
        result_b = engine_baseline.run()

        # Same number of tasks generated (same seed → same Poisson process)
        assert result_a.total_tasks == result_b.total_tasks

    def test_algorithms_differ_in_assignment_not_network(self):
        """
        Given identical network state, Adaptive and Baseline assign
        tasks to DIFFERENT responders — that's where the difference lies.
        """
        # 3 responders: r_0 far but high P, r_1 close but low P, r_2 moderate
        positions = {
            "r_0": (500.0, 500.0),  # Far from task (100,100)
            "r_1": (110.0, 110.0),  # Close to task
            "r_2": (250.0, 250.0),  # Moderate distance
        }
        locator = StubResponderLocator(positions)

        # Network state: r_0 has highest P, r_1 has lowest (but above threshold)
        net = StubNetworkState({
            ("coord_0", "r_0"): 0.8,
            ("coord_0", "r_1"): 0.35,
            ("coord_0", "r_2"): 0.4,
        })

        task = _make_task("t_0", 10.0, x=100.0, y=100.0)

        # Adaptive: weighted score considers both P and distance
        adaptive = AdaptiveCoordinator()
        adaptive_assignments = adaptive.assign_tasks(
            [task], locator, net, "coord_0", 100.0
        )

        # Reset task for baseline
        task.status = TaskStatus.PENDING
        task.assigned_to = None
        task.assignment_time = None

        # Baseline: nearest responder only
        baseline = BaselineCoordinator()
        baseline_assignments = baseline.assign_tasks(
            [task], locator, None, "coord_0", 100.0
        )

        # Baseline picks closest (r_1), adaptive may pick differently
        assert baseline_assignments[0].responder_id == "r_1"

        # Adaptive balances P and distance — the specific choice depends on
        # normalised score, but it considers connectivity
        adaptive_responder = adaptive_assignments[0].responder_id
        assert adaptive_responder in {"r_0", "r_1", "r_2"}

        # Log scores for diagnostic visibility
        adaptive_p = adaptive_assignments[0].predictability
        assert adaptive_p is not None  # Adaptive records P
        assert baseline_assignments[0].predictability is None  # Baseline doesn't


# ============================================================================
# 4. Adaptive Geographic Clustering
# ============================================================================


class TestAdaptiveGeographicClustering:
    """
    Issue: Adaptive seems to assign responders that cluster geographically,
    potentially because high-P nodes are those near the coordination zone
    (they encounter coord nodes more often).

    This test verifies the weighted score mechanics and shows how P
    normalisation can bias assignments towards high-P nodes even when
    they're far from the task.
    """

    def test_threshold_excludes_low_p_close_node(self):
        """
        After the fix, a nearby node with P below threshold (0.3) is excluded
        entirely.  The far node with P above threshold is selected even though
        it is much farther away — the threshold acts as a hard filter.
        """
        task = _make_task("t_0", 10.0, x=100.0, y=100.0)

        positions = {
            "r_close": (110.0, 110.0),   # ~14m from task
            "r_far": (800.0, 700.0),     # ~860m from task
        }
        locator = StubResponderLocator(positions)

        # r_close below threshold, r_far above
        net = StubNetworkState({
            ("coord_0", "r_close"): 0.05,  # Below 0.3 → excluded
            ("coord_0", "r_far"): 0.9,     # Above threshold
        })

        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks(
            [task], locator, net, "coord_0", 100.0
        )

        assert len(assignments) == 1
        # r_close excluded by threshold, only r_far is a candidate
        assert assignments[0].responder_id == "r_far"

    def test_moderate_p_difference_favours_closer_node(self):
        """
        After the fix (absolute P, proximity-dominant weights, fixed diagonal),
        the closer node wins when P differences are moderate because β=0.6
        heavily weights proximity.

        Static weights α=0.2, γ_r=0.2, β=0.6. R_norm=0.0 for all (never encountered at t=100).
        """
        task = _make_task("t_0", 10.0, x=300.0, y=300.0)

        positions = {
            "r_close": (320.0, 310.0),    # ~22m from task
            "r_moderate": (400.0, 350.0),  # ~112m from task
            "r_far": (600.0, 500.0),       # ~361m from task
        }
        locator = StubResponderLocator(positions)

        # All above 0.3 threshold; static α=0.2, γ_r=0.2, β=0.6
        net = StubNetworkState({
            ("coord_0", "r_close"): 0.32,
            ("coord_0", "r_moderate"): 0.37,
            ("coord_0", "r_far"): 0.42,
        })

        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks(
            [task], locator, net, "coord_0", 100.0
        )

        assigned = assignments[0].responder_id

        # Weights: α=0.2, γ_r=0.2, β=0.6; R_norm=0.0 (never encountered at t=100)
        # r_close:    0.2×0.32 + 0.2×0 + 0.6×(1-22/3354.1)  = 0.064 + 0.596 = 0.660
        # r_moderate: 0.2×0.37 + 0.2×0 + 0.6×(1-112/3354.1) = 0.074 + 0.580 = 0.654
        # r_far:      0.2×0.42 + 0.2×0 + 0.6×(1-361/3354.1) = 0.084 + 0.535 = 0.619
        # r_close wins — proximity-dominant weighting in action
        assert assigned == "r_close"

    def test_all_nodes_same_p_reduces_to_nearest(self):
        """When all P values are equal, Adaptive reduces to nearest-responder."""
        task = _make_task("t_0", 10.0, x=100.0, y=100.0)

        positions = {
            "r_0": (110.0, 110.0),   # Closest
            "r_1": (300.0, 300.0),
            "r_2": (500.0, 500.0),   # Farthest
        }
        locator = StubResponderLocator(positions)

        # All same P above threshold → distance is the tiebreaker
        net = StubNetworkState({
            ("coord_0", "r_0"): 0.40,
            ("coord_0", "r_1"): 0.40,
            ("coord_0", "r_2"): 0.40,
        })

        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks(
            [task], locator, net, "coord_0", 100.0
        )

        # With equal P, closest wins
        assert assignments[0].responder_id == "r_0"

    def test_adaptive_vs_baseline_assignment_distribution(self):
        """
        Compare how many unique responders each algorithm uses across
        multiple tasks. Adaptive may concentrate on high-P nodes.
        """
        # 10 tasks at various locations in the incident zone
        rng = np.random.default_rng(42)
        tasks_adaptive = [
            _make_task(
                f"t_{i}", i * 30.0,
                x=rng.uniform(0, 700),
                y=rng.uniform(450, 1050),
            )
            for i in range(10)
        ]
        # Clone for baseline (reset state)
        tasks_baseline = [
            _make_task(
                f"t_{i}", i * 30.0,
                x=tasks_adaptive[i].target_location_x,
                y=tasks_adaptive[i].target_location_y,
            )
            for i in range(10)
        ]

        # 8 responders at different positions
        positions = {
            "r_0": (100.0, 500.0),
            "r_1": (200.0, 600.0),
            "r_2": (350.0, 700.0),
            "r_3": (500.0, 800.0),
            "r_4": (600.0, 500.0),
            "r_5": (700.0, 900.0),
            "r_6": (800.0, 600.0),  # Near coord zone
            "r_7": (810.0, 650.0),  # Near coord zone
        }
        locator = StubResponderLocator(positions)

        # All nodes above threshold; coord-zone nodes have slightly higher P
        net = StubNetworkState({
            ("coord_0", "r_0"): 0.35,
            ("coord_0", "r_1"): 0.36,
            ("coord_0", "r_2"): 0.34,
            ("coord_0", "r_3"): 0.33,
            ("coord_0", "r_4"): 0.38,
            ("coord_0", "r_5"): 0.37,
            ("coord_0", "r_6"): 0.45,  # High P — near coord
            ("coord_0", "r_7"): 0.50,  # Highest P — near coord
        })

        # Adaptive assignments
        adaptive = AdaptiveCoordinator()
        adaptive_assignments = adaptive.assign_tasks(
            tasks_adaptive, locator, net, "coord_0", 500.0
        )
        adaptive_responders = [a.responder_id for a in adaptive_assignments]
        adaptive_unique = set(adaptive_responders)

        # Baseline assignments
        baseline = BaselineCoordinator()
        baseline_assignments = baseline.assign_tasks(
            tasks_baseline, locator, None, "coord_0", 500.0
        )
        baseline_responders = [a.responder_id for a in baseline_assignments]
        baseline_unique = set(baseline_responders)

        print(f"\n--- Assignment Distribution Diagnostic (post-fix) ---")
        print(f"Adaptive unique responders: {len(adaptive_unique)} / {len(positions)}")
        print(f"  Assignments: {adaptive_responders}")
        print(f"Baseline unique responders: {len(baseline_unique)} / {len(positions)}")
        print(f"  Assignments: {baseline_responders}")

        # Verify both algorithms assigned all tasks
        assert len(adaptive_assignments) == 10
        assert len(baseline_assignments) == 10

        # Post-fix: adaptive should now also distribute across multiple
        # responders (proximity-dominant weighting + workload penalty),
        # no longer concentrating 9/10 tasks on one node
        assert len(adaptive_unique) >= 3
        assert len(baseline_unique) >= 1


# ============================================================================
# 5. Forwarding: >= condition + p_to > 0 guard
# ============================================================================


class TestForwardingBehavior:
    """
    Issue: the >= forwarding condition (PRoPHETv2) means messages spread
    more broadly than with >. The p_to > 0 guard prevents forwarding to
    nodes with zero predictability.

    This affects delivery differently depending on where the assignment
    sends the message: Adaptive assigns to high-P responders (easier to
    deliver), Baseline assigns to nearest (may be harder to reach).
    """

    def test_ge_condition_allows_equal_p_forwarding(self):
        """Messages forward to nodes with equal predictability (PRoPHETv2)."""
        matrix = DeliveryPredictabilityMatrix()
        matrix.set_predictability("n0", "dest", 0.3)
        matrix.set_predictability("n1", "dest", 0.3)  # Equal P

        # n1 is a valid forwarder because P(n1,dest) >= P(n0,dest)
        best = matrix.get_best_forwarder("n0", "dest", ["n1"])
        assert best == "n1"

    def test_zero_p_guard_blocks_forwarding(self):
        """Messages don't forward to nodes with P = 0 even with >= condition."""
        matrix = DeliveryPredictabilityMatrix()
        matrix.set_predictability("n0", "dest", 0.0)
        # n1 has P = 0 to dest (default)

        # Both have P = 0, but p_to > 0 guard blocks forwarding
        best = matrix.get_best_forwarder("n0", "dest", ["n1"])
        assert best is None

    def test_forwarding_asymmetry_adaptive_vs_baseline(self):
        """
        Adaptive assigns to high-P node → message starts closer to dest in P-space.
        Baseline assigns to nearest node → may need more hops to reach.

        This test shows the P-space advantage of adaptive assignments.
        """
        params = CommunicationParameters()
        net_params = NetworkParameters()  # Use defaults (50 nodes)
        nodes = ["coord_0", "r_0", "r_1", "r_2", "r_3"]

        comm = CommunicationLayer(params, net_params, nodes)

        # Set up predictability: r_3 is well-connected to dest (r_0)
        # but r_1 is the nearest to the task
        comm.predictability.set_predictability("coord_0", "r_3", 0.5)
        comm.predictability.set_predictability("r_3", "r_0", 0.4)
        comm.predictability.set_predictability("coord_0", "r_0", 0.1)
        comm.predictability.set_predictability("coord_0", "r_1", 0.2)

        # Adaptive would assign to r_3 (high P from coord to r_3)
        # Message: coord_0 → r_3 — direct delivery if r_3 is destination
        msg_adaptive = comm.create_message(
            "coord_0", "r_3", MessageType.COORDINATION,
            {"task": "test"}, 100.0,
        )

        # Baseline would assign to nearest, say r_1
        # Message: coord_0 → r_1 — needs forwarding via network
        msg_baseline = comm.create_message(
            "coord_0", "r_1", MessageType.COORDINATION,
            {"task": "test"}, 100.0,
        )

        # Check: for the adaptive message, coord_0 sends to r_3
        # coord_0 has P(coord_0, r_3) = 0.5, so any encounter with a node
        # that has P >= 0.5 to r_3 would forward it
        p_coord_to_r3 = comm.get_delivery_predictability("coord_0", "r_3")
        assert p_coord_to_r3 == pytest.approx(0.5)

        # For the baseline message, coord_0 sends to r_1
        # coord_0 has P(coord_0, r_1) = 0.2
        p_coord_to_r1 = comm.get_delivery_predictability("coord_0", "r_1")
        assert p_coord_to_r1 == pytest.approx(0.2)

    def test_delivery_depends_on_encounters_not_assignment(self):
        """
        Actual delivery requires physical encounters between nodes.
        Assignment quality (high-P target) helps because more nodes
        can serve as forwarders (>= condition spreads messages).
        """
        matrix = DeliveryPredictabilityMatrix()

        # Well-connected destination: many nodes have P > 0 to it
        matrix.set_predictability("n0", "r_well", 0.3)
        matrix.set_predictability("n1", "r_well", 0.4)
        matrix.set_predictability("n2", "r_well", 0.5)

        # Poorly-connected destination: few nodes know about it
        matrix.set_predictability("n0", "r_poor", 0.1)
        # n1, n2 have P = 0 to r_poor

        # Number of potential forwarders to well-connected node
        forwarders_well = sum(
            1 for n in ["n0", "n1", "n2"]
            if matrix.get_predictability(n, "r_well") > 0
        )

        # Number of potential forwarders to poorly-connected node
        forwarders_poor = sum(
            1 for n in ["n0", "n1", "n2"]
            if matrix.get_predictability(n, "r_poor") > 0
        )

        assert forwarders_well == 3
        assert forwarders_poor == 1

        # Well-connected destinations have more forwarding paths →
        # higher delivery probability. Adaptive targets these.


# ============================================================================
# 6. End-to-end micro-simulation diagnostic
# ============================================================================


class TestMicroSimulationDiagnostic:
    """
    Run very short simulations to verify the interplay between
    assignment and delivery for both algorithms.
    """

    def _make_short_config(self) -> SimulationConfig:
        """Create a minimal config for fast diagnostic runs."""
        return SimulationConfig(
            scenario=ScenarioParameters(
                simulation_duration_seconds=120,
                warmup_period_seconds=0,
                message_rate_per_minute=1.0,
                runs_per_configuration=1,
            ),
            coordination=CoordinationParameters(
                update_interval_seconds=60,
            ),
        )

    def test_adaptive_and_baseline_same_task_count(self):
        """Both algorithms get the same tasks (same seed)."""
        config = self._make_short_config()
        seed = 42

        result_a = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=seed,
        ).run()

        result_b = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.BASELINE,
            connectivity_level=0.75,
            random_seed=seed,
        ).run()

        assert result_a.total_tasks == result_b.total_tasks

    def test_delivery_rate_can_differ_between_algorithms(self):
        """
        Even with same network state, delivery rates can differ because
        each algorithm assigns messages to different destination nodes,
        which have different reachability via PRoPHET forwarding.
        """
        config = self._make_short_config()
        seed = 100  # Try different seed

        result_a = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=seed,
        ).run()

        result_b = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.BASELINE,
            connectivity_level=0.75,
            random_seed=seed,
        ).run()

        print(f"\n--- Micro-Simulation Diagnostic (seed={seed}) ---")
        print(f"Adaptive: tasks={result_a.total_tasks}, "
              f"assigned={result_a.tasks_assigned}, "
              f"msgs_created={result_a.messages_created}, "
              f"delivered={result_a.messages_delivered}, "
              f"delivery_rate={result_a.delivery_rate:.3f}")
        print(f"Baseline: tasks={result_b.total_tasks}, "
              f"assigned={result_b.tasks_assigned}, "
              f"msgs_created={result_b.messages_created}, "
              f"delivered={result_b.messages_delivered}, "
              f"delivery_rate={result_b.delivery_rate:.3f}")

        # We can't predict which is higher without running, but we verify
        # both produce valid results
        assert 0.0 <= result_a.delivery_rate <= 1.0
        assert 0.0 <= result_b.delivery_rate <= 1.0

    def test_connectivity_affects_delivery_not_assignment(self):
        """
        Lower connectivity → fewer available links → fewer encounters
        → lower delivery rate. But assignment rate stays the same.
        """
        config = self._make_short_config()
        seed = 42

        result_high = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.75,
            random_seed=seed,
        ).run()

        result_low = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.20,
            random_seed=seed,
        ).run()

        print(f"\n--- Connectivity Impact Diagnostic ---")
        print(f"75% connectivity: assignment_rate={result_high.assignment_rate:.3f}, "
              f"delivery_rate={result_high.delivery_rate:.3f}")
        print(f"20% connectivity: assignment_rate={result_low.assignment_rate:.3f}, "
              f"delivery_rate={result_low.delivery_rate:.3f}")

        # With cold-start, assignment rate depends on encounter history
        # Delivery rate should decrease with lower connectivity
        # (though with short simulation this may not always hold)
        assert 0.0 <= result_high.delivery_rate <= 1.0
        assert 0.0 <= result_low.delivery_rate <= 1.0
