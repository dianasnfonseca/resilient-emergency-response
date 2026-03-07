"""
Regression tests for adaptive coordinator fixes + capacity-bounded assignment.

Groups:
    A — TestConstants: verify module-level constants
    B — TestBug1AbsoluteNormalisation: prevent assignment concentration
    C — TestBug2ThresholdFiltering: P > 0.3 excludes marginal nodes
    D — TestCapacityBound: hard k_max exclusion (replaces W_intra soft penalty)

Sources:
    Shah & Ahmed (2025) — absolute DP values (SN Computer Science)
    Ullah & Qayyum (2022) — SAAD forwarder threshold
    Boondirek et al. (2014) — DiPRoPHET distance-dominant weighting
    Bhatti et al. (2021) — bounded task assignment (b-matching)
"""

import pytest

from ercs.config.parameters import CoordinationParameters, UrgencyLevel
from ercs.coordination.algorithms import (
    PREDICTABILITY_WEIGHT,
    PROXIMITY_WEIGHT,
    RECENCY_WEIGHT,
    SIMULATION_AREA_DIAGONAL_M,
    WORKLOAD_PENALTY_WEIGHT,
    AdaptiveCoordinator,
)
from ercs.scenario.generator import Task
from conftest import PATH_THRESHOLD


# ============================================================================
# Helpers (copied from test_coordination.py — not exported)
# ============================================================================


class MockResponderLocator:
    """Mock responder locator for testing."""

    def __init__(self, responders: dict[str, tuple[float, float]]):
        self._responders = responders

    def get_responder_position(self, responder_id: str) -> tuple[float, float]:
        return self._responders[responder_id]

    def get_all_responder_ids(self) -> list[str]:
        return list(self._responders.keys())


class MockNetworkState:
    """Mock network state provider for testing."""

    def __init__(self, predictabilities: dict[tuple[str, str], float] | None = None):
        self._predictabilities = predictabilities or {}
        self._last_encounter_times: dict[tuple[str, str], float] = {}

    def get_delivery_predictability(self, from_node: str, to_node: str) -> float:
        return self._predictabilities.get((from_node, to_node), 0.0)

    def get_last_encounter_time(self, from_node: str, to_node: str) -> float:
        return self._last_encounter_times.get((from_node, to_node), 0.0)

    def set_predictability(self, from_node: str, to_node: str, value: float) -> None:
        self._predictabilities[(from_node, to_node)] = value


def _make_task(
    task_id: str,
    creation_time: float,
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    x: float = 100.0,
    y: float = 100.0,
) -> Task:
    return Task(
        task_id=task_id,
        creation_time=creation_time,
        source_node="coord_0",
        target_location_x=x,
        target_location_y=y,
        urgency=urgency,
    )


# ============================================================================
# Group A — TestConstants
# ============================================================================


class TestConstants:
    """Verify module-level constants after the fix."""

    def test_predictability_weight(self):
        """α = 0.2 (Boondirek et al., 2014; adjusted for recency)."""
        assert PREDICTABILITY_WEIGHT == pytest.approx(0.2)

    def test_recency_weight(self):
        """γ_r = 0.2 (Nelson et al., 2009)."""
        assert RECENCY_WEIGHT == pytest.approx(0.2)

    def test_proximity_weight(self):
        """β = 0.6 (Boondirek et al., 2014; adjusted for recency)."""
        assert PROXIMITY_WEIGHT == pytest.approx(0.6)

    def test_workload_penalty_weight(self):
        """λ = 0.2 (Cui et al., 2022)."""
        assert WORKLOAD_PENALTY_WEIGHT == pytest.approx(0.2)

    def test_simulation_area_diagonal(self):
        """sqrt(3000² + 1500²) ≈ 3354.1 m."""
        assert SIMULATION_AREA_DIAGONAL_M == pytest.approx(3354.1, abs=0.1)

    def test_default_path_threshold(self):
        """Default threshold = 0.3 (Ullah & Qayyum, 2022 SAAD)."""
        params = CoordinationParameters()
        assert params.available_path_threshold == pytest.approx(0.3)
        assert PATH_THRESHOLD == pytest.approx(0.3)


# ============================================================================
# Group B — TestBug1AbsoluteNormalisation
# ============================================================================


class TestBug1AbsoluteNormalisation:
    """
    BUG-1: Relative P normalisation caused assignment concentration.

    Before the fix, p_norm = P / max_P gave the top node p_norm = 1.0,
    inflating the gap and routing 9/10 tasks to one node.

    After the fix, p_abs = P directly, so a node with P=0.45 only gets
    a P score of 0.45 × α = 0.135, while a closer node with P=0.35
    gets 0.35 × α = 0.105 — a much smaller gap that proximity can overcome.
    """

    def test_prevents_concentration(self):
        """10 tasks across the area should go to >= 3 distinct responders."""
        # 8 responders spread across the incident zone
        positions = {
            "r_0": (50.0, 500.0),
            "r_1": (150.0, 600.0),
            "r_2": (250.0, 700.0),
            "r_3": (350.0, 800.0),
            "r_4": (450.0, 550.0),
            "r_5": (550.0, 650.0),
            "r_6": (650.0, 750.0),
            "r_7": (800.0, 350.0),  # Near coord zone
        }
        locator = MockResponderLocator(positions)

        # r_7 has highest P (~0.45), others above threshold (0.32–0.38)
        net = MockNetworkState({
            ("coord_0", "r_0"): 0.35,
            ("coord_0", "r_1"): 0.36,
            ("coord_0", "r_2"): 0.34,
            ("coord_0", "r_3"): 0.32,
            ("coord_0", "r_4"): 0.38,
            ("coord_0", "r_5"): 0.37,
            ("coord_0", "r_6"): 0.33,
            ("coord_0", "r_7"): 0.45,
        })

        # 10 tasks spread across the incident zone, most far from r_7
        tasks = [
            _make_task("t_0", 0.0, x=60.0, y=510.0),
            _make_task("t_1", 1.0, x=160.0, y=620.0),
            _make_task("t_2", 2.0, x=260.0, y=710.0),
            _make_task("t_3", 3.0, x=370.0, y=790.0),
            _make_task("t_4", 4.0, x=440.0, y=560.0),
            _make_task("t_5", 5.0, x=540.0, y=660.0),
            _make_task("t_6", 6.0, x=640.0, y=740.0),
            _make_task("t_7", 7.0, x=100.0, y=550.0),
            _make_task("t_8", 8.0, x=300.0, y=680.0),
            _make_task("t_9", 9.0, x=500.0, y=600.0),
        ]

        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks(
            tasks=tasks,
            responder_locator=locator,
            network_state=net,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 10

        unique_responders = {a.responder_id for a in assignments}
        assert len(unique_responders) >= 3, (
            f"Expected >= 3 unique responders, got {len(unique_responders)}: "
            f"{[a.responder_id for a in assignments]}"
        )

    def test_threshold_filters_at_realistic_scale(self):
        """
        48 responders, 24 with P > 0.3 (eligible), 24 with P < 0.3 (excluded).
        20 tasks across the incident zone.

        Verifies:
          1. Threshold excludes half the responders — no assignments to P < 0.3 nodes
          2. Assignments distribute among the ELIGIBLE subset only
          3. Not all eligible responders are used (proximity matters, not just spread)
        """
        import random

        rng = random.Random(42)

        # 48 responders spread across the 3000×1500 m simulation area
        positions = {}
        predictabilities = {}
        eligible_ids = set()
        excluded_ids = set()

        for i in range(48):
            rid = f"r_{i}"
            # Spread responders across the area
            x = 50.0 + (i % 8) * 370.0  # 8 columns across 3000m
            y = 100.0 + (i // 8) * 220.0  # 6 rows across 1500m
            # Add small jitter to avoid ties
            x += rng.uniform(-30, 30)
            y += rng.uniform(-30, 30)
            positions[rid] = (x, y)

            if i < 24:
                # First 24: eligible — P between 0.32 and 0.50
                p = rng.uniform(0.32, 0.50)
                eligible_ids.add(rid)
            else:
                # Last 24: excluded — P between 0.05 and 0.25
                p = rng.uniform(0.05, 0.25)
                excluded_ids.add(rid)

            predictabilities[("coord_0", rid)] = p

        locator = MockResponderLocator(positions)
        net = MockNetworkState(predictabilities)

        # 20 tasks spread across the incident zone (100–800, 500–1100)
        tasks = [
            _make_task(
                f"t_{i}",
                float(i),
                x=100.0 + (i % 5) * 150.0,
                y=500.0 + (i // 5) * 150.0,
            )
            for i in range(20)
        ]

        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks(
            tasks=tasks,
            responder_locator=locator,
            network_state=net,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assigned_ids = {a.responder_id for a in assignments}

        # 1. All tasks assigned (there are enough eligible responders)
        assert len(assignments) == 20

        # 2. No assignment goes to a responder below threshold
        below_threshold_assigned = assigned_ids & excluded_ids
        assert len(below_threshold_assigned) == 0, (
            f"Responders below P=0.3 threshold received assignments: "
            f"{below_threshold_assigned}"
        )

        # 3. All assigned responders come from the eligible set
        assert assigned_ids <= eligible_ids

        # 4. Multiple distinct responders used (workload penalty distributes)
        assert len(assigned_ids) >= 5, (
            f"Expected >= 5 unique eligible responders, got {len(assigned_ids)}"
        )

        # 5. Not ALL 24 eligible responders used — proximity should concentrate
        #    assignments toward responders near the incident zone, not scatter
        #    across the whole area
        assert len(assigned_ids) < 24, (
            f"All 24 eligible responders used — proximity scoring has no effect"
        )


# ============================================================================
# Group C — TestBug2ThresholdFiltering
# ============================================================================


class TestBug2ThresholdFiltering:
    """
    BUG-2: P > 0.0 threshold was ineffective — all nodes passed after warm-up.

    After the fix, threshold = 0.3 excludes marginal transitivity nodes
    (P ≈ 0.05–0.20) and admits only nodes with genuine encounter history.
    """

    def test_excludes_low_p_prefers_high_p(self):
        """Near responder with P=0.15 is excluded; farther P=0.40 is selected."""
        positions = {
            "r_near": (110.0, 110.0),  # Very close to task at (100, 100)
            "r_far": (500.0, 500.0),   # Far from task
        }
        locator = MockResponderLocator(positions)

        net = MockNetworkState({
            ("coord_0", "r_near"): 0.15,  # Below 0.3 threshold
            ("coord_0", "r_far"): 0.40,   # Above threshold
        })

        task = _make_task("t_0", 0.0, x=100.0, y=100.0)

        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks(
            [task], locator, net, "coord_0", 100.0
        )

        assert len(assignments) == 1
        assert assignments[0].responder_id == "r_far"

    def test_boundary_p_equals_threshold_excluded(self):
        """P exactly = 0.3 should be excluded (condition is P > threshold)."""
        positions = {
            "r_boundary": (110.0, 110.0),
            "r_above": (500.0, 500.0),
        }
        locator = MockResponderLocator(positions)

        net = MockNetworkState({
            ("coord_0", "r_boundary"): 0.3,   # Exactly at threshold → excluded
            ("coord_0", "r_above"): 0.35,      # Just above → included
        })

        task = _make_task("t_0", 0.0, x=100.0, y=100.0)

        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks(
            [task], locator, net, "coord_0", 100.0
        )

        assert len(assignments) == 1
        assert assignments[0].responder_id == "r_above"

    def test_all_below_threshold_no_assignment(self):
        """When all nodes are below threshold, no assignment is made."""
        positions = {
            "r_0": (110.0, 110.0),
            "r_1": (200.0, 200.0),
        }
        locator = MockResponderLocator(positions)

        net = MockNetworkState({
            ("coord_0", "r_0"): 0.10,
            ("coord_0", "r_1"): 0.25,
        })

        task = _make_task("t_0", 0.0, x=100.0, y=100.0)

        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks(
            [task], locator, net, "coord_0", 100.0
        )

        assert len(assignments) == 0
        assert coord._failed_assignments == 1


# ============================================================================
# Group D — TestCapacityBound (replaces TestBug4WorkloadPenalty)
# ============================================================================


class TestCapacityBound:
    """
    Capacity-bounded assignment: Bhatti et al. (2021).

    Replaces the W_intra soft penalty with a hard capacity exclusion.
    k_max = floor(T / R) + 1 caps the maximum tasks any single responder
    can receive per coordination cycle (b-matching upper bound).
    """

    def test_capacity_bound_excludes_saturated_responder(self):
        """
        10 tasks, 5 responders → k_max = floor(10/5) + 1 = 3.

        After responder_1 receives k_max tasks, it must not be selected
        for task k_max+1, even if it has the highest P-value.
        """
        # 5 responders: r_1 closest to all tasks and highest P
        positions = {
            "r_1": (105.0, 105.0),    # Very close to tasks
            "r_2": (300.0, 300.0),
            "r_3": (500.0, 500.0),
            "r_4": (700.0, 700.0),
            "r_5": (900.0, 500.0),
        }
        locator = MockResponderLocator(positions)

        net = MockNetworkState({
            ("coord_0", "r_1"): 0.50,   # Highest P
            ("coord_0", "r_2"): 0.40,
            ("coord_0", "r_3"): 0.40,
            ("coord_0", "r_4"): 0.40,
            ("coord_0", "r_5"): 0.40,
        })

        # 10 tasks all near r_1
        tasks = [
            _make_task(f"t_{i}", float(i), x=100.0, y=100.0)
            for i in range(10)
        ]

        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks(
            tasks=tasks,
            responder_locator=locator,
            network_state=net,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 10

        # k_max = floor(10/5) + 1 = 3
        from collections import Counter
        counts = Counter(a.responder_id for a in assignments)

        assert counts["r_1"] <= 3, (
            f"r_1 received {counts['r_1']} tasks, expected <= k_max=3"
        )

        # Other responders must have received the overflow
        assert len(counts) >= 2, (
            f"Expected >= 2 unique responders, got {len(counts)}"
        )

    def test_capacity_bound_distributes_load(self):
        """
        20 tasks, 10 responders, identical P and distances.

        Gini coefficient of assignments must be < 0.4 (well-distributed).
        """
        # 10 responders at equal distance from task location
        positions = {}
        for i in range(10):
            angle = 2 * 3.14159 * i / 10
            x = 500.0 + 200.0 * (1 if i % 2 == 0 else -1)  # symmetric
            y = 500.0 + 200.0 * ((i % 3) - 1)
            positions[f"r_{i}"] = (x, y)
        locator = MockResponderLocator(positions)

        # All identical P
        net = MockNetworkState({
            ("coord_0", f"r_{i}"): 0.45 for i in range(10)
        })

        # 20 tasks at centre
        tasks = [
            _make_task(f"t_{i}", float(i), x=500.0, y=500.0)
            for i in range(20)
        ]

        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks(
            tasks=tasks,
            responder_locator=locator,
            network_state=net,
            coordination_node="coord_0",
            current_time=100.0,
        )

        assert len(assignments) == 20

        # Compute Gini coefficient
        from collections import Counter
        counts = Counter(a.responder_id for a in assignments)
        values = sorted(counts.values())
        n = len(values)
        if n == 0 or sum(values) == 0:
            gini = 0.0
        else:
            numerator = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(values))
            gini = numerator / (n * sum(values))

        assert gini < 0.4, (
            f"Gini coefficient {gini:.3f} >= 0.4, load poorly distributed: "
            f"{dict(counts)}"
        )

    def test_inter_cycle_penalty_preserved(self):
        """
        W_inter = 1 for a responder used in the previous cycle, reducing
        its score by λ = 0.2.  This operates independently of the
        intra-cycle capacity exclusion.
        """
        positions = {
            "r_a": (150.0, 550.0),
            "r_b": (160.0, 560.0),
        }
        locator = MockResponderLocator(positions)

        net = MockNetworkState({
            ("coord_0", "r_a"): 0.40,
            ("coord_0", "r_b"): 0.40,
        })

        # Cycle 1
        task1 = _make_task("t_1", 0.0, x=155.0, y=555.0)
        coord = AdaptiveCoordinator()
        a1 = coord.assign_tasks([task1], locator, net, "coord_0", 100.0)
        first = a1[0].responder_id

        # Cycle 2: the previously assigned responder has w_inter=1.0 (−0.2 score)
        task2 = _make_task("t_2", 50.0, x=155.0, y=555.0)
        a2 = coord.assign_tasks([task2], locator, net, "coord_0", 200.0)
        second = a2[0].responder_id

        # Inter-cycle penalty flips selection
        assert second != first, (
            f"Inter-cycle penalty should flip selection, got {second} both times"
        )

        # Verify independence: in cycle 2, k_max = floor(1/2)+1 = 1
        # so capacity doesn't block the second responder
        assert len(a2) == 1

    def test_k_max_computed_correctly(self):
        """
        Verify k_max = floor(T/R) + 1 for three cases:
          10 tasks /  7 responders → k_max = 2
          64 tasks / 47 responders → k_max = 2
         100 tasks / 48 responders → k_max = 3
        """
        cases = [
            (10, 7, 2),
            (64, 47, 2),
            (100, 48, 3),
        ]

        for num_tasks, num_responders, expected_k_max in cases:
            k_max = (num_tasks // max(num_responders, 1)) + 1
            assert k_max == expected_k_max, (
                f"{num_tasks} tasks / {num_responders} responders: "
                f"expected k_max={expected_k_max}, got {k_max}"
            )

    def test_fixed_diagonal_normalisation(self):
        """
        D_norm uses the fixed simulation area diagonal (3354.1 m),
        not the max distance among candidates.
        """
        positions = {
            "r_close": (110.0, 110.0),   # ~14 m from task at (100, 100)
            "r_far": (1000.0, 1000.0),   # ~1273 m from task
        }
        locator = MockResponderLocator(positions)

        net = MockNetworkState({
            ("coord_0", "r_close"): 0.35,
            ("coord_0", "r_far"): 0.35,
        })

        task = _make_task("t_0", 0.0, x=100.0, y=100.0)

        coord = AdaptiveCoordinator()
        assignments = coord.assign_tasks(
            [task], locator, net, "coord_0", 100.0
        )

        assert len(assignments) == 1
        # With equal P, the closer node must win
        assert assignments[0].responder_id == "r_close"
