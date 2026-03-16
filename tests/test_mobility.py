"""
Tests for Role-Based Random Waypoint Mobility Model.

These tests verify that the mobility system correctly implements
the Random Waypoint model with role-based zone constraints:
- Role assignment: 60% RESCUE, 25% TRANSPORT, 15% LIAISON
- Zone constraints per role (incident / shuttle / full)
- Speed ranges per role
- Backward compatibility when role is None
- Dynamic edge updates based on proximity (100m range)
- PRoPHET encounter triggering
"""

import numpy as np
import pytest
from conftest import RADIO_RANGE_M, SIMULATION_DURATION_S, SPEED_MAX_MPS

from ercs.config.parameters import NetworkParameters, ResponderRole
from ercs.network.mobility import (
    MobileNodeState,
    MobilityManager,
    MobilityState,
    Waypoint,
    _assign_roles,
    _build_role_configs,
    calculate_encounters,
)

# Build role configs from default parameters (single source of truth)
ROLE_CONFIGS = _build_role_configs(NetworkParameters())


class TestWaypoint:
    """Tests for Waypoint dataclass."""

    def test_waypoint_creation(self):
        """Test basic waypoint creation."""
        wp = Waypoint(x=100.0, y=200.0, speed=10.0, pause_duration=5.0)

        assert wp.x == 100.0
        assert wp.y == 200.0
        assert wp.speed == 10.0
        assert wp.pause_duration == 5.0

    def test_waypoint_default_pause(self):
        """Test waypoint with default pause duration."""
        wp = Waypoint(x=100.0, y=200.0, speed=10.0)

        assert wp.pause_duration == 0.0


class TestMobileNodeState:
    """Tests for MobileNodeState dataclass."""

    def test_state_creation(self):
        """Test basic state creation."""
        state = MobileNodeState(
            node_id="mobile_0",
            x=50.0,
            y=100.0,
        )

        assert state.node_id == "mobile_0"
        assert state.x == 50.0
        assert state.y == 100.0
        assert state.state == MobilityState.PAUSED
        assert state.waypoint is None

    def test_distance_to_waypoint_no_waypoint(self):
        """Test distance calculation when no waypoint set."""
        state = MobileNodeState(node_id="test", x=0.0, y=0.0)

        assert state.distance_to_waypoint() == 0.0

    def test_distance_to_waypoint(self):
        """Test distance calculation to waypoint."""
        state = MobileNodeState(
            node_id="test",
            x=0.0,
            y=0.0,
            waypoint=Waypoint(x=3.0, y=4.0, speed=10.0),
        )

        # 3-4-5 triangle
        assert state.distance_to_waypoint() == 5.0


class TestMobilityManager:
    """Tests for MobilityManager class."""

    @pytest.fixture
    def params(self) -> NetworkParameters:
        """Default network parameters."""
        return NetworkParameters()

    @pytest.fixture
    def manager(self, params: NetworkParameters) -> MobilityManager:
        """Create mobility manager with default params."""
        return MobilityManager(parameters=params)

    @pytest.fixture
    def initialized_manager(self, manager: MobilityManager) -> MobilityManager:
        """Create and initialize mobility manager."""
        mobile_ids = [f"mobile_{i}" for i in range(5)]
        initial_positions = {
            "mobile_0": (100.0, 500.0),
            "mobile_1": (200.0, 500.0),
            "mobile_2": (300.0, 500.0),
            "mobile_3": (400.0, 500.0),
            "mobile_4": (500.0, 500.0),
        }

        manager.initialize(
            mobile_node_ids=mobile_ids,
            initial_positions=initial_positions,
            random_seed=42,
        )

        return manager

    def test_manager_creation(self, params: NetworkParameters):
        """Test manager creation with parameters."""
        manager = MobilityManager(
            parameters=params,
            speed_min=0.0,
            speed_max=SPEED_MAX_MPS,
        )

        assert manager.speed_min == 0.0
        assert manager.speed_max == SPEED_MAX_MPS

    def test_initialize(self, manager: MobilityManager):
        """Test initialization with mobile nodes."""
        mobile_ids = ["mobile_0", "mobile_1"]
        positions = {
            "mobile_0": (100.0, 200.0),
            "mobile_1": (300.0, 400.0),
        }

        manager.initialize(mobile_ids, positions, random_seed=42)

        assert len(manager.get_node_ids()) == 2
        assert manager.get_position("mobile_0") == (100.0, 200.0)
        assert manager.get_position("mobile_1") == (300.0, 400.0)

    def test_step_moves_nodes(self, initialized_manager: MobilityManager):
        """Test that stepping moves nodes."""
        initial_positions = initialized_manager.get_all_positions()

        # Step multiple times
        for i in range(10):
            initialized_manager.step(current_time=float(i), delta_time=1.0)

        final_positions = initialized_manager.get_all_positions()

        # At least some nodes should have moved
        moved = False
        for node_id in initial_positions:
            if initial_positions[node_id] != final_positions[node_id]:
                moved = True
                break

        assert moved, "No nodes moved after 10 steps"

    def test_speed_within_bounds(self, initialized_manager: MobilityManager):
        """Test that assigned speeds are within bounds."""
        # Access internal state to check waypoint speeds
        for state in initialized_manager._node_states.values():
            if state.waypoint is not None:
                assert 1.0 <= state.waypoint.speed <= SPEED_MAX_MPS

    def test_reproducibility_with_seed(self, params: NetworkParameters):
        """Test that same seed produces same movements."""
        mobile_ids = ["mobile_0"]
        positions = {"mobile_0": (100.0, 100.0)}

        manager1 = MobilityManager(parameters=params)
        manager1.initialize(mobile_ids, positions, random_seed=42)

        manager2 = MobilityManager(parameters=params)
        manager2.initialize(mobile_ids, positions, random_seed=42)

        # Step both
        for i in range(10):
            manager1.step(float(i), 1.0)
            manager2.step(float(i), 1.0)

        pos1 = manager1.get_position("mobile_0")
        pos2 = manager2.get_position("mobile_0")

        assert pos1 == pos2

    def test_positions_within_simulation_area(
        self, initialized_manager: MobilityManager
    ):
        """Test that nodes stay within simulation area."""
        params = initialized_manager.parameters

        # Run many steps
        for i in range(100):
            initialized_manager.step(float(i), 1.0)

        positions = initialized_manager.get_all_positions()

        for _node_id, (x, y) in positions.items():
            assert 0 <= x <= params.simulation_area.width_m
            assert 0 <= y <= params.simulation_area.height_m


class TestCalculateEncounters:
    """Tests for calculate_encounters function."""

    def test_no_encounters_when_far_apart(self):
        """Test no encounters when nodes are far apart."""
        positions = {
            "node_a": (0.0, 0.0),
            "node_b": (1000.0, 1000.0),  # Far apart
        }

        encounters = calculate_encounters(positions, radio_range=RADIO_RANGE_M)

        assert len(encounters) == 0

    def test_encounter_when_within_range(self):
        """Test encounter detected when within range."""
        positions = {
            "node_a": (0.0, 0.0),
            "node_b": (50.0, 0.0),  # 50m apart, within 100m range
        }

        encounters = calculate_encounters(positions, radio_range=RADIO_RANGE_M)

        assert len(encounters) == 1
        assert encounters[0][0] == "node_a"
        assert encounters[0][1] == "node_b"
        assert encounters[0][2] == 50.0

    def test_encounter_at_exact_range(self):
        """Test encounter at exactly radio range."""
        positions = {
            "node_a": (0.0, 0.0),
            "node_b": (100.0, 0.0),  # Exactly 100m
        }

        encounters = calculate_encounters(positions, radio_range=RADIO_RANGE_M)

        assert len(encounters) == 1

    def test_multiple_encounters(self):
        """Test multiple encounters detected."""
        positions = {
            "node_a": (0.0, 0.0),
            "node_b": (50.0, 0.0),  # In range of A
            "node_c": (80.0, 0.0),  # In range of A and B
        }

        encounters = calculate_encounters(positions, radio_range=RADIO_RANGE_M)

        # Should have A-B, A-C, B-C
        assert len(encounters) == 3


class TestMobilityIntegration:
    """Integration tests for mobility with network topology."""

    def test_nodes_can_reach_coordination_zone(self):
        """Test that mobile nodes can eventually reach coordination zone."""
        params = NetworkParameters()
        manager = MobilityManager(
            parameters=params,
            speed_min=5.0,  # Use higher minimum speed for faster test
            speed_max=SPEED_MAX_MPS,
        )

        # Start node far from coordination zone
        mobile_ids = ["mobile_0"]
        positions = {"mobile_0": (100.0, 500.0)}  # In incident zone

        manager.initialize(mobile_ids, positions, random_seed=42)

        coord_zone = params.coordination_zone
        coord_center_x = coord_zone.origin_x + coord_zone.width_m / 2
        coord_center_y = coord_zone.origin_y + coord_zone.height_m / 2

        # Run for simulated time
        min_distance_seen = float("inf")

        for i in range(SIMULATION_DURATION_S):  # 6000 seconds
            manager.step(float(i), 1.0)
            pos = manager.get_position("mobile_0")

            # Check distance to coordination zone center
            dx = pos[0] - coord_center_x
            dy = pos[1] - coord_center_y
            distance = np.sqrt(dx * dx + dy * dy)

            min_distance_seen = min(min_distance_seen, distance)

            # If within range of coordination zone, success
            if distance < RADIO_RANGE_M:
                break

        # Node should have gotten reasonably close to coordination zone
        # at some point during the simulation
        assert (
            min_distance_seen < 500.0
        ), f"Node never got close to coordination zone, min distance: {min_distance_seen}"

    def test_dynamic_encounter_creation(self):
        """Test that encounters are created as nodes move together."""
        params = NetworkParameters()
        manager = MobilityManager(parameters=params)

        # Start nodes far apart
        mobile_ids = ["mobile_0", "mobile_1"]
        positions = {
            "mobile_0": (0.0, 0.0),
            "mobile_1": (500.0, 500.0),  # Far apart
        }

        manager.initialize(mobile_ids, positions, random_seed=42)

        # Track if we ever see an encounter
        encountered = False

        for i in range(SIMULATION_DURATION_S):
            manager.step(float(i), 1.0)

            current_positions = manager.get_all_positions()
            encounters = calculate_encounters(
                current_positions, radio_range=RADIO_RANGE_M
            )

            if encounters:
                encountered = True
                break

        # Over a full simulation duration (6000s) of random movement,
        # two nodes should meet at some point
        assert encountered, "Nodes never encountered each other in 6000 seconds"


# =============================================================================
# Role-Based Mobility Tests
# =============================================================================


class TestRoleAssignment:
    """Tests for deterministic role assignment."""

    def test_48_node_distribution(self):
        """With 48 nodes: 29 RESCUE, 12 TRANSPORT, 7 LIAISON."""
        roles = _assign_roles(48)

        assert len(roles) == 48
        assert roles.count(ResponderRole.RESCUE) == 29
        assert roles.count(ResponderRole.TRANSPORT) == 12
        assert roles.count(ResponderRole.LIAISON) == 7

    def test_5_node_distribution(self):
        """With 5 nodes: 3 RESCUE, 1 TRANSPORT, 1 LIAISON."""
        roles = _assign_roles(5)

        assert roles.count(ResponderRole.RESCUE) == 3
        assert roles.count(ResponderRole.TRANSPORT) == 1
        assert roles.count(ResponderRole.LIAISON) == 1

    def test_order_is_rescue_transport_liaison(self):
        """Roles are assigned in order: RESCUE first, then TRANSPORT, then LIAISON."""
        roles = _assign_roles(10)

        # First roles should all be RESCUE
        assert roles[0] == ResponderRole.RESCUE
        # Last should be LIAISON
        assert roles[-1] == ResponderRole.LIAISON
        # Somewhere in the middle should be TRANSPORT
        assert ResponderRole.TRANSPORT in roles

    def test_single_node(self):
        """Single node gets RESCUE (round(1*0.6) = 1)."""
        roles = _assign_roles(1)
        assert roles == [ResponderRole.RESCUE]

    def test_roles_assigned_during_initialize(self):
        """Roles are assigned to node states during initialize()."""
        params = NetworkParameters()
        manager = MobilityManager(parameters=params)

        mobile_ids = [f"mobile_{i}" for i in range(10)]
        positions = dict.fromkeys(mobile_ids, (100.0, 500.0))

        manager.initialize(mobile_ids, positions, random_seed=42)

        roles = [manager._node_states[nid].role for nid in mobile_ids]
        # 10 nodes: round(10*0.6)=6 RESCUE, round(10*0.25)=2 TRANSPORT, 2 LIAISON
        assert roles.count(ResponderRole.RESCUE) == 6
        assert (
            roles.count(ResponderRole.TRANSPORT) == 2
            or roles.count(ResponderRole.TRANSPORT) == 3
        )
        # Total should match
        assert len(roles) == 10


class TestRoleBasedWaypoints:
    """Tests for role-aware waypoint generation."""

    @pytest.fixture
    def params(self) -> NetworkParameters:
        return NetworkParameters()

    def test_rescue_waypoints_within_incident_zone(self, params: NetworkParameters):
        """RESCUE nodes generate waypoints only within the incident zone."""
        manager = MobilityManager(parameters=params)

        mobile_ids = [f"mobile_{i}" for i in range(48)]
        positions = dict.fromkeys(mobile_ids, (350.0, 750.0))
        manager.initialize(mobile_ids, positions, random_seed=42)

        iz = params.incident_zone

        # Check all RESCUE node waypoints
        for state in manager._node_states.values():
            if state.role == ResponderRole.RESCUE and state.waypoint is not None:
                assert (
                    iz.origin_x <= state.waypoint.x <= iz.origin_x + iz.width_m
                ), f"RESCUE waypoint x={state.waypoint.x} outside incident zone"
                assert (
                    iz.origin_y <= state.waypoint.y <= iz.origin_y + iz.height_m
                ), f"RESCUE waypoint y={state.waypoint.y} outside incident zone"

    def test_transport_waypoints_shuttle_zones(self, params: NetworkParameters):
        """TRANSPORT nodes alternate between incident and coordination zones."""
        manager = MobilityManager(parameters=params)

        mobile_ids = [f"mobile_{i}" for i in range(48)]
        # Start TRANSPORT nodes in the incident zone
        positions = dict.fromkeys(mobile_ids, (350.0, 750.0))
        manager.initialize(mobile_ids, positions, random_seed=42)

        iz = params.incident_zone
        cz = params.coordination_zone

        # TRANSPORT nodes starting in incident zone should head to coordination zone
        for state in manager._node_states.values():
            if state.role == ResponderRole.TRANSPORT and state.waypoint is not None:
                wp = state.waypoint
                in_iz = (
                    iz.origin_x <= wp.x <= iz.origin_x + iz.width_m
                    and iz.origin_y <= wp.y <= iz.origin_y + iz.height_m
                )
                in_cz = (
                    cz.origin_x <= wp.x <= cz.origin_x + cz.width_m
                    and cz.origin_y <= wp.y <= cz.origin_y + cz.height_m
                )
                assert in_iz or in_cz, (
                    f"TRANSPORT waypoint ({wp.x:.0f}, {wp.y:.0f}) "
                    f"not in incident or coordination zone"
                )

    def test_liaison_waypoints_full_area(self, params: NetworkParameters):
        """LIAISON nodes generate waypoints anywhere in the simulation area."""
        manager = MobilityManager(parameters=params)

        mobile_ids = [f"mobile_{i}" for i in range(48)]
        positions = dict.fromkeys(mobile_ids, (350.0, 750.0))
        manager.initialize(mobile_ids, positions, random_seed=42)

        sa = params.simulation_area

        for state in manager._node_states.values():
            if state.role == ResponderRole.LIAISON and state.waypoint is not None:
                assert sa.origin_x <= state.waypoint.x <= sa.origin_x + sa.width_m
                assert sa.origin_y <= state.waypoint.y <= sa.origin_y + sa.height_m

    def test_rescue_speed_range(self, params: NetworkParameters):
        """RESCUE nodes have speed in [1, 5] m/s."""
        manager = MobilityManager(parameters=params)

        mobile_ids = [f"mobile_{i}" for i in range(48)]
        positions = dict.fromkeys(mobile_ids, (350.0, 750.0))
        manager.initialize(mobile_ids, positions, random_seed=42)

        for state in manager._node_states.values():
            if state.role == ResponderRole.RESCUE and state.waypoint is not None:
                assert (
                    1.0 <= state.waypoint.speed <= 5.0
                ), f"RESCUE speed {state.waypoint.speed} outside [1, 5]"

    def test_transport_speed_range(self, params: NetworkParameters):
        """TRANSPORT nodes have speed in [5, 20] m/s."""
        manager = MobilityManager(parameters=params)

        mobile_ids = [f"mobile_{i}" for i in range(48)]
        positions = dict.fromkeys(mobile_ids, (350.0, 750.0))
        manager.initialize(mobile_ids, positions, random_seed=42)

        for state in manager._node_states.values():
            if state.role == ResponderRole.TRANSPORT and state.waypoint is not None:
                assert (
                    5.0 <= state.waypoint.speed <= 20.0
                ), f"TRANSPORT speed {state.waypoint.speed} outside [5, 20]"

    def test_liaison_speed_range(self, params: NetworkParameters):
        """LIAISON nodes have speed in [1, 10] m/s."""
        manager = MobilityManager(parameters=params)

        mobile_ids = [f"mobile_{i}" for i in range(48)]
        positions = dict.fromkeys(mobile_ids, (350.0, 750.0))
        manager.initialize(mobile_ids, positions, random_seed=42)

        for state in manager._node_states.values():
            if state.role == ResponderRole.LIAISON and state.waypoint is not None:
                assert (
                    1.0 <= state.waypoint.speed <= 10.0
                ), f"LIAISON speed {state.waypoint.speed} outside [1, 10]"


class TestBackwardCompatibility:
    """Tests that role=None falls back to original RWP behaviour."""

    def test_none_role_uses_full_area(self):
        """Nodes with role=None use the full simulation area."""
        params = NetworkParameters()
        manager = MobilityManager(parameters=params)

        mobile_ids = ["mobile_0"]
        positions = {"mobile_0": (100.0, 500.0)}
        manager.initialize(mobile_ids, positions, random_seed=42)

        # Override role to None
        state = manager._node_states["mobile_0"]
        state.role = None
        manager._assign_new_waypoint(state)

        sa = params.simulation_area
        assert sa.origin_x <= state.waypoint.x <= sa.origin_x + sa.width_m
        assert sa.origin_y <= state.waypoint.y <= sa.origin_y + sa.height_m

    def test_none_role_uses_manager_speed(self):
        """Nodes with role=None use the MobilityManager's speed range."""
        params = NetworkParameters()
        manager = MobilityManager(
            parameters=params,
            speed_min=1.0,
            speed_max=20.0,
        )

        mobile_ids = ["mobile_0"]
        positions = {"mobile_0": (100.0, 500.0)}
        manager.initialize(mobile_ids, positions, random_seed=42)

        state = manager._node_states["mobile_0"]
        state.role = None
        manager._assign_new_waypoint(state)

        assert 1.0 <= state.waypoint.speed <= 20.0


class TestRoleDeterminism:
    """Tests that role-based mobility is deterministic with same seed."""

    def test_same_seed_same_roles(self):
        """Same seed produces identical role assignments."""
        params = NetworkParameters()
        ids = [f"mobile_{i}" for i in range(48)]
        positions = dict.fromkeys(ids, (350.0, 750.0))

        m1 = MobilityManager(parameters=params)
        m1.initialize(ids, positions, random_seed=42)

        m2 = MobilityManager(parameters=params)
        m2.initialize(ids, positions, random_seed=42)

        for nid in ids:
            assert m1._node_states[nid].role == m2._node_states[nid].role

    def test_same_seed_same_waypoints(self):
        """Same seed produces identical initial waypoints."""
        params = NetworkParameters()
        ids = [f"mobile_{i}" for i in range(10)]
        positions = dict.fromkeys(ids, (350.0, 750.0))

        m1 = MobilityManager(parameters=params)
        m1.initialize(ids, positions, random_seed=99)

        m2 = MobilityManager(parameters=params)
        m2.initialize(ids, positions, random_seed=99)

        for nid in ids:
            wp1 = m1._node_states[nid].waypoint
            wp2 = m2._node_states[nid].waypoint
            assert wp1.x == wp2.x
            assert wp1.y == wp2.y
            assert wp1.speed == wp2.speed
