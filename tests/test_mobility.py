"""
Tests for Random Waypoint Mobility Model.

These tests verify that the mobility system correctly implements
the Random Waypoint model as specified in Chapter1_v2:
- Speed range: 0-20 m/s
- Operating zone: Entire simulation area
- Dynamic edge updates based on proximity (100m range)
- PRoPHET encounter triggering
"""

import pytest
import numpy as np

from ercs.config.parameters import NetworkParameters
from conftest import RADIO_RANGE_M, SIMULATION_DURATION_S, SPEED_MAX_MPS
from ercs.network.mobility import (
    MobilityManager,
    MobileNodeState,
    MobilityState,
    Waypoint,
    calculate_encounters,
    update_topology_edges,
)


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
    
    def test_positions_within_simulation_area(self, initialized_manager: MobilityManager):
        """Test that nodes stay within simulation area."""
        params = initialized_manager.parameters
        
        # Run many steps
        for i in range(100):
            initialized_manager.step(float(i), 1.0)
        
        positions = initialized_manager.get_all_positions()
        
        for node_id, (x, y) in positions.items():
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
            "node_b": (50.0, 0.0),   # In range of A
            "node_c": (80.0, 0.0),   # In range of A and B
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
        min_distance_seen = float('inf')
        
        for i in range(SIMULATION_DURATION_S):  # 6000 seconds
            manager.step(float(i), 1.0)
            pos = manager.get_position("mobile_0")

            # Check distance to coordination zone center
            dx = pos[0] - coord_center_x
            dy = pos[1] - coord_center_y
            distance = np.sqrt(dx*dx + dy*dy)

            min_distance_seen = min(min_distance_seen, distance)

            # If within range of coordination zone, success
            if distance < RADIO_RANGE_M:
                break
        
        # Node should have gotten reasonably close to coordination zone
        # at some point during the simulation
        assert min_distance_seen < 500.0, \
            f"Node never got close to coordination zone, min distance: {min_distance_seen}"
    
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
            encounters = calculate_encounters(current_positions, radio_range=RADIO_RANGE_M)

            if encounters:
                encountered = True
                break

        # Over a full simulation duration (6000s) of random movement,
        # two nodes should meet at some point
        assert encountered, "Nodes never encountered each other in 6000 seconds"
