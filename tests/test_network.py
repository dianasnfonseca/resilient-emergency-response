"""
Tests for Network Topology Generation (Phase 1).

These tests verify that the network topology generator correctly implements
the specifications from the Phase 1 documentation:
- 50-node primary configuration (2 coordination + 48 mobile)
- Two-zone model: incident zone (700×600 m²) and coordination zone (50×50 m²)
- 100m radio range
- Connectivity scenarios: 75%, 40%, 20%
- Random Waypoint mobility preparation

Sources verified:
    - Ullah & Qayyum (2022): Network parameters
    - Karaman et al. (2026): Connectivity levels
"""

import numpy as np
import pytest

from conftest import (
    BUFFER_SIZE_BYTES,
    COORDINATION_NODE_COUNT,
    COORDINATION_ZONE_HEIGHT_M,
    COORDINATION_ZONE_WIDTH_M,
    INCIDENT_ZONE_HEIGHT_M,
    INCIDENT_ZONE_WIDTH_M,
    MOBILE_RESPONDER_COUNT,
    NODE_COUNT,
    RADIO_RANGE_M,
    SIMULATION_AREA_HEIGHT_M,
    SIMULATION_AREA_WIDTH_M,
)
from ercs.config.parameters import NetworkParameters, ZoneConfig
from ercs.network.topology import (
    generate_topology,
    NetworkTopology,
    Node,
    NodeType,
    TopologyGenerator,
)


class TestNodeClass:
    """Tests for the Node dataclass."""
    
    def test_node_creation(self):
        """Test basic node creation with all attributes."""
        node = Node(
            node_id="test_1",
            node_type=NodeType.COORDINATION,
            x=100.0,
            y=200.0,
            is_mobile=False,
            buffer_size_bytes=BUFFER_SIZE_BYTES,
        )
        
        assert node.node_id == "test_1"
        assert node.node_type == NodeType.COORDINATION
        assert node.x == 100.0
        assert node.y == 200.0
        assert node.is_mobile is False
        assert node.buffer_size_bytes == BUFFER_SIZE_BYTES
    
    def test_distance_calculation(self):
        """Test Euclidean distance calculation between nodes."""
        node_a = Node("a", NodeType.MOBILE_RESPONDER, 0.0, 0.0, True, BUFFER_SIZE_BYTES)
        node_b = Node("b", NodeType.MOBILE_RESPONDER, 3.0, 4.0, True, BUFFER_SIZE_BYTES)
        
        # 3-4-5 triangle
        assert node_a.distance_to(node_b) == 5.0
        assert node_b.distance_to(node_a) == 5.0
    
    def test_distance_same_location(self):
        """Test distance to node at same location is zero."""
        node_a = Node("a", NodeType.MOBILE_RESPONDER, 100.0, 100.0, True, BUFFER_SIZE_BYTES)
        node_b = Node("b", NodeType.MOBILE_RESPONDER, 100.0, 100.0, True, BUFFER_SIZE_BYTES)
        
        assert node_a.distance_to(node_b) == 0.0
    
    def test_within_range_true(self):
        """Test nodes within communication range."""
        node_a = Node("a", NodeType.MOBILE_RESPONDER, 0.0, 0.0, True, BUFFER_SIZE_BYTES)
        node_b = Node("b", NodeType.MOBILE_RESPONDER, 50.0, 0.0, True, BUFFER_SIZE_BYTES)
        
        # 50m apart, 100m range
        assert node_a.is_within_range(node_b, 100.0) == True
    
    def test_within_range_false(self):
        """Test nodes outside communication range."""
        node_a = Node("a", NodeType.MOBILE_RESPONDER, 0.0, 0.0, True, BUFFER_SIZE_BYTES)
        node_b = Node("b", NodeType.MOBILE_RESPONDER, 150.0, 0.0, True, BUFFER_SIZE_BYTES)
        
        # 150m apart, 100m range
        assert node_a.is_within_range(node_b, 100.0) == False
    
    def test_within_range_exact_boundary(self):
        """Test nodes exactly at communication range boundary."""
        node_a = Node("a", NodeType.MOBILE_RESPONDER, 0.0, 0.0, True, BUFFER_SIZE_BYTES)
        node_b = Node("b", NodeType.MOBILE_RESPONDER, 100.0, 0.0, True, BUFFER_SIZE_BYTES)
        
        # Exactly 100m apart, 100m range (should be within)
        assert node_a.is_within_range(node_b, 100.0) == True
    
    def test_to_dict(self):
        """Test conversion to dictionary for NetworkX attributes."""
        node = Node("test", NodeType.COORDINATION, 50.0, 75.0, False, BUFFER_SIZE_BYTES)
        
        d = node.to_dict()
        
        assert d["node_type"] == "coordination"
        assert d["x"] == 50.0
        assert d["y"] == 75.0
        assert d["is_mobile"] is False
        assert d["buffer_size_bytes"] == BUFFER_SIZE_BYTES


class TestTopologyGenerator:
    """Tests for the TopologyGenerator class."""
    
    @pytest.fixture
    def default_params(self) -> NetworkParameters:
        """Default network parameters matching Phase 1 spec."""
        return NetworkParameters()
    
    @pytest.fixture
    def generator(self, default_params: NetworkParameters) -> TopologyGenerator:
        """Create a generator with default parameters and fixed seed."""
        return TopologyGenerator(default_params, random_seed=42)
    
    def test_generates_correct_node_count(self, generator: TopologyGenerator):
        """Verify total node count matches specification (50 nodes)."""
        topology = generator.generate()
        
        assert topology.total_nodes == NODE_COUNT

    def test_generates_correct_coordination_nodes(self, generator: TopologyGenerator):
        """Verify coordination node count (2 fixed nodes)."""
        topology = generator.generate()

        assert len(topology.coordination_nodes) == COORDINATION_NODE_COUNT
        
        for node_id in topology.coordination_nodes:
            node = topology.get_node(node_id)
            assert node.node_type == NodeType.COORDINATION
            assert node.is_mobile is False
    
    def test_generates_correct_mobile_nodes(self, generator: TopologyGenerator):
        """Verify mobile responder count (48 mobile nodes)."""
        topology = generator.generate()
        
        assert len(topology.mobile_nodes) == MOBILE_RESPONDER_COUNT

        for node_id in topology.mobile_nodes:
            node = topology.get_node(node_id)
            assert node.node_type == NodeType.MOBILE_RESPONDER
            assert node.is_mobile is True

    def test_coordination_nodes_in_coordination_zone(self, generator: TopologyGenerator):
        """Verify coordination nodes are placed within coordination zone."""
        topology = generator.generate()
        zone = generator.parameters.coordination_zone
        
        for node_id in topology.coordination_nodes:
            node = topology.get_node(node_id)
            
            assert zone.origin_x <= node.x <= zone.origin_x + zone.width_m
            assert zone.origin_y <= node.y <= zone.origin_y + zone.height_m
    
    def test_mobile_nodes_in_incident_zone(self, generator: TopologyGenerator):
        """Verify mobile responders are placed within incident zone."""
        topology = generator.generate()
        zone = generator.parameters.incident_zone
        
        for node_id in topology.mobile_nodes:
            node = topology.get_node(node_id)
            
            assert zone.origin_x <= node.x <= zone.origin_x + zone.width_m
            assert zone.origin_y <= node.y <= zone.origin_y + zone.height_m
    
    def test_reproducible_with_seed(self, default_params: NetworkParameters):
        """Verify same seed produces identical topology."""
        gen1 = TopologyGenerator(default_params, random_seed=12345)
        gen2 = TopologyGenerator(default_params, random_seed=12345)
        
        topo1 = gen1.generate()
        topo2 = gen2.generate()
        
        # Check all mobile node positions are identical
        for node_id in topo1.mobile_nodes:
            node1 = topo1.get_node(node_id)
            node2 = topo2.get_node(node_id)
            
            assert node1.x == node2.x
            assert node1.y == node2.y
    
    def test_different_seeds_produce_different_topologies(
        self, default_params: NetworkParameters
    ):
        """Verify different seeds produce different node positions."""
        gen1 = TopologyGenerator(default_params, random_seed=111)
        gen2 = TopologyGenerator(default_params, random_seed=222)
        
        topo1 = gen1.generate()
        topo2 = gen2.generate()
        
        # At least some mobile nodes should have different positions
        differences = 0
        for node_id in topo1.mobile_nodes:
            node1 = topo1.get_node(node_id)
            node2 = topo2.get_node(node_id)
            
            if node1.x != node2.x or node1.y != node2.y:
                differences += 1
        
        assert differences > 0
    
    def test_buffer_size_assigned(self, generator: TopologyGenerator):
        """Verify all nodes have correct buffer size (5 MB)."""
        topology = generator.generate()
        expected_buffer = BUFFER_SIZE_BYTES
        
        for node in topology.nodes.values():
            assert node.buffer_size_bytes == expected_buffer


class TestConnectivityScenarios:
    """Tests for connectivity level scenarios from Karaman et al. (2026)."""
    
    @pytest.fixture
    def generator(self) -> TopologyGenerator:
        """Create generator with fixed seed for reproducibility."""
        return TopologyGenerator(NetworkParameters(), random_seed=42)
    
    def test_full_connectivity_has_most_edges(self, generator: TopologyGenerator):
        """Verify full connectivity (1.0) has maximum edges."""
        full = generator.generate(connectivity_level=None)
        partial = generator.generate(connectivity_level=0.75)
        
        assert full.total_edges >= partial.total_edges
    
    def test_mild_degradation_75_percent(self, generator: TopologyGenerator):
        """Test 75% connectivity scenario (mild degradation)."""
        topology = generator.generate(connectivity_level=0.75)
        full_topology = generator.generate(connectivity_level=None)
        
        # Should have approximately 75% of full edges (allow wider variance due to randomness)
        if full_topology.total_edges > 0:
            ratio = topology.total_edges / full_topology.total_edges
            assert 0.60 <= ratio <= 0.90  # Allow wider variance
    
    def test_moderate_degradation_40_percent(self, generator: TopologyGenerator):
        """Test 40% connectivity scenario (moderate degradation)."""
        topology = generator.generate(connectivity_level=0.40)
        full_topology = generator.generate(connectivity_level=None)
        
        if full_topology.total_edges > 0:
            ratio = topology.total_edges / full_topology.total_edges
            assert 0.30 <= ratio <= 0.50
    
    def test_severe_degradation_20_percent(self, generator: TopologyGenerator):
        """Test 20% connectivity scenario (severe degradation)."""
        topology = generator.generate(connectivity_level=0.20)
        full_topology = generator.generate(connectivity_level=None)
        
        if full_topology.total_edges > 0:
            ratio = topology.total_edges / full_topology.total_edges
            assert 0.10 <= ratio <= 0.30
    
    def test_zero_connectivity_no_edges(self, generator: TopologyGenerator):
        """Test 0% connectivity produces no edges."""
        topology = generator.generate(connectivity_level=0.0)
        
        assert topology.total_edges == 0


class TestNetworkTopology:
    """Tests for the NetworkTopology container class."""
    
    @pytest.fixture
    def topology(self) -> NetworkTopology:
        """Generate a test topology."""
        return generate_topology(random_seed=42)
    
    def test_get_node(self, topology: NetworkTopology):
        """Test node retrieval by ID."""
        node = topology.get_node("coord_0")
        
        assert node is not None
        assert node.node_type == NodeType.COORDINATION
    
    def test_get_node_invalid_id(self, topology: NetworkTopology):
        """Test that invalid node ID raises KeyError."""
        with pytest.raises(KeyError):
            topology.get_node("nonexistent_node")
    
    def test_get_neighbours(self, topology: NetworkTopology):
        """Test neighbour retrieval for a node."""
        # Coordination nodes should be neighbours (within 50m zone, 100m range)
        neighbours = topology.get_neighbours("coord_0")
        
        # Should include at least coord_1
        assert "coord_1" in neighbours
    
    def test_are_connected(self, topology: NetworkTopology):
        """Test connection check between nodes."""
        # Coordination nodes should be connected
        assert topology.are_connected("coord_0", "coord_1") is True
    
    def test_get_distance(self, topology: NetworkTopology):
        """Test distance calculation between nodes."""
        distance = topology.get_distance("coord_0", "coord_1")
        
        # Distance should be positive and less than zone width
        assert distance > 0
        assert distance < 100  # Less than radio range since they're connected
    
    def test_nodes_by_type_coordination(self, topology: NetworkTopology):
        """Test filtering nodes by coordination type."""
        coord_nodes = list(topology.nodes_by_type(NodeType.COORDINATION))
        
        assert len(coord_nodes) == COORDINATION_NODE_COUNT
        for node in coord_nodes:
            assert node.node_type == NodeType.COORDINATION

    def test_nodes_by_type_mobile(self, topology: NetworkTopology):
        """Test filtering nodes by mobile type."""
        mobile_nodes = list(topology.nodes_by_type(NodeType.MOBILE_RESPONDER))

        assert len(mobile_nodes) == MOBILE_RESPONDER_COUNT
        for node in mobile_nodes:
            assert node.node_type == NodeType.MOBILE_RESPONDER


class TestGenerateTopologyFunction:
    """Tests for the convenience function generate_topology()."""
    
    def test_default_parameters(self):
        """Test generation with all defaults."""
        topology = generate_topology()
        
        assert topology.total_nodes == NODE_COUNT
        assert len(topology.coordination_nodes) == COORDINATION_NODE_COUNT
        assert len(topology.mobile_nodes) == MOBILE_RESPONDER_COUNT

    def test_with_connectivity_level(self):
        """Test generation with specific connectivity."""
        topology = generate_topology(connectivity_level=0.5, random_seed=42)

        assert topology.total_nodes == NODE_COUNT
    
    def test_with_custom_parameters(self):
        """Test generation with custom parameters."""
        params = NetworkParameters(
            primary_node_count=30,
            coordination_node_count=2,
            mobile_responder_count=28,
        )
        
        topology = generate_topology(parameters=params, random_seed=42)
        
        assert topology.total_nodes == 30
        assert len(topology.coordination_nodes) == 2
        assert len(topology.mobile_nodes) == 28
    
    def test_reproducibility(self):
        """Test that same seed produces same topology."""
        topo1 = generate_topology(random_seed=999)
        topo2 = generate_topology(random_seed=999)
        
        for node_id in topo1.nodes:
            assert topo1.get_node(node_id).x == topo2.get_node(node_id).x
            assert topo1.get_node(node_id).y == topo2.get_node(node_id).y


class TestZoneBoundaryConditions:
    """Tests for zone boundary and edge cases."""
    
    def test_incident_zone_dimensions(self):
        """Verify incident zone matches spec: 700 × 600 m²."""
        params = NetworkParameters()
        
        assert params.incident_zone.width_m == INCIDENT_ZONE_WIDTH_M
        assert params.incident_zone.height_m == INCIDENT_ZONE_HEIGHT_M
    
    def test_coordination_zone_dimensions(self):
        """Verify coordination zone matches spec: 50 × 50 m²."""
        params = NetworkParameters()
        
        assert params.coordination_zone.width_m == COORDINATION_ZONE_WIDTH_M
        assert params.coordination_zone.height_m == COORDINATION_ZONE_HEIGHT_M
    
    def test_simulation_area_dimensions(self):
        """Verify simulation area matches spec: 3000 × 1500 m²."""
        params = NetworkParameters()
        
        assert params.simulation_area.width_m == SIMULATION_AREA_WIDTH_M
        assert params.simulation_area.height_m == SIMULATION_AREA_HEIGHT_M
    
    def test_coordination_zone_at_eastern_edge(self):
        """Verify coordination zone is positioned at eastern edge."""
        params = NetworkParameters()
        
        # Coordination zone should be near the right edge (x ≈ 2900)
        coord_zone = params.coordination_zone
        sim_area = params.simulation_area
        
        # Zone should end near the simulation area's right boundary
        zone_right_edge = coord_zone.origin_x + coord_zone.width_m
        assert zone_right_edge <= sim_area.width_m
        assert zone_right_edge >= sim_area.width_m - 200  # Within 200m of edge
    
    def test_incident_zone_at_western_portion(self):
        """Verify incident zone is in western portion of simulation area."""
        params = NetworkParameters()
        
        incident_zone = params.incident_zone
        sim_area = params.simulation_area
        
        # Incident zone should start at or near x=0
        assert incident_zone.origin_x < sim_area.width_m / 2
    
    def test_zones_do_not_overlap(self):
        """Verify incident and coordination zones don't overlap."""
        params = NetworkParameters()
        
        incident = params.incident_zone
        coord = params.coordination_zone
        
        # Check x-axis separation
        incident_right = incident.origin_x + incident.width_m
        coord_left = coord.origin_x
        
        assert incident_right < coord_left or coord.origin_x + coord.width_m < incident.origin_x


class TestGraphProperties:
    """Tests for NetworkX graph properties and structure."""
    
    def test_graph_is_undirected(self):
        """Verify the graph is undirected (bidirectional communication)."""
        topology = generate_topology(random_seed=42)
        
        assert not topology.graph.is_directed()
    
    def test_all_nodes_in_graph(self):
        """Verify all nodes are added to the graph."""
        topology = generate_topology(random_seed=42)
        
        assert len(topology.graph.nodes) == topology.total_nodes
    
    def test_node_attributes_in_graph(self):
        """Verify node attributes are stored in graph."""
        topology = generate_topology(random_seed=42)
        
        for node_id, data in topology.graph.nodes(data=True):
            assert "node_type" in data
            assert "x" in data
            assert "y" in data
            assert "is_mobile" in data
            assert "buffer_size_bytes" in data
    
    def test_edge_has_distance_weight(self):
        """Verify edges have distance as weight attribute."""
        topology = generate_topology(random_seed=42)
        
        for u, v, data in topology.graph.edges(data=True):
            assert "distance" in data
            assert data["distance"] >= 0
            assert data["distance"] <= topology.parameters.radio_range_m
    
    def test_edges_respect_radio_range(self):
        """Verify all edges are within radio range."""
        topology = generate_topology(random_seed=42)
        radio_range = topology.parameters.radio_range_m
        
        for u, v in topology.graph.edges():
            distance = topology.get_distance(u, v)
            assert distance <= radio_range


class TestSensitivityAnalysisSupport:
    """Tests for sensitivity analysis configurations (30, 50, 70 nodes)."""
    
    @pytest.mark.parametrize("node_count", [30, 50, 70])
    def test_sensitivity_node_counts(self, node_count: int):
        """Test topology generation for sensitivity analysis node counts."""
        # Maintain ratio: coordination = 2, mobile = rest
        params = NetworkParameters(
            primary_node_count=node_count,
            coordination_node_count=2,
            mobile_responder_count=node_count - 2,
        )
        
        topology = generate_topology(parameters=params, random_seed=42)
        
        assert topology.total_nodes == node_count
        assert len(topology.coordination_nodes) == 2
        assert len(topology.mobile_nodes) == node_count - 2
