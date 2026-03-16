"""
Network Topology Generation (Phase 1).

This module generates network topologies for emergency response simulation,
implementing the two-zone disaster area model with coordination and mobile
responder nodes.

"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

import networkx as nx
import numpy as np

from ercs.config.parameters import NetworkParameters, ZoneConfig


class NodeType(str, Enum):
    """Node type classification for DTN simulation."""

    COORDINATION = "coordination"
    MOBILE_RESPONDER = "mobile_responder"


@dataclass
class Node:
    """
    Represents a node in the emergency response network.

    Attributes:
        node_id: Unique identifier for the node
        node_type: Classification (coordination or mobile_responder)
        x: X coordinate in metres
        y: Y coordinate in metres
        is_mobile: Whether the node can move (False for coordination nodes)
        buffer_size_bytes: Message buffer capacity
    """

    node_id: str
    node_type: NodeType
    x: float
    y: float
    is_mobile: bool
    buffer_size_bytes: int

    def distance_to(self, other: "Node") -> float:
        """Calculate Euclidean distance to another node in metres."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def is_within_range(self, other: "Node", radio_range_m: float) -> bool:
        """Check if another node is within communication range."""
        return self.distance_to(other) <= radio_range_m

    def to_dict(self) -> dict:
        """Convert node to dictionary for NetworkX node attributes."""
        return {
            "node_type": self.node_type.value,
            "x": self.x,
            "y": self.y,
            "is_mobile": self.is_mobile,
            "buffer_size_bytes": self.buffer_size_bytes,
        }


@dataclass
class NetworkTopology:
    """
    Complete network topology for emergency response simulation.

    Contains the network graph, node objects, and configuration parameters.
    Provides methods for querying network structure and connectivity.
    """

    graph: nx.Graph
    nodes: dict[str, Node]
    parameters: NetworkParameters
    random_seed: int | None = None

    # Computed properties
    _coordination_nodes: list[str] = field(default_factory=list, repr=False)
    _mobile_nodes: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Classify nodes by type after initialization."""
        self._coordination_nodes = [
            nid
            for nid, node in self.nodes.items()
            if node.node_type == NodeType.COORDINATION
        ]
        self._mobile_nodes = [
            nid
            for nid, node in self.nodes.items()
            if node.node_type == NodeType.MOBILE_RESPONDER
        ]

    @property
    def coordination_nodes(self) -> list[str]:
        """List of coordination node IDs."""
        return self._coordination_nodes

    @property
    def mobile_nodes(self) -> list[str]:
        """List of mobile responder node IDs."""
        return self._mobile_nodes

    @property
    def total_nodes(self) -> int:
        """Total number of nodes in the network."""
        return len(self.nodes)

    @property
    def total_edges(self) -> int:
        """Total number of edges (potential communication links)."""
        return self.graph.number_of_edges()

    def get_node(self, node_id: str) -> Node:
        """Get a node by ID."""
        return self.nodes[node_id]

    def get_neighbours(self, node_id: str) -> list[str]:
        """Get IDs of nodes within communication range."""
        return list(self.graph.neighbors(node_id))

    def are_connected(self, node_a: str, node_b: str) -> bool:
        """Check if two nodes have a direct communication link."""
        return self.graph.has_edge(node_a, node_b)

    def get_distance(self, node_a: str, node_b: str) -> float:
        """Get Euclidean distance between two nodes in metres."""
        return self.nodes[node_a].distance_to(self.nodes[node_b])

    def nodes_by_type(self, node_type: NodeType) -> Iterator[Node]:
        """Iterate over nodes of a specific type."""
        for node in self.nodes.values():
            if node.node_type == node_type:
                yield node

    def get_all_node_ids(self) -> list[str]:
        """Get list of all node IDs."""
        return list(self.nodes.keys())

    def get_coordination_node_ids(self) -> list[str]:
        """Get list of coordination node IDs."""
        return list(self._coordination_nodes)

    def get_mobile_responder_ids(self) -> list[str]:
        """Get list of mobile responder node IDs."""
        return list(self._mobile_nodes)

    def get_node_position(self, node_id: str) -> tuple[float, float] | None:
        """Get (x, y) position of a node."""
        node = self.nodes.get(node_id)
        if node is None:
            return None
        return (node.x, node.y)

    def update_node_position(self, node_id: str, x: float, y: float) -> None:
        """
        Update a node's position.

        Used by the mobility system to update node positions during simulation.
        Also updates the graph node attributes.

        Args:
            node_id: ID of node to update
            x: New X coordinate (metres)
            y: New Y coordinate (metres)

        Raises:
            KeyError: If node_id not found
            ValueError: If trying to move a non-mobile node
        """
        node = self.nodes[node_id]  # Raises KeyError if not found

        if not node.is_mobile:
            raise ValueError(f"Cannot move non-mobile node: {node_id}")

        node.x = x
        node.y = y

        self.graph.nodes[node_id]["x"] = x
        self.graph.nodes[node_id]["y"] = y

    def update_edges_from_positions(self) -> list[tuple[str, str]]:
        """
        Recalculate all edges based on current node positions.

        Removes edges for nodes that are now out of range and adds
        edges for nodes that are now in range.

        Returns:
            List of (node_a, node_b) tuples for NEW connections
            (nodes that just came into range)
        """
        new_connections = []
        radio_range = self.parameters.radio_range_m
        node_ids = list(self.nodes.keys())

        for i, node_a_id in enumerate(node_ids):
            node_a = self.nodes[node_a_id]

            for node_b_id in node_ids[i + 1 :]:
                node_b = self.nodes[node_b_id]

                distance = node_a.distance_to(node_b)
                currently_connected = self.graph.has_edge(node_a_id, node_b_id)
                should_connect = distance <= radio_range

                if should_connect and not currently_connected:
                    self.graph.add_edge(node_a_id, node_b_id, distance=distance)
                    new_connections.append((node_a_id, node_b_id))
                elif not should_connect and currently_connected:
                    self.graph.remove_edge(node_a_id, node_b_id)
                elif should_connect and currently_connected:
                    self.graph[node_a_id][node_b_id]["distance"] = distance

        return new_connections

    def get_connected_pairs(self) -> list[tuple[str, str]]:
        """
        Get all currently connected node pairs.

        Returns:
            List of (node_a, node_b) tuples for all edges
        """
        return list(self.graph.edges())


class TopologyGenerator:
    """
    Generates network topologies for emergency response simulation.

    Implements the two-zone disaster area model:
    - Incident zone: Where emergency tasks originate, mobile responders operate
    - Coordination zone: Fixed infrastructure for message aggregation

    """

    def __init__(self, parameters: NetworkParameters, random_seed: int | None = None):
        """
        Initialize the topology generator.

        Args:
            parameters: Network configuration parameters from Phase 1 spec
            random_seed: Optional seed for reproducible node placement
        """
        self.parameters = parameters
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def generate(self) -> NetworkTopology:
        """
        Generate a complete network topology.

        All nodes within radio range are connected.  Infrastructure damage
        (connectivity_level) is applied at encounter time in the simulation
        engine via ``_is_link_available()``, not here.

        Returns:
            NetworkTopology object containing graph and node data
        """
        # Generate nodes
        nodes = self._generate_nodes()

        # Create network graph (full range-based connectivity)
        graph = self._create_graph(nodes)

        return NetworkTopology(
            graph=graph,
            nodes=nodes,
            parameters=self.parameters,
            random_seed=self.random_seed,
        )

    def _generate_nodes(self) -> dict[str, Node]:
        """
        Generate all nodes with positions according to zone specifications.

        Coordination nodes: Fixed positions within coordination zone
        Mobile responders: Uniform random distribution within incident zone

        Returns:
            Dictionary mapping node IDs to Node objects
        """
        nodes: dict[str, Node] = {}

        # Generate coordination nodes (fixed in coordination zone)
        coord_nodes = self._generate_coordination_nodes()
        nodes.update(coord_nodes)

        # Generate mobile responder nodes (random in incident zone)
        mobile_nodes = self._generate_mobile_nodes()
        nodes.update(mobile_nodes)

        return nodes

    def _generate_coordination_nodes(self) -> dict[str, Node]:
        """
        Generate coordination nodes with fixed positions in coordination zone.

        The coordination zone (50 × 50 m²) is smaller than the communication
        range (100 m), ensuring coordination nodes can always communicate
        with each other.

        Returns:
            Dictionary of coordination node IDs to Node objects
        """
        nodes: dict[str, Node] = {}
        zone = self.parameters.coordination_zone
        count = self.parameters.coordination_node_count

        # Place coordination nodes in a grid pattern within the zone
        # For 2 nodes: place at 1/3 and 2/3 positions to ensure spacing
        for i in range(count):
            # Distribute evenly across the zone
            fraction = (i + 1) / (count + 1)
            x = zone.origin_x + zone.width_m * fraction
            y = zone.origin_y + zone.height_m * 0.5  # Center vertically

            node_id = f"coord_{i}"
            nodes[node_id] = Node(
                node_id=node_id,
                node_type=NodeType.COORDINATION,
                x=x,
                y=y,
                is_mobile=False,
                buffer_size_bytes=self.parameters.buffer_size_bytes,
            )

        return nodes

    def _generate_mobile_nodes(self) -> dict[str, Node]:
        """
        Generate mobile responder nodes with uniform random positions.

        Initial positions are uniformly distributed within the incident zone,
        following standard DTN simulation practice.

        Returns:
            Dictionary of mobile node IDs to Node objects
        """
        nodes: dict[str, Node] = {}
        zone = self.parameters.incident_zone
        count = self.parameters.mobile_responder_count

        # Generate uniform random positions within incident zone
        x_positions = self._rng.uniform(
            zone.origin_x, zone.origin_x + zone.width_m, size=count
        )
        y_positions = self._rng.uniform(
            zone.origin_y, zone.origin_y + zone.height_m, size=count
        )

        for i in range(count):
            node_id = f"mobile_{i}"
            nodes[node_id] = Node(
                node_id=node_id,
                node_type=NodeType.MOBILE_RESPONDER,
                x=float(x_positions[i]),
                y=float(y_positions[i]),
                is_mobile=True,
                buffer_size_bytes=self.parameters.buffer_size_bytes,
            )

        return nodes

    def _create_graph(self, nodes: dict[str, Node]) -> nx.Graph:
        """
        Create NetworkX graph with edges based on communication range.

        All node pairs within radio range receive an edge.  Infrastructure
        damage filtering (connectivity_level) is applied at encounter time
        in the simulation engine, not during topology generation — this
        avoids the inconsistency where ``update_edges_from_positions()``
        would re-add all range-based edges on the first mobility step,
        washing out any initial filtering.

        Args:
            nodes: Dictionary of node IDs to Node objects

        Returns:
            NetworkX Graph with nodes and edges
        """
        graph = nx.Graph()

        # Add nodes with attributes
        for node_id, node in nodes.items():
            graph.add_node(node_id, **node.to_dict())

        # Add all edges for nodes within radio range
        node_ids = list(nodes.keys())
        radio_range = self.parameters.radio_range_m

        for i, node_a_id in enumerate(node_ids):
            for node_b_id in node_ids[i + 1 :]:
                node_a = nodes[node_a_id]
                node_b = nodes[node_b_id]

                distance = node_a.distance_to(node_b)
                if distance <= radio_range:
                    graph.add_edge(node_a_id, node_b_id, distance=distance)

        return graph

    def get_zone_bounds(self, zone: ZoneConfig) -> tuple[float, float, float, float]:
        """
        Get the bounding box coordinates for a zone.

        Returns:
            Tuple of (min_x, max_x, min_y, max_y)
        """
        return (
            zone.origin_x,
            zone.origin_x + zone.width_m,
            zone.origin_y,
            zone.origin_y + zone.height_m,
        )


def generate_topology(
    parameters: NetworkParameters | None = None,
    random_seed: int | None = None,
    **kwargs,
) -> NetworkTopology:
    """
    Convenience function to generate a network topology.

    All nodes within radio range are connected.  Infrastructure damage
    (connectivity_level) is applied at encounter time in the simulation
    engine, not during topology generation.

    Args:
        parameters: Network parameters (uses defaults if None)
        random_seed: Random seed for reproducibility

    Returns:
        Generated NetworkTopology

    Example:
        >>> from ercs.network import generate_topology
        >>> topology = generate_topology(random_seed=42)
        >>> print(f"Nodes: {topology.total_nodes}, Edges: {topology.total_edges}")
    """
    # Accept (and ignore) legacy connectivity_level kwarg for
    # backward compatibility with scripts and tests.
    kwargs.pop("connectivity_level", None)

    if parameters is None:
        parameters = NetworkParameters()

    generator = TopologyGenerator(parameters, random_seed)
    return generator.generate()
