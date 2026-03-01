"""
Random Waypoint Mobility Model (Phase 1 Extension).

This module implements node mobility for the emergency response simulation,
enabling dynamic network topology changes essential for PRoPHET protocol
operation and realistic DTN evaluation.

The Random Waypoint model:
1. Node selects random destination within operating zone
2. Node moves toward destination at random speed (0-20 m/s)
3. Optional pause at destination
4. Repeat

Operating Zone:
- Mobile responders can move within the entire simulation area
- This enables transit between incident zone and coordination zone
- Creates encounter opportunities for message relay

Sources:
    - Ullah & Qayyum (2022): Random Waypoint model, speed range 0-20 m/s
    - Keykhaei et al. (2024): Multi-agent mobility in emergency evacuation
    - Chapter1_v2: Mobility model selection rationale
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from ercs.config.parameters import NetworkParameters, ZoneConfig


class MobilityState(str, Enum):
    """State of a mobile node."""
    
    MOVING = "moving"
    PAUSED = "paused"


@dataclass
class Waypoint:
    """
    Target waypoint for a mobile node.
    
    Attributes:
        x: Target X coordinate (metres)
        y: Target Y coordinate (metres)
        speed: Movement speed toward waypoint (m/s)
        pause_duration: Time to pause at waypoint (seconds)
    """
    
    x: float
    y: float
    speed: float
    pause_duration: float = 0.0


@dataclass
class MobileNodeState:
    """
    Mobility state for a single node.
    
    Attributes:
        node_id: Unique identifier
        x: Current X position (metres)
        y: Current Y position (metres)
        state: Current mobility state
        waypoint: Current target waypoint (if moving)
        pause_end_time: When pause ends (if paused)
    """
    
    node_id: str
    x: float
    y: float
    state: MobilityState = MobilityState.PAUSED
    waypoint: Waypoint | None = None
    pause_end_time: float = 0.0
    
    def distance_to_waypoint(self) -> float:
        """Calculate distance to current waypoint."""
        if self.waypoint is None:
            return 0.0
        dx = self.waypoint.x - self.x
        dy = self.waypoint.y - self.y
        return np.sqrt(dx * dx + dy * dy)


class NodePositionUpdater(Protocol):
    """Protocol for updating node positions in the topology."""
    
    def update_node_position(self, node_id: str, x: float, y: float) -> None:
        """Update a node's position."""
        ...


@dataclass
class MobilityManager:
    """
    Manages Random Waypoint mobility for all mobile nodes.
    
    Implements the Random Waypoint model where nodes:
    1. Select a random destination within the operating zone
    2. Move toward it at a random speed (0-20 m/s)
    3. Optionally pause upon arrival
    4. Select a new destination
    
    The operating zone encompasses the entire simulation area to enable
    mobile responders to transit between incident and coordination zones,
    creating encounter opportunities essential for DTN message relay.
    
    Sources:
        - Ullah & Qayyum (2022): Speed range 0-20 m/s
        - Chapter1_v2: Operating zone rationale
    
    Attributes:
        parameters: Network configuration
        speed_min: Minimum movement speed (m/s)
        speed_max: Maximum movement speed (m/s)
        pause_min: Minimum pause duration (seconds)
        pause_max: Maximum pause duration (seconds)
    """
    
    parameters: NetworkParameters
    speed_min: float = 0.0
    speed_max: float = 20.0  # Source: Ullah & Qayyum (2022)
    pause_min: float = 0.0
    pause_max: float = 30.0  # Brief pauses for realism
    
    # Internal state
    _node_states: dict[str, MobileNodeState] = field(default_factory=dict)
    _rng: np.random.Generator = field(default=None)
    
    def __post_init__(self) -> None:
        """Initialize random generator."""
        if self._rng is None:
            self._rng = np.random.default_rng()
    
    def initialize(
        self,
        mobile_node_ids: list[str],
        initial_positions: dict[str, tuple[float, float]],
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize mobility states for all mobile nodes.
        
        Args:
            mobile_node_ids: IDs of mobile nodes
            initial_positions: Mapping of node_id to (x, y) position
            random_seed: Optional seed for reproducibility
        """
        self._rng = np.random.default_rng(random_seed)
        self._node_states.clear()
        
        for node_id in mobile_node_ids:
            if node_id in initial_positions:
                x, y = initial_positions[node_id]
                state = MobileNodeState(
                    node_id=node_id,
                    x=x,
                    y=y,
                    state=MobilityState.PAUSED,
                    pause_end_time=0.0,
                )
                self._node_states[node_id] = state
                
                # Immediately assign initial waypoint
                self._assign_new_waypoint(state)
    
    def step(
        self,
        current_time: float,
        delta_time: float,
    ) -> list[str]:
        """
        Advance mobility simulation by one time step.
        
        Updates positions of all mobile nodes based on their current
        waypoints and movement speeds.
        
        Args:
            current_time: Current simulation time (seconds)
            delta_time: Time step duration (seconds)
            
        Returns:
            List of node IDs that moved during this step
        """
        moved_nodes = []
        
        for node_id, state in self._node_states.items():
            if self._update_node(state, current_time, delta_time):
                moved_nodes.append(node_id)
        
        return moved_nodes
    
    def _update_node(
        self,
        state: MobileNodeState,
        current_time: float,
        delta_time: float,
    ) -> bool:
        """
        Update a single node's position.
        
        Returns True if the node moved.
        """
        if state.state == MobilityState.PAUSED:
            # Check if pause is over
            if current_time >= state.pause_end_time:
                state.state = MobilityState.MOVING
                self._assign_new_waypoint(state)
            return False
        
        # Node is moving
        if state.waypoint is None:
            self._assign_new_waypoint(state)
            return False
        
        # Calculate movement
        distance_to_target = state.distance_to_waypoint()
        max_distance = state.waypoint.speed * delta_time
        
        if distance_to_target <= max_distance:
            # Arrived at waypoint
            state.x = state.waypoint.x
            state.y = state.waypoint.y
            
            # Start pause
            state.state = MobilityState.PAUSED
            state.pause_end_time = current_time + state.waypoint.pause_duration
            state.waypoint = None
            
            return True
        
        # Move toward waypoint
        direction_x = (state.waypoint.x - state.x) / distance_to_target
        direction_y = (state.waypoint.y - state.y) / distance_to_target
        
        state.x += direction_x * max_distance
        state.y += direction_y * max_distance
        
        return True
    
    def _assign_new_waypoint(self, state: MobileNodeState) -> None:
        """Assign a new random waypoint within the operating zone."""
        # Operating zone is the entire simulation area
        # This allows nodes to transit between incident and coordination zones
        zone = self.parameters.simulation_area
        
        # Generate random destination
        target_x = self._rng.uniform(zone.origin_x, zone.origin_x + zone.width_m)
        target_y = self._rng.uniform(zone.origin_y, zone.origin_y + zone.height_m)
        
        # Generate random speed (avoiding zero to ensure movement)
        # Use minimum of 1 m/s to avoid very slow nodes
        speed = self._rng.uniform(max(1.0, self.speed_min), self.speed_max)
        
        # Generate random pause duration
        pause = self._rng.uniform(self.pause_min, self.pause_max)
        
        state.waypoint = Waypoint(
            x=target_x,
            y=target_y,
            speed=speed,
            pause_duration=pause,
        )
        state.state = MobilityState.MOVING
    
    def get_position(self, node_id: str) -> tuple[float, float] | None:
        """Get current position of a node."""
        state = self._node_states.get(node_id)
        if state is None:
            return None
        return (state.x, state.y)
    
    def get_all_positions(self) -> dict[str, tuple[float, float]]:
        """Get positions of all mobile nodes."""
        return {
            node_id: (state.x, state.y)
            for node_id, state in self._node_states.items()
        }
    
    def get_node_ids(self) -> list[str]:
        """Get list of managed node IDs."""
        return list(self._node_states.keys())


def calculate_encounters(
    positions: dict[str, tuple[float, float]],
    radio_range: float,
) -> list[tuple[str, str, float]]:
    """
    Calculate which node pairs are within communication range.
    
    Args:
        positions: Mapping of node_id to (x, y) position
        radio_range: Communication range in metres
        
    Returns:
        List of (node_a, node_b, distance) tuples for nodes within range
    """
    encounters = []
    node_ids = list(positions.keys())
    
    for i, node_a in enumerate(node_ids):
        pos_a = positions[node_a]
        
        for node_b in node_ids[i + 1:]:
            pos_b = positions[node_b]
            
            # Calculate Euclidean distance
            dx = pos_b[0] - pos_a[0]
            dy = pos_b[1] - pos_a[1]
            distance = np.sqrt(dx * dx + dy * dy)
            
            if distance <= radio_range:
                encounters.append((node_a, node_b, distance))
    
    return encounters


def update_topology_edges(
    topology,  # NetworkTopology - avoid circular import
    all_positions: dict[str, tuple[float, float]],
    radio_range: float,
) -> list[tuple[str, str]]:
    """
    Update topology graph edges based on current node positions.
    
    Removes edges for nodes that moved out of range and adds edges
    for nodes that moved into range.
    
    Args:
        topology: NetworkTopology to update
        all_positions: Current positions of all nodes
        radio_range: Communication range in metres
        
    Returns:
        List of new edges (node pairs that just came into range)
    """
    import networkx as nx
    
    new_encounters = []
    node_ids = list(all_positions.keys())
    
    # Calculate current distances and update edges
    for i, node_a in enumerate(node_ids):
        pos_a = all_positions[node_a]
        
        for node_b in node_ids[i + 1:]:
            pos_b = all_positions[node_b]
            
            # Calculate distance
            dx = pos_b[0] - pos_a[0]
            dy = pos_b[1] - pos_a[1]
            distance = np.sqrt(dx * dx + dy * dy)
            
            currently_connected = topology.graph.has_edge(node_a, node_b)
            should_be_connected = distance <= radio_range
            
            if should_be_connected and not currently_connected:
                # New encounter - add edge
                topology.graph.add_edge(node_a, node_b, distance=distance)
                new_encounters.append((node_a, node_b))
                
            elif not should_be_connected and currently_connected:
                # Nodes moved apart - remove edge
                topology.graph.remove_edge(node_a, node_b)
                
            elif should_be_connected and currently_connected:
                # Update distance weight
                topology.graph[node_a][node_b]['distance'] = distance
    
    return new_encounters
