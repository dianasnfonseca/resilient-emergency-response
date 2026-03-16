"""
Role-Based Random Waypoint Mobility Model (Phase 1 Extension).

This module implements node mobility for the emergency response simulation,
enabling dynamic network topology changes essential for PRoPHET protocol
operation and realistic DTN evaluation.

Two mobility models are supported:

**Random Waypoint** (original):
  All nodes select random destinations uniformly across the entire
  simulation area with identical speed ranges and pause times.

**Role-Based Random Waypoint** (default):
  Nodes are assigned one of three roles — RESCUE, TRANSPORT, LIAISON —
  each with a distinct waypoint zone and speed range.  This creates
  heterogeneous encounter patterns that PRoPHET can exploit, allowing
  the Adaptive algorithm to differentiate between responders.

Role definitions:
  - RESCUE (~60%): confined to incident zone, slow (1-5 m/s),
    long pauses — rarely encounter coordination nodes.
  - TRANSPORT (~25%): shuttle between incident and coordination zones,
    fast (5-20 m/s), moderate pauses — frequent coord encounters.
  - LIAISON (~15%): free movement across entire area, medium speed
    (1-10 m/s), brief pauses — variable encounter patterns.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from ercs.config.parameters import NetworkParameters, ResponderRole, ZoneConfig


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
class RoleConfig:
    """Per-role mobility parameters.

    Attributes:
        speed_min: Minimum movement speed (m/s)
        speed_max: Maximum movement speed (m/s)
        pause_min: Minimum pause duration (seconds)
        pause_max: Maximum pause duration (seconds)
        zone_mode: Which zone the role selects waypoints from —
            ``"incident"`` (incident zone only),
            ``"shuttle"`` (alternates incident ↔ coordination), or
            ``"full"`` (entire simulation area).
    """

    speed_min: float
    speed_max: float
    pause_min: float
    pause_max: float
    zone_mode: str  # "incident" | "shuttle" | "full"


def _build_role_configs(params: NetworkParameters) -> dict[ResponderRole, RoleConfig]:
    """Build role configs from NetworkParameters (single source of truth)."""
    return {
        ResponderRole.RESCUE: RoleConfig(
            speed_min=params.role_rescue_speed_min,
            speed_max=params.role_rescue_speed_max,
            pause_min=params.role_rescue_pause_min,
            pause_max=params.role_rescue_pause_max,
            zone_mode="incident",
        ),
        ResponderRole.TRANSPORT: RoleConfig(
            speed_min=params.role_transport_speed_min,
            speed_max=params.role_transport_speed_max,
            pause_min=params.role_transport_pause_min,
            pause_max=params.role_transport_pause_max,
            zone_mode="shuttle",
        ),
        ResponderRole.LIAISON: RoleConfig(
            speed_min=params.role_liaison_speed_min,
            speed_max=params.role_liaison_speed_max,
            pause_min=params.role_liaison_pause_min,
            pause_max=params.role_liaison_pause_max,
            zone_mode="full",
        ),
    }


def _build_role_distribution(params: NetworkParameters) -> dict[ResponderRole, float]:
    """Build role distribution from NetworkParameters (single source of truth)."""
    return {
        ResponderRole.RESCUE: params.role_rescue_fraction,
        ResponderRole.TRANSPORT: params.role_transport_fraction,
        ResponderRole.LIAISON: 1.0
        - params.role_rescue_fraction
        - params.role_transport_fraction,
    }


@dataclass
class MobileNodeState:
    """
    Mobility state for a single node.

    Attributes:
        node_id: Unique identifier
        x: Current X position (metres)
        y: Current Y position (metres)
        role: Responder role (determines waypoint zone and speed)
        state: Current mobility state
        waypoint: Current target waypoint (if moving)
        pause_end_time: When pause ends (if paused)
    """

    node_id: str
    x: float
    y: float
    role: ResponderRole | None = None
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

    Attributes:
        parameters: Network configuration
        speed_min: Minimum movement speed (m/s)
        speed_max: Maximum movement speed (m/s)
        pause_min: Minimum pause duration (seconds)
        pause_max: Maximum pause duration (seconds)
    """

    parameters: NetworkParameters
    speed_min: float = 0.0
    speed_max: float = 20.0
    pause_min: float = 0.0
    pause_max: float = 30.0  # Brief pauses for realism

    # Internal state
    _node_states: dict[str, MobileNodeState] = field(default_factory=dict)
    _rng: np.random.Generator = field(default=None)

    def __post_init__(self) -> None:
        """Initialize random generator and role configs from parameters."""
        if self._rng is None:
            self._rng = np.random.default_rng()
        self._role_configs = _build_role_configs(self.parameters)
        self._role_distribution = _build_role_distribution(self.parameters)

    def initialize(
        self,
        mobile_node_ids: list[str],
        initial_positions: dict[str, tuple[float, float]],
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize mobility states for all mobile nodes.

        Assigns responder roles deterministically based on node index
        (RESCUE → first 60%, TRANSPORT → next 25%, LIAISON → remaining 15%).

        Args:
            mobile_node_ids: IDs of mobile nodes
            initial_positions: Mapping of node_id to (x, y) position
            random_seed: Optional seed for reproducibility
        """
        self._rng = np.random.default_rng(random_seed)
        self._node_states.clear()

        # Deterministic role assignment by index
        roles = _assign_roles(len(mobile_node_ids), self._role_distribution)

        for idx, node_id in enumerate(mobile_node_ids):
            if node_id in initial_positions:
                x, y = initial_positions[node_id]
                state = MobileNodeState(
                    node_id=node_id,
                    x=x,
                    y=y,
                    role=roles[idx],
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
        """Assign a new random waypoint, respecting the node's role."""
        role_cfg = (
            self._role_configs.get(state.role) if state.role is not None else None
        )

        if role_cfg is None:
            # Fallback: original RWP behaviour (full area, manager speeds)
            zone = self.parameters.simulation_area
            speed = self._rng.uniform(max(1.0, self.speed_min), self.speed_max)
            pause = self._rng.uniform(self.pause_min, self.pause_max)
        else:
            zone = self._get_target_zone(state, role_cfg)
            speed = self._rng.uniform(role_cfg.speed_min, role_cfg.speed_max)
            pause = self._rng.uniform(role_cfg.pause_min, role_cfg.pause_max)

        target_x = self._rng.uniform(zone.origin_x, zone.origin_x + zone.width_m)
        target_y = self._rng.uniform(zone.origin_y, zone.origin_y + zone.height_m)

        state.waypoint = Waypoint(
            x=target_x,
            y=target_y,
            speed=speed,
            pause_duration=pause,
        )
        state.state = MobilityState.MOVING

    def _get_target_zone(
        self, state: MobileNodeState, role_cfg: RoleConfig
    ) -> ZoneConfig:
        """Select the target zone for a waypoint based on role."""
        if role_cfg.zone_mode == "incident":
            return self.parameters.incident_zone
        elif role_cfg.zone_mode == "shuttle":
            # Alternate between incident and coordination zone:
            # if currently closer to incident zone, head to coordination,
            # and vice versa.
            iz = self.parameters.incident_zone
            cz = self.parameters.coordination_zone
            iz_cx = iz.origin_x + iz.width_m / 2
            iz_cy = iz.origin_y + iz.height_m / 2
            cz_cx = cz.origin_x + cz.width_m / 2
            cz_cy = cz.origin_y + cz.height_m / 2
            dist_to_iz = np.sqrt((state.x - iz_cx) ** 2 + (state.y - iz_cy) ** 2)
            dist_to_cz = np.sqrt((state.x - cz_cx) ** 2 + (state.y - cz_cy) ** 2)
            return cz if dist_to_iz < dist_to_cz else iz
        else:  # "full"
            return self.parameters.simulation_area

    def get_position(self, node_id: str) -> tuple[float, float] | None:
        """Get current position of a node."""
        state = self._node_states.get(node_id)
        if state is None:
            return None
        return (state.x, state.y)

    def get_all_positions(self) -> dict[str, tuple[float, float]]:
        """Get positions of all mobile nodes."""
        return {
            node_id: (state.x, state.y) for node_id, state in self._node_states.items()
        }

    def get_node_ids(self) -> list[str]:
        """Get list of managed node IDs."""
        return list(self._node_states.keys())


def _assign_roles(
    n_nodes: int,
    role_distribution: dict[ResponderRole, float] | None = None,
) -> list[ResponderRole]:
    """Assign responder roles deterministically by index.

    Distribution (approximate):
      - RESCUE:    first 60% of nodes
      - TRANSPORT: next 25%
      - LIAISON:   remaining 15%

    Args:
        n_nodes: Total number of mobile nodes.
        role_distribution: Fraction per role. Uses default NetworkParameters
            values if None.

    Returns:
        List of roles, one per node index.
    """
    if role_distribution is None:
        role_distribution = _build_role_distribution(NetworkParameters())
    n_rescue = round(n_nodes * role_distribution[ResponderRole.RESCUE])
    n_transport = round(n_nodes * role_distribution[ResponderRole.TRANSPORT])
    # Remainder goes to LIAISON
    n_liaison = n_nodes - n_rescue - n_transport

    roles: list[ResponderRole] = (
        [ResponderRole.RESCUE] * n_rescue
        + [ResponderRole.TRANSPORT] * n_transport
        + [ResponderRole.LIAISON] * n_liaison
    )
    return roles


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

        for node_b in node_ids[i + 1 :]:
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

    new_encounters = []
    node_ids = list(all_positions.keys())

    # Calculate current distances and update edges
    for i, node_a in enumerate(node_ids):
        pos_a = all_positions[node_a]

        for node_b in node_ids[i + 1 :]:
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
                topology.graph[node_a][node_b]["distance"] = distance

    return new_encounters
