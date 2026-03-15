"""
Network Topology Generation (Phase 1).

This module provides network topology generation for emergency response
simulation, implementing the two-zone disaster area model.

Classes:
    NetworkTopology: Network graph with nodes and edges
    TopologyGenerator: Generates topologies with configurable parameters
    Node: Individual network node

Enums:
    NodeType: Classification of nodes (coordination, mobile_responder)

Factory Functions:
    generate_topology: Create a topology with given parameters

"""

from ercs.network.topology import (
    NetworkTopology,
    Node,
    NodeType,
    TopologyGenerator,
    generate_topology,
)

from ercs.network.mobility import (
    MobilityManager,
    MobileNodeState,
    MobilityState,
    Waypoint,
    calculate_encounters,
)

__all__ = [
    "NetworkTopology",
    "Node",
    "NodeType",
    "TopologyGenerator",
    "generate_topology",
    "MobilityManager",
    "MobileNodeState",
    "MobilityState",
    "Waypoint",
    "calculate_encounters",
]