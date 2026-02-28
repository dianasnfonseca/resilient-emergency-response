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

Sources:
    - Ullah & Qayyum (2022): Network parameters
    - Karaman et al. (2026): Connectivity scenarios
    - Keykhaei et al. (2024): Node placement
"""

from ercs.network.topology import (
    NetworkTopology,
    Node,
    NodeType,
    TopologyGenerator,
    generate_topology,
)

__all__ = [
    "NetworkTopology",
    "Node",
    "NodeType",
    "TopologyGenerator",
    "generate_topology",
]