"""
Coordination Layer (Phase 4).

This module implements task assignment algorithms for emergency response
coordination, comparing adaptive (network-aware) and baseline (proximity-only)
approaches.

Classes:
    CoordinatorBase: Abstract base class for coordinators
    AdaptiveCoordinator: Network-aware urgency-first assignment
    BaselineCoordinator: Proximity-only FCFS assignment
    CoordinationManager: Manages coordination cycles
    Assignment: Task assignment record
    CoordinationEvent: Event log entry

Protocols:
    NetworkStateProvider: Interface for network state queries
    ResponderLocator: Interface for responder location queries

Enums:
    EventType: Types of coordination events

Factory Functions:
    create_coordinator: Create coordinator by algorithm type

"""

from ercs.coordination.algorithms import (
    AdaptiveCoordinator,
    Assignment,
    BaselineCoordinator,
    CoordinationEvent,
    CoordinationManager,
    CoordinatorBase,
    EventType,
    NetworkStateProvider,
    ResponderLocator,
    create_coordinator,
)

__all__ = [
    "AdaptiveCoordinator",
    "Assignment",
    "BaselineCoordinator",
    "CoordinationEvent",
    "CoordinationManager",
    "CoordinatorBase",
    "EventType",
    "NetworkStateProvider",
    "ResponderLocator",
    "create_coordinator",
]