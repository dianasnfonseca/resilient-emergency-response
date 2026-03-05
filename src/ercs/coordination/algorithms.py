"""
Coordination Layer (Phase 4).

This module implements task assignment algorithms for emergency response
coordination, comparing adaptive (network-aware) and baseline (proximity-only)
approaches.

The coordination layer provides:
- Adaptive algorithm: urgency-first ordering with network-aware assignment
- Baseline algorithm: FCFS ordering with proximity-only assignment
- Integration with PRoPHET communication layer for network state queries
- Event logging for performance metric collection

Key differences between algorithms:
- Adaptive: considers communication path availability (P > 0.3 threshold)
- Adaptive: uses static α=0.2, γ_r=0.2, β=0.6 weighting (predictability + recency + proximity)
- Adaptive: prioritises high-urgency tasks
- Baseline: assigns to nearest responder regardless of connectivity
- Baseline: processes tasks in creation order (FCFS)

Sources:
    - Boondirek et al. (2014): Distance-dominant weighting (DiPRoPHET)
    - Cui et al. (2022): Workload-aware scoring (AdaptiveSpray)
    - Kaji et al. (2025): Distributed architecture, 30-minute update interval
    - Rosas et al. (2023): Three priority levels
    - Ullah & Qayyum (2022): P > threshold for path availability
    - Keykhaei et al. (2024): Euclidean distance for proximity
    - Oksuz & Satoglu (2024): Urgency-based prioritisation
    - Cabral et al. (2018): 30-minute response time standard
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from ercs.config.parameters import (
    AlgorithmType,
    CoordinationParameters,
    UrgencyLevel,
)
from ercs.scenario.generator import Task

# Static scoring weights for the adaptive algorithm.
# Source: Boondirek et al. (2014) DiPRoPHET (ICUIMC) — distance as dominant criterion.
# Recency weight added to break P-value saturation ties (Nelson et al., 2009).
PREDICTABILITY_WEIGHT = 0.2  # α
RECENCY_WEIGHT = 0.2  # γ_r (encounter recency)
PROXIMITY_WEIGHT = 0.6  # β

# Workload penalty weight — discourages re-assigning the same responder
# across consecutive coordination cycles, preventing node overload.
# Source: Cui et al. (2022) AdaptiveSpray (China Communications)
WORKLOAD_PENALTY_WEIGHT = 0.2  # λ

# Fixed denominator for distance normalisation — the simulation area diagonal
# sqrt(3000² + 1500²) ≈ 3354.1 m.  Using a fixed denominator avoids the
# relative normalisation problem where D_norm = 1.0 for the closest candidate
# even when all candidates are far from the task.
SIMULATION_AREA_DIAGONAL_M = 3354.1


class NetworkStateProvider(Protocol):
    """Protocol for querying network state from communication layer."""

    def get_delivery_predictability(self, from_node: str, to_node: str) -> float:
        """Get delivery predictability P(from_node, to_node)."""
        ...

    def get_last_encounter_time(self, from_node: str, to_node: str) -> float:
        """Get time of last direct encounter between two nodes (0.0 if never)."""
        ...


class ResponderLocator(Protocol):
    """Protocol for getting responder locations."""

    def get_responder_position(self, responder_id: str) -> tuple[float, float]:
        """Get (x, y) position of a responder."""
        ...

    def get_all_responder_ids(self) -> list[str]:
        """Get list of all responder node IDs."""
        ...


class EventType(str, Enum):
    """Types of coordination events for logging."""

    TASK_ASSIGNED = "task_assigned"
    TASK_REASSIGNED = "task_reassigned"
    ASSIGNMENT_FAILED = "assignment_failed"
    COORDINATION_CYCLE = "coordination_cycle"


@dataclass
class CoordinationEvent:
    """
    Record of a coordination event for metric collection.

    Events are logged during coordination algorithm execution to enable
    performance analysis in Phase 6.

    Attributes:
        event_type: Type of event
        timestamp: Simulation time when event occurred
        task_id: Associated task ID (if applicable)
        responder_id: Associated responder ID (if applicable)
        details: Additional event-specific information
    """

    event_type: EventType
    timestamp: float
    task_id: str | None = None
    responder_id: str | None = None
    details: dict = field(default_factory=dict)


@dataclass
class Assignment:
    """
    Represents a task assignment to a responder.

    Attributes:
        task: The assigned task
        responder_id: ID of the assigned responder
        assignment_time: Simulation time of assignment
        distance: Distance from responder to task location
        predictability: Delivery predictability to responder (adaptive only)
    """

    task: Task
    responder_id: str
    assignment_time: float
    distance: float
    predictability: float | None = None

    @property
    def task_id(self) -> str:
        """Get the task ID."""
        return self.task.task_id


class CoordinatorBase(ABC):
    """
    Abstract base class for coordination algorithms.

    Provides common functionality for task assignment including:
    - Distance calculations
    - Event logging
    - Assignment tracking

    Subclasses implement specific assignment logic through the
    _select_responder method.

    Attributes:
        params: Coordination parameters
        algorithm_type: Type identifier for this algorithm
    """

    def __init__(
        self,
        params: CoordinationParameters | None = None,
        algorithm_type: AlgorithmType = AlgorithmType.BASELINE,
    ):
        """
        Initialize the coordinator.

        Args:
            params: Coordination parameters (uses defaults if None)
            algorithm_type: Algorithm type identifier
        """
        self.params = params or CoordinationParameters()
        self.algorithm_type = algorithm_type

        # Event log
        self._events: list[CoordinationEvent] = []

        # Assignment tracking
        self._assignments: dict[str, Assignment] = {}  # task_id -> Assignment
        self._responder_assignments: dict[str, list[str]] = (
            {}
        )  # responder -> [task_ids]

        # Statistics
        self._total_assignments = 0
        self._failed_assignments = 0

    @abstractmethod
    def assign_tasks(
        self,
        tasks: list[Task],
        responder_locator: ResponderLocator,
        network_state: NetworkStateProvider | None,
        coordination_node: str,
        current_time: float,
        all_coordination_nodes: list[str] | None = None,
    ) -> list[Assignment]:
        """
        Assign pending tasks to responders.

        Args:
            tasks: List of tasks to assign
            responder_locator: Provider of responder locations
            network_state: Provider of network state (for adaptive algorithm)
            coordination_node: ID of the coordination node making assignments
            current_time: Current simulation time
            all_coordination_nodes: All coordination node IDs for multi-node
                P-value queries (Adaptive uses max P across all coord nodes)

        Returns:
            List of new assignments made
        """
        pass

    @abstractmethod
    def _select_responder(
        self,
        task: Task,
        responder_locator: ResponderLocator,
        network_state: NetworkStateProvider | None,
        coordination_node: str,
        current_time: float = 0.0,
        all_coordination_nodes: list[str] | None = None,
    ) -> tuple[str | None, float, float | None]:
        """
        Select the best responder for a task.

        Args:
            task: Task to assign
            responder_locator: Provider of responder locations
            network_state: Provider of network state
            coordination_node: Coordination node making the decision
            current_time: Current simulation time
            all_coordination_nodes: All coordination node IDs for multi-node queries

        Returns:
            Tuple of (responder_id, distance, predictability) or (None, 0, None) if no suitable responder
        """
        pass

    def _calculate_distance(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        """
        Calculate Euclidean distance between two points.

        Source: Keykhaei et al. (2024) - standard practice for simulation

        Args:
            x1, y1: First point coordinates
            x2, y2: Second point coordinates

        Returns:
            Euclidean distance in metres
        """
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _log_event(
        self,
        event_type: EventType,
        timestamp: float,
        task_id: str | None = None,
        responder_id: str | None = None,
        **details,
    ) -> None:
        """Log a coordination event."""
        event = CoordinationEvent(
            event_type=event_type,
            timestamp=timestamp,
            task_id=task_id,
            responder_id=responder_id,
            details=details,
        )
        self._events.append(event)

    def _record_assignment(self, assignment: Assignment) -> None:
        """Record an assignment for tracking."""
        self._assignments[assignment.task_id] = assignment

        if assignment.responder_id not in self._responder_assignments:
            self._responder_assignments[assignment.responder_id] = []
        self._responder_assignments[assignment.responder_id].append(assignment.task_id)

        self._total_assignments += 1

    def get_assignment(self, task_id: str) -> Assignment | None:
        """Get assignment for a task."""
        return self._assignments.get(task_id)

    def get_responder_tasks(self, responder_id: str) -> list[str]:
        """Get task IDs assigned to a responder."""
        return self._responder_assignments.get(responder_id, [])

    def get_events(self) -> list[CoordinationEvent]:
        """Get all logged events."""
        return list(self._events)

    def get_events_by_type(self, event_type: EventType) -> list[CoordinationEvent]:
        """Get events of a specific type."""
        return [e for e in self._events if e.event_type == event_type]

    @property
    def statistics(self) -> dict:
        """Get coordination statistics."""
        return {
            "algorithm_type": self.algorithm_type.value,
            "total_assignments": self._total_assignments,
            "failed_assignments": self._failed_assignments,
            "total_events": len(self._events),
            "unique_responders_used": len(self._responder_assignments),
        }

    def reset(self) -> None:
        """Reset coordinator state."""
        self._events.clear()
        self._assignments.clear()
        self._responder_assignments.clear()
        self._total_assignments = 0
        self._failed_assignments = 0


class AdaptiveCoordinator(CoordinatorBase):
    """
    Adaptive coordination algorithm with network awareness.

    The adaptive algorithm:
    1. Processes tasks in urgency order (High > Medium > Low)
    2. Considers only responders with available communication paths (P > 0.3)
    3. Calculates a weighted score combining predictability, recency, and proximity
    4. Selects the responder with highest combined score

    Score = α × P_abs + γ_r × R_norm + β × D_norm − λ × W_penalty

    - P_abs: absolute delivery predictability (already in [0, 1])
    - R_norm: encounter recency = 1 − min(Δt / 1800, 1.0), where Δt is time
      since last direct encounter with the coordination node
    - D_norm: 1 - (distance / area_diagonal), using the fixed simulation diagonal
    - W_penalty: 1 if the responder was assigned in the prior cycle, else 0
    - α = 0.2, γ_r = 0.2, β = 0.6, λ = 0.2

    Encounter recency addresses P-value saturation: under Random Waypoint
    mobility, PRoPHET P values converge to near-uniform ~0.45–0.50 for all
    encountered responders (Boondirek et al., 2014; Nelson et al., 2009).
    The recency signal discriminates between recently-contacted nodes
    (high confidence in current position/state) and stale contacts.

    Sources:
        - Boondirek et al. (2014): Distance-dominant weighting (DiPRoPHET)
        - Nelson et al. (2009): Encounter-based routing, recency as signal
        - Shah & Ahmed (2025): Absolute DP values (SN Computer Science)
        - Cui et al. (2022): Workload-aware scoring (AdaptiveSpray)
        - Kaji et al. (2025): Urgency-first prioritisation, 30-min interval
        - Ullah & Qayyum (2022): P > threshold, predictability-based selection
    """

    def __init__(self, params: CoordinationParameters | None = None):
        """Initialize the adaptive coordinator."""
        super().__init__(params=params, algorithm_type=AlgorithmType.ADAPTIVE)

        # Urgency ordering for task prioritisation
        self._urgency_order = {
            UrgencyLevel.HIGH: 0,
            UrgencyLevel.MEDIUM: 1,
            UrgencyLevel.LOW: 2,
        }

    def assign_tasks(
        self,
        tasks: list[Task],
        responder_locator: ResponderLocator,
        network_state: NetworkStateProvider | None,
        coordination_node: str,
        current_time: float,
        all_coordination_nodes: list[str] | None = None,
    ) -> list[Assignment]:
        """
        Assign tasks using adaptive algorithm.

        Tasks are sorted by urgency (high first), then by creation time.
        Only responders with communication paths (P > 0) are considered.

        Args:
            tasks: List of pending tasks
            responder_locator: Provider of responder locations
            network_state: Provider of network state (required for adaptive)
            coordination_node: Coordination node making assignments
            current_time: Current simulation time
            all_coordination_nodes: All coordination node IDs for multi-node
                P-value queries (uses max P across all coord nodes)

        Returns:
            List of new assignments
        """
        if network_state is None:
            raise ValueError("Adaptive algorithm requires network_state provider")

        # Log coordination cycle
        self._log_event(
            EventType.COORDINATION_CYCLE,
            current_time,
            pending_tasks=len(tasks),
            algorithm="adaptive",
        )

        # Sort tasks: urgency first, then creation time
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (self._urgency_order[t.urgency], t.creation_time),
        )

        assignments = []

        for task in sorted_tasks:
            # Skip already assigned tasks
            if task.task_id in self._assignments:
                continue

            # Select responder
            responder_id, distance, predictability = self._select_responder(
                task, responder_locator, network_state, coordination_node,
                current_time=current_time,
                all_coordination_nodes=all_coordination_nodes,
            )

            if responder_id is not None:
                assignment = Assignment(
                    task=task,
                    responder_id=responder_id,
                    assignment_time=current_time,
                    distance=distance,
                    predictability=predictability,
                )

                # Update task status
                task.assign(responder_id, current_time)

                # Record assignment
                self._record_assignment(assignment)
                assignments.append(assignment)

                # Compute recency for this assignment (for logging)
                T_REF = 1800.0
                coord_nodes_to_query = all_coordination_nodes or [coordination_node]
                last_enc = max(
                    network_state.get_last_encounter_time(cn, responder_id)
                    for cn in coord_nodes_to_query
                )
                delta_t = max(0.0, current_time - last_enc)
                recency = 1.0 - min(delta_t / T_REF, 1.0)

                # Log event
                self._log_event(
                    EventType.TASK_ASSIGNED,
                    current_time,
                    task_id=task.task_id,
                    responder_id=responder_id,
                    urgency=task.urgency.value,
                    distance=distance,
                    predictability=predictability,
                    recency=recency,
                )
            else:
                # No suitable responder found
                self._failed_assignments += 1
                self._log_event(
                    EventType.ASSIGNMENT_FAILED,
                    current_time,
                    task_id=task.task_id,
                    reason="no_reachable_responder",
                    urgency=task.urgency.value,
                )

        return assignments

    def _select_responder(
        self,
        task: Task,
        responder_locator: ResponderLocator,
        network_state: NetworkStateProvider | None,
        coordination_node: str,
        current_time: float = 0.0,
        all_coordination_nodes: list[str] | None = None,
    ) -> tuple[str | None, float, float | None]:
        """
        Select responder using weighted score of predictability, recency, and proximity.

        Considers only responders with P > threshold (available communication
        path), then calculates a combined score:

        Score = α × P_abs + γ_r × R_norm + β × D_norm − λ × W_penalty

        where R_norm = 1 − min(Δt / T_REF, 1.0) and T_REF = 1800 s (i_typ).

        When multiple coordination nodes are available, uses max P and most
        recent encounter across all coord nodes to avoid single-node isolation.

        Sources:
            Shah & Ahmed (2025) — absolute DP values
            Boondirek et al. (2014) — distance-dominant weighting
            Cui et al. (2022) — workload-aware scoring
            Nelson et al. (2009) — encounter recency as routing signal

        Args:
            task: Task to assign
            responder_locator: Provider of responder locations
            network_state: Provider of network state
            coordination_node: Primary coordination node
            current_time: Current simulation time (for recency calculation)
            all_coordination_nodes: All coordination node IDs for multi-node
                P-value queries (uses max P across all coord nodes)

        Returns:
            (responder_id, distance, predictability) or (None, 0, None)
        """
        if network_state is None:
            return None, 0.0, None

        responders = responder_locator.get_all_responder_ids()
        threshold = self.params.available_path_threshold

        # T_REF matches PRoPHET i_typ parameter (seconds) — inter-encounter half-life
        T_REF = 1800.0

        # Use all coord nodes for P queries to avoid single-node isolation
        coord_nodes_to_query = all_coordination_nodes or [coordination_node]

        # First pass: collect all reachable candidates with their metrics
        candidates = []

        for responder_id in responders:
            # Check communication path availability using max P across all coord nodes
            predictability = max(
                network_state.get_delivery_predictability(cn, responder_id)
                for cn in coord_nodes_to_query
            )

            if predictability <= threshold:
                continue  # No available path

            # Calculate distance to task
            rx, ry = responder_locator.get_responder_position(responder_id)
            distance = self._calculate_distance(
                task.target_location_x,
                task.target_location_y,
                rx,
                ry,
            )

            # Encounter recency: use most recent encounter across all coord nodes
            last_enc = max(
                network_state.get_last_encounter_time(cn, responder_id)
                for cn in coord_nodes_to_query
            )
            delta_t = max(0.0, current_time - last_enc)
            r_norm = 1.0 - min(delta_t / T_REF, 1.0)

            candidates.append({
                "id": responder_id,
                "predictability": predictability,
                "distance": distance,
                "recency": r_norm,
                "last_encounter": last_enc,
            })

        # No reachable candidates
        if not candidates:
            return None, 0.0, None

        # Static weights
        alpha = self.params.predictability_weight   # 0.2
        gamma_r = self.params.recency_weight        # 0.2
        beta = self.params.proximity_weight          # 0.6

        # Second pass: calculate weighted scores and select best
        best_responder = None
        best_score = -1.0
        best_distance = 0.0
        best_predictability = 0.0

        for candidate in candidates:
            # Absolute P directly (already in [0, 1])
            p_abs = candidate["predictability"]

            # Encounter recency
            r_norm = candidate["recency"]

            # Normalise distance against fixed simulation diagonal
            d_norm = 1.0 - (candidate["distance"] / SIMULATION_AREA_DIAGONAL_M)

            # Workload penalty — discourage re-assigning same responder
            w_penalty = 1.0 if candidate["id"] in self._responder_assignments else 0.0

            # Calculate weighted score
            score = (
                alpha * p_abs
                + gamma_r * r_norm
                + beta * d_norm
                - WORKLOAD_PENALTY_WEIGHT * w_penalty
            )

            if score > best_score:
                best_score = score
                best_responder = candidate["id"]
                best_distance = candidate["distance"]
                best_predictability = candidate["predictability"]

        return best_responder, best_distance, best_predictability


class BaselineCoordinator(CoordinatorBase):
    """
    Baseline coordination algorithm with proximity-only assignment.

    The baseline algorithm:
    1. Processes tasks in creation order (FCFS)
    2. Considers all responders regardless of connectivity
    3. Selects the nearest responder by Euclidean distance

    This algorithm represents conventional nearest-neighbour assignment
    without network awareness, serving as the control condition for
    evaluating the adaptive algorithm.

    Sources:
        - Design decision: FCFS as standard baseline approach
        - Keykhaei et al. (2024): Euclidean distance
    """

    def __init__(self, params: CoordinationParameters | None = None):
        """Initialize the baseline coordinator."""
        super().__init__(params=params, algorithm_type=AlgorithmType.BASELINE)

    def assign_tasks(
        self,
        tasks: list[Task],
        responder_locator: ResponderLocator,
        network_state: NetworkStateProvider | None,
        coordination_node: str,
        current_time: float,
        all_coordination_nodes: list[str] | None = None,  # noqa: ARG002 - baseline ignores
    ) -> list[Assignment]:
        """
        Assign tasks using baseline algorithm.

        Tasks are processed in creation order (FCFS).
        All responders are considered regardless of connectivity.

        Args:
            tasks: List of pending tasks
            responder_locator: Provider of responder locations
            network_state: Not used by baseline (can be None)
            coordination_node: Coordination node making assignments
            current_time: Current simulation time
            all_coordination_nodes: Not used by baseline

        Returns:
            List of new assignments
        """
        # Log coordination cycle
        self._log_event(
            EventType.COORDINATION_CYCLE,
            current_time,
            pending_tasks=len(tasks),
            algorithm="baseline",
        )

        # Sort tasks by creation time only (FCFS)
        sorted_tasks = sorted(tasks, key=lambda t: t.creation_time)

        assignments = []

        for task in sorted_tasks:
            # Skip already assigned tasks
            if task.task_id in self._assignments:
                continue

            # Select responder (proximity only)
            responder_id, distance, _ = self._select_responder(
                task, responder_locator, network_state, coordination_node,
                current_time=current_time,
            )

            if responder_id is not None:
                assignment = Assignment(
                    task=task,
                    responder_id=responder_id,
                    assignment_time=current_time,
                    distance=distance,
                    predictability=None,  # Not tracked by baseline
                )

                # Update task status
                task.assign(responder_id, current_time)

                # Record assignment
                self._record_assignment(assignment)
                assignments.append(assignment)

                # Log event
                self._log_event(
                    EventType.TASK_ASSIGNED,
                    current_time,
                    task_id=task.task_id,
                    responder_id=responder_id,
                    urgency=task.urgency.value,
                    distance=distance,
                )
            else:
                # No responder available (shouldn't happen with baseline)
                self._failed_assignments += 1
                self._log_event(
                    EventType.ASSIGNMENT_FAILED,
                    current_time,
                    task_id=task.task_id,
                    reason="no_responder_available",
                )

        return assignments

    def _select_responder(
        self,
        task: Task,
        responder_locator: ResponderLocator,
        network_state: NetworkStateProvider | None,  # noqa: ARG002 - baseline ignores
        coordination_node: str,  # noqa: ARG002 - baseline ignores
        current_time: float = 0.0,  # noqa: ARG002 - baseline ignores
        all_coordination_nodes: list[str] | None = None,  # noqa: ARG002 - baseline ignores
    ) -> tuple[str | None, float, float | None]:
        """
        Select nearest responder by proximity only.

        Does not consider network state - all responders are eligible.

        Args:
            task: Task to assign
            responder_locator: Provider of responder locations
            network_state: Ignored by baseline
            coordination_node: Not used
            current_time: Not used by baseline
            all_coordination_nodes: Not used by baseline

        Returns:
            (responder_id, distance, None)
        """
        responders = responder_locator.get_all_responder_ids()

        if not responders:
            return None, 0.0, None

        best_responder = None
        best_distance = float("inf")

        for responder_id in responders:
            rx, ry = responder_locator.get_responder_position(responder_id)
            distance = self._calculate_distance(
                task.target_location_x,
                task.target_location_y,
                rx,
                ry,
            )

            if distance < best_distance:
                best_distance = distance
                best_responder = responder_id

        if best_responder is not None:
            return best_responder, best_distance, None
        return None, 0.0, None


class CoordinationManager:
    """
    Manages coordination across multiple coordination cycles.

    The manager handles:
    - Periodic coordination updates (default: 30 minutes)
    - Task queue management
    - Integration with communication layer for message sending
    - Performance tracking across the simulation

    Sources:
        - Kaji et al. (2025): 30-minute update interval
        - Cabral et al. (2018): Response time standards
    """

    def __init__(
        self,
        coordinator: CoordinatorBase,
        params: CoordinationParameters | None = None,
    ):
        """
        Initialize the coordination manager.

        Args:
            coordinator: Coordination algorithm to use
            params: Coordination parameters
        """
        self.coordinator = coordinator
        self.params = params or CoordinationParameters()

        # Timing
        self._last_update_time = 0.0
        self._update_interval = self.params.update_interval_seconds

        # Task management
        self._pending_tasks: list[Task] = []
        self._all_assignments: list[Assignment] = []

        # Cycle tracking
        self._cycle_count = 0

    def add_task(self, task: Task) -> None:
        """Add a task to the pending queue."""
        self._pending_tasks.append(task)

    def add_tasks(self, tasks: list[Task]) -> None:
        """Add multiple tasks to the pending queue."""
        self._pending_tasks.extend(tasks)

    def should_update(self, current_time: float) -> bool:
        """Check if a coordination update is due."""
        return current_time >= self._last_update_time + self._update_interval

    def run_coordination_cycle(
        self,
        responder_locator: ResponderLocator,
        network_state: NetworkStateProvider | None,
        coordination_node: str,
        current_time: float,
        all_coordination_nodes: list[str] | None = None,
    ) -> list[Assignment]:
        """
        Run a coordination cycle to assign pending tasks.

        Args:
            responder_locator: Provider of responder locations
            network_state: Provider of network state
            coordination_node: Coordination node making assignments
            current_time: Current simulation time
            all_coordination_nodes: All coordination node IDs for multi-node
                P-value queries (passed through to coordinator)

        Returns:
            List of new assignments made in this cycle
        """
        # Get pending tasks that have arrived by current time
        tasks_to_assign = [
            t
            for t in self._pending_tasks
            if t.creation_time <= current_time and t.is_pending()
        ]

        if not tasks_to_assign:
            return []

        # Run assignment algorithm
        assignments = self.coordinator.assign_tasks(
            tasks=tasks_to_assign,
            responder_locator=responder_locator,
            network_state=network_state,
            coordination_node=coordination_node,
            current_time=current_time,
            all_coordination_nodes=all_coordination_nodes,
        )

        # Track assignments
        self._all_assignments.extend(assignments)

        # Remove assigned tasks from pending
        assigned_ids = {a.task_id for a in assignments}
        self._pending_tasks = [
            t for t in self._pending_tasks if t.task_id not in assigned_ids
        ]

        # Update timing
        self._last_update_time = current_time
        self._cycle_count += 1

        return assignments

    def get_pending_count(self) -> int:
        """Get number of pending tasks."""
        return len(self._pending_tasks)

    def get_all_assignments(self) -> list[Assignment]:
        """Get all assignments made."""
        return list(self._all_assignments)

    @property
    def statistics(self) -> dict:
        """Get manager statistics."""
        coordinator_stats = self.coordinator.statistics
        return {
            **coordinator_stats,
            "cycles_completed": self._cycle_count,
            "pending_tasks": len(self._pending_tasks),
            "total_assignments_made": len(self._all_assignments),
        }

    def reset(self) -> None:
        """Reset manager state."""
        self.coordinator.reset()
        self._pending_tasks.clear()
        self._all_assignments.clear()
        self._last_update_time = 0.0
        self._cycle_count = 0


def create_coordinator(
    algorithm_type: AlgorithmType | str,
    params: CoordinationParameters | None = None,
) -> CoordinatorBase:
    """
    Factory function to create a coordinator.

    Args:
        algorithm_type: Type of algorithm ("adaptive" or "baseline")
        params: Coordination parameters

    Returns:
        Appropriate coordinator instance

    Example:
        >>> from ercs.coordination import create_coordinator
        >>> coordinator = create_coordinator("adaptive")
    """
    if isinstance(algorithm_type, str):
        algorithm_type = AlgorithmType(algorithm_type)

    if algorithm_type == AlgorithmType.ADAPTIVE:
        return AdaptiveCoordinator(params)
    else:
        return BaselineCoordinator(params)
