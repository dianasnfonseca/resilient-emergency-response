"""
Simulation Engine (Phase 5) with Random Waypoint Mobility.

This module implements the discrete event simulation engine that integrates
all components from Phases 1-4:
- Network topology (Phase 1)
- PRoPHET communication layer (Phase 2)
- Scenario generation (Phase 3)
- Coordination algorithms (Phase 4)
- Random Waypoint mobility (Phase 1 extension)

The simulation engine provides:
- Event-driven simulation loop
- Component integration and coordination
- Node mobility with dynamic edge updates
- Results collection for Phase 6 analysis
- Reproducible experimental execution

Sources:
    - Ullah & Qayyum (2022): Simulation duration (6000s), Random Waypoint mobility
    - Law (2015): Statistical design (30 runs per configuration)
    - Kaji et al. (2025): 30-minute coordination update interval
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

from ercs.communication.prophet import (
    CommunicationLayer,
    MessageStatus,
    MessageType,
)
from ercs.config.parameters import (
    AlgorithmType,
    SimulationConfig,
)
from ercs.coordination.algorithms import (
    CoordinationManager,
    CoordinatorBase,
    NetworkStateProvider,
    ResponderLocator,
    create_coordinator,
)
from ercs.network.topology import NetworkTopology, generate_topology
from ercs.network.mobility import MobilityManager
from ercs.scenario.generator import Scenario, ScenarioGenerator, Task


class SimulationEventType(str, Enum):
    """Types of events in the simulation."""

    SIMULATION_START = "simulation_start"
    SIMULATION_END = "simulation_end"
    WARMUP_END = "warmup_end"
    TASK_CREATED = "task_created"
    COORDINATION_CYCLE = "coordination_cycle"
    TASK_ASSIGNED = "task_assigned"
    MESSAGE_CREATED = "message_created"
    MESSAGE_DELIVERED = "message_delivered"
    MESSAGE_EXPIRED = "message_expired"
    NODE_ENCOUNTER = "node_encounter"
    CONNECTIVITY_UPDATE = "connectivity_update"
    MOBILITY_UPDATE = "mobility_update"


@dataclass
class SimulationEvent:
    """
    A discrete event in the simulation.

    Attributes:
        event_type: Type of event
        timestamp: When event occurs (simulation time in seconds)
        data: Event-specific data
    """

    event_type: SimulationEventType
    timestamp: float
    data: dict = field(default_factory=dict)

    def __lt__(self, other: "SimulationEvent") -> bool:
        """Enable sorting by timestamp."""
        return self.timestamp < other.timestamp


@dataclass
class SimulationResults:
    """
    Results from a single simulation run.

    Attributes:
        config: Configuration used
        algorithm: Algorithm type used
        connectivity_level: Network connectivity level
        run_number: Run number within configuration
        random_seed: Random seed used

        # Task metrics
        total_tasks: Total tasks generated
        tasks_assigned: Tasks successfully assigned
        tasks_by_urgency: Task counts by urgency level

        # Message metrics
        messages_created: Total messages created
        messages_delivered: Messages successfully delivered
        messages_expired: Messages that expired
        delivery_rate: Successful delivery percentage

        # Timing metrics
        response_times: List of (task_id, response_time) pairs
        average_response_time: Mean response time
        response_time_by_urgency: Response times grouped by urgency

        # Event log
        events: All simulation events
    """

    # Configuration
    config: SimulationConfig
    algorithm: AlgorithmType
    connectivity_level: float
    run_number: int
    random_seed: int | None

    # Task metrics
    total_tasks: int = 0
    tasks_assigned: int = 0
    tasks_by_urgency: dict[str, int] = field(default_factory=dict)

    # Message metrics
    messages_created: int = 0
    messages_delivered: int = 0
    messages_expired: int = 0

    # Timing metrics
    response_times: list[tuple[str, float]] = field(default_factory=list)
    delivery_times: list[tuple[str, float]] = field(default_factory=list)

    # Events
    events: list[SimulationEvent] = field(default_factory=list)

    @property
    def delivery_rate(self) -> float:
        """Calculate message delivery rate."""
        if self.messages_created == 0:
            return 0.0
        return self.messages_delivered / self.messages_created

    @property
    def assignment_rate(self) -> float:
        """Calculate task assignment rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.tasks_assigned / self.total_tasks

    @property
    def average_response_time(self) -> float | None:
        """Calculate average response time (assignment - creation)."""
        if not self.response_times:
            return None
        times = [t for _, t in self.response_times]
        return sum(times) / len(times)

    @property
    def average_delivery_time(self) -> float | None:
        """Calculate average delivery time (delivery - creation)."""
        if not self.delivery_times:
            return None
        times = [t for _, t in self.delivery_times]
        return sum(times) / len(times)

    def summary(self) -> dict:
        """Generate results summary."""
        return {
            "algorithm": self.algorithm.value,
            "connectivity_level": self.connectivity_level,
            "run_number": self.run_number,
            "total_tasks": self.total_tasks,
            "tasks_assigned": self.tasks_assigned,
            "assignment_rate": self.assignment_rate,
            "messages_created": self.messages_created,
            "messages_delivered": self.messages_delivered,
            "delivery_rate": self.delivery_rate,
            "average_response_time": self.average_response_time,
            "average_delivery_time": self.average_delivery_time,
        }


class TopologyAdapter(ResponderLocator, NetworkStateProvider):
    """
    Adapts NetworkTopology to coordinator interfaces.

    Implements both ResponderLocator and NetworkStateProvider protocols
    to bridge the network topology and communication layer with the
    coordination algorithms.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        communication: CommunicationLayer,
    ):
        """
        Initialize the adapter.

        Args:
            topology: Network topology
            communication: Communication layer for network state
        """
        self.topology = topology
        self.communication = communication

    def get_responder_position(self, responder_id: str) -> tuple[float, float]:
        """Get (x, y) position of a responder."""
        position = self.topology.get_node_position(responder_id)
        if position is None:
            raise ValueError(f"Unknown responder: {responder_id}")
        return position

    def get_all_responder_ids(self) -> list[str]:
        """Get list of all mobile responder node IDs."""
        return self.topology.get_mobile_responder_ids()

    def get_delivery_predictability(self, from_node: str, to_node: str) -> float:
        """Get delivery predictability P(from_node, to_node)."""
        return self.communication.get_delivery_predictability(from_node, to_node)


class SimulationEngine:
    """
    Discrete event simulation engine for emergency response coordination.

    Integrates all components from Phases 1-4 into a complete simulation
    that can be run with different algorithms and connectivity levels.

    The engine implements:
    - Event-driven simulation with priority queue
    - Periodic coordination cycles (30-minute intervals)
    - Random Waypoint mobility for mobile nodes
    - Dynamic edge updates based on node positions
    - Node encounters based on proximity (100m radio range)
    - Message delivery tracking via PRoPHET protocol

    Sources:
        - Ullah & Qayyum (2022): 6000s simulation duration, Random Waypoint
        - Kaji et al. (2025): 30-minute coordination updates
        - Law (2015): 30 runs per configuration

    Attributes:
        config: Simulation configuration
        algorithm_type: Coordination algorithm to use
        connectivity_level: Network connectivity level
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        algorithm_type: AlgorithmType = AlgorithmType.ADAPTIVE,
        connectivity_level: float = 0.75,
        random_seed: int | None = None,
    ):
        """
        Initialize the simulation engine.

        Args:
            config: Simulation configuration (uses defaults if None)
            algorithm_type: Algorithm to use (adaptive or baseline)
            connectivity_level: Network connectivity level (0-1)
            random_seed: Random seed for reproducibility
        """
        self.config = config or SimulationConfig()
        self.algorithm_type = algorithm_type
        self.connectivity_level = connectivity_level
        self.random_seed = random_seed

        self._rng = np.random.default_rng(random_seed)

        # Simulation state
        self._current_time = 0.0
        self._event_queue: list[SimulationEvent] = []
        self._is_running = False

        # Components (initialized on run)
        self._topology: NetworkTopology | None = None
        self._communication: CommunicationLayer | None = None
        self._coordinator: CoordinatorBase | None = None
        self._manager: CoordinationManager | None = None
        self._adapter: TopologyAdapter | None = None
        self._scenario: Scenario | None = None
        self._mobility: MobilityManager | None = None

        # Link availability cache (deterministic per node pair per run)
        self._link_availability: dict[tuple[str, str], bool] = {}

        # Active links: tracks currently established connections so that
        # PRoPHET encounter updates fire only on connection-up events,
        # not on every periodic encounter check (RFC 6693).
        self._active_links: set[tuple[str, str]] = set()

        # Results tracking
        self._events: list[SimulationEvent] = []
        self._task_message_map: dict[str, str] = {}  # task_id -> message_id
        self._message_creation_times: dict[str, float] = {}  # message_id -> time

    def run(self, run_number: int = 0) -> SimulationResults:
        """
        Run a single simulation.

        Args:
            run_number: Run number for identification

        Returns:
            SimulationResults with all metrics
        """
        # Initialize components
        self._initialize_components()

        # Create results container
        results = SimulationResults(
            config=self.config,
            algorithm=self.algorithm_type,
            connectivity_level=self.connectivity_level,
            run_number=run_number,
            random_seed=self.random_seed,
        )

        # Schedule initial events
        self._schedule_initial_events()

        # Log simulation start
        self._log_event(SimulationEventType.SIMULATION_START, 0.0)

        # Run simulation loop
        self._is_running = True
        total_duration = self.config.total_simulation_duration

        while self._is_running and self._event_queue:
            # Get next event
            event = self._pop_next_event()

            if event.timestamp > total_duration:
                break

            self._current_time = event.timestamp

            # Process event
            self._process_event(event, results)

        # Log simulation end
        self._log_event(SimulationEventType.SIMULATION_END, self._current_time)

        # Finalize results
        self._finalize_results(results)

        return results

    def _initialize_components(self) -> None:
        """Initialize all simulation components."""
        # Phase 1: Network Topology
        self._topology = generate_topology(
            parameters=self.config.network,
            connectivity_level=self.connectivity_level,
            random_seed=self.random_seed,
        )

        # Phase 2: Communication Layer
        self._communication = CommunicationLayer(
            comm_params=self.config.communication,
            network_params=self.config.network,
            node_ids=self._topology.get_all_node_ids(),
        )

        # Phase 4: Coordination
        self._coordinator = create_coordinator(
            algorithm_type=self.algorithm_type,
            params=self.config.coordination,
        )
        self._manager = CoordinationManager(
            coordinator=self._coordinator,
            params=self.config.coordination,
        )

        # Adapter for coordinator interfaces
        self._adapter = TopologyAdapter(
            topology=self._topology,
            communication=self._communication,
        )

        # Phase 3: Scenario (generate tasks)
        generator = ScenarioGenerator(
            scenario_params=self.config.scenario,
            network_params=self.config.network,
            random_seed=self.random_seed,
        )
        self._scenario = generator.generate(
            connectivity_level=self.connectivity_level,
            coordination_nodes=self._topology.get_coordination_node_ids(),
        )

        # Add all tasks to manager
        self._manager.add_tasks(self._scenario.tasks)

        # Phase 1b: Mobility (Random Waypoint)
        self._initialize_mobility()

        # Reset state
        self._current_time = 0.0
        self._event_queue.clear()
        self._events.clear()
        self._task_message_map.clear()
        self._message_creation_times.clear()
        self._link_availability.clear()
        self._active_links.clear()

    def _initialize_mobility(self) -> None:
        """
        Initialize Random Waypoint mobility for mobile nodes.
        
        Sources:
            - Ullah & Qayyum (2022): Random Waypoint model, speed 0-20 m/s
            - Chapter1_v2: Mobility model selection rationale
        """
        self._mobility = MobilityManager(
            parameters=self.config.network,
            speed_min=1.0,  # Minimum 1 m/s to ensure movement
            speed_max=20.0,  # Source: Ullah & Qayyum (2022)
            pause_min=0.0,
            pause_max=30.0,  # Brief pauses for realism
        )
        
        # Get initial positions from topology
        mobile_ids = self._topology.get_mobile_responder_ids()
        initial_positions = {}
        
        for node_id in mobile_ids:
            pos = self._topology.get_node_position(node_id)
            if pos is not None:
                initial_positions[node_id] = pos
        
        # Initialize mobility manager
        self._mobility.initialize(
            mobile_node_ids=mobile_ids,
            initial_positions=initial_positions,
            random_seed=self.random_seed,
        )

    def _schedule_initial_events(self) -> None:
        """Schedule initial simulation events with warm-up period.

        During warm-up, only mobility and encounter events run so that
        PRoPHET delivery predictability builds through actual node
        encounters.  Task creation and coordination cycles are delayed
        until after the warm-up period ends.
        """
        warmup_duration = self.config.scenario.warmup_period_seconds
        active_start = warmup_duration  # When active simulation begins
        total_duration = self.config.total_simulation_duration

        # Schedule mobility updates throughout ENTIRE simulation (including warm-up)
        mobility_interval = 1.0
        t = mobility_interval
        while t <= total_duration:
            self._schedule_event(
                SimulationEventType.MOBILITY_UPDATE,
                t,
                {"delta_time": mobility_interval},
            )
            t += mobility_interval

        # Schedule node encounter checks throughout ENTIRE simulation
        encounter_interval = 10.0
        t = encounter_interval
        while t <= total_duration:
            self._schedule_event(
                SimulationEventType.NODE_ENCOUNTER,
                t,
                {},
            )
            t += encounter_interval

        # Log warm-up end / active simulation start
        if warmup_duration > 0:
            self._schedule_event(
                SimulationEventType.WARMUP_END,
                warmup_duration,
                {},
            )

        # Schedule coordination cycles ONLY after warm-up
        update_interval = self.config.coordination.update_interval_seconds
        t = active_start
        while t <= total_duration:
            self._schedule_event(
                SimulationEventType.COORDINATION_CYCLE,
                t,
                {"cycle_time": t},
            )
            t += update_interval

        # Schedule task creation events ONLY after warm-up
        # Shift task times to start from active_start
        for task in self._scenario.tasks:
            adjusted_time = active_start + task.creation_time
            if adjusted_time <= total_duration:
                self._schedule_event(
                    SimulationEventType.TASK_CREATED,
                    adjusted_time,
                    {"task_id": task.task_id},
                )

    def _schedule_event(
        self,
        event_type: SimulationEventType,
        timestamp: float,
        data: dict | None = None,
    ) -> None:
        """Schedule an event."""
        event = SimulationEvent(
            event_type=event_type,
            timestamp=timestamp,
            data=data or {},
        )
        # Insert maintaining sorted order
        self._event_queue.append(event)
        self._event_queue.sort()

    def _pop_next_event(self) -> SimulationEvent:
        """Get and remove the next event."""
        return self._event_queue.pop(0)

    def _process_event(
        self,
        event: SimulationEvent,
        results: SimulationResults,
    ) -> None:
        """Process a simulation event."""
        if event.event_type == SimulationEventType.TASK_CREATED:
            self._handle_task_created(event, results)

        elif event.event_type == SimulationEventType.COORDINATION_CYCLE:
            self._handle_coordination_cycle(event, results)

        elif event.event_type == SimulationEventType.MOBILITY_UPDATE:
            self._handle_mobility_update(event, results)

        elif event.event_type == SimulationEventType.NODE_ENCOUNTER:
            self._handle_node_encounters(event, results)

        elif event.event_type == SimulationEventType.WARMUP_END:
            self._handle_warmup_end(event, results)

    def _handle_warmup_end(
        self,
        event: SimulationEvent,
        results: SimulationResults,
    ) -> None:
        """Handle end of warm-up period.

        Logs predictability statistics at the transition from warm-up
        to active simulation so we can verify that PRoPHET encounter
        history built meaningful (non-uniform) values.
        """
        total_nonzero = 0
        total_pairs = 0

        coord_nodes = self._topology.get_coordination_node_ids()
        mobile_nodes = self._topology.get_mobile_responder_ids()

        for coord_id in coord_nodes:
            for mobile_id in mobile_nodes:
                p = self._communication.get_delivery_predictability(
                    coord_id, mobile_id
                )
                total_pairs += 1
                if p > 0:
                    total_nonzero += 1

        coverage_pct = 100 * total_nonzero / total_pairs if total_pairs > 0 else 0

        self._log_event(
            SimulationEventType.WARMUP_END,
            event.timestamp,
            {
                "nonzero_predictabilities": total_nonzero,
                "total_pairs": total_pairs,
                "coverage_pct": coverage_pct,
            },
        )

    def _handle_task_created(
        self,
        event: SimulationEvent,
        results: SimulationResults,
    ) -> None:
        """Handle task creation event."""
        self._log_event(event.event_type, event.timestamp, event.data)
        results.total_tasks += 1

        task_id = event.data["task_id"]
        task = self._get_task(task_id)
        if task:
            urgency = task.urgency.value
            results.tasks_by_urgency[urgency] = (
                results.tasks_by_urgency.get(urgency, 0) + 1
            )

    def _handle_coordination_cycle(
        self,
        event: SimulationEvent,
        results: SimulationResults,
    ) -> None:
        """Handle coordination cycle event."""
        self._log_event(event.event_type, event.timestamp, event.data)

        # Get coordination nodes
        coord_nodes = self._topology.get_coordination_node_ids()
        coord_node = coord_nodes[0] if coord_nodes else "coord_0"

        # Run coordination (uses first coord node for network state queries)
        assignments = self._manager.run_coordination_cycle(
            responder_locator=self._adapter,
            network_state=self._adapter,
            coordination_node=coord_node,
            current_time=event.timestamp,
        )

        # Process assignments — distribute messages across coord nodes (round-robin)
        for i, assignment in enumerate(assignments):
            results.tasks_assigned += 1

            # Record response time
            response_time = event.timestamp - assignment.task.creation_time
            results.response_times.append((assignment.task_id, response_time))

            # Round-robin: distribute message origination across coord nodes
            source_node = coord_nodes[i % len(coord_nodes)]

            # Create assignment message
            message = self._communication.create_message(
                source_id=source_node,
                destination_id=assignment.responder_id,
                message_type=MessageType.COORDINATION,
                payload={
                    "task_id": assignment.task_id,
                    "location": (
                        assignment.task.target_location_x,
                        assignment.task.target_location_y,
                    ),
                    "urgency": assignment.task.urgency.value,
                },
                urgency_level=assignment.task.urgency,
                current_time=event.timestamp,
            )

            # Track message
            self._task_message_map[assignment.task_id] = message.message_id
            self._message_creation_times[message.message_id] = event.timestamp
            results.messages_created += 1

            # Log assignment event
            self._log_event(
                SimulationEventType.TASK_ASSIGNED,
                event.timestamp,
                {
                    "task_id": assignment.task_id,
                    "responder_id": assignment.responder_id,
                    "response_time": response_time,
                },
            )

            self._log_event(
                SimulationEventType.MESSAGE_CREATED,
                event.timestamp,
                {
                    "message_id": message.message_id,
                    "task_id": assignment.task_id,
                },
            )

    def _is_link_available(self, node_a: str, node_b: str) -> bool:
        """
        Determine if a communication link exists between two nodes.

        Models infrastructure damage (Karaman et al., 2026) where some links
        are unavailable due to base station failures. Uses deterministic hash
        so the same node pair has consistent link status within a run.

        At connectivity_level=0.75, ~75% of potential links exist.
        At connectivity_level=0.20, only ~20% exist.
        """
        pair_key = tuple(sorted([node_a, node_b]))

        if pair_key in self._link_availability:
            return self._link_availability[pair_key]

        # Deterministic: hash node pair + run seed
        pair_hash = hash((pair_key, self.random_seed)) % 10000
        threshold = int(self.connectivity_level * 10000)
        available = pair_hash < threshold

        self._link_availability[pair_key] = available
        return available

    def _handle_mobility_update(
        self,
        event: SimulationEvent,
        results: SimulationResults,
    ) -> None:
        """
        Handle mobility update event.
        
        1. Update node positions via MobilityManager
        2. Update topology node positions
        3. Recalculate edges based on new positions
        4. Process any new encounters for PRoPHET
        """
        delta_time = event.data.get("delta_time", 1.0)
        
        # Step 1: Update mobility positions
        moved_nodes = self._mobility.step(
            current_time=event.timestamp,
            delta_time=delta_time,
        )
        
        if not moved_nodes:
            return  # No nodes moved, nothing to update
        
        # Step 2: Update topology with new positions
        all_positions = self._mobility.get_all_positions()
        
        for node_id, (x, y) in all_positions.items():
            self._topology.update_node_position(node_id, x, y)
        
        # Step 3: Recalculate edges and find new connections
        new_connections = self._topology.update_edges_from_positions()
        
        # Step 4: Process new encounters for PRoPHET (connection-up only)
        # Apply connectivity filter: infrastructure damage means some links
        # don't exist (Karaman et al., 2026)
        for node_a, node_b in new_connections:
            if not self._is_link_available(node_a, node_b):
                continue

            link_key = (min(node_a, node_b), max(node_a, node_b))

            if link_key not in self._active_links:
                # Connection-up: full PRoPHET encounter + transitivity
                self._active_links.add(link_key)
                delivered = self._communication.process_encounter(
                    node_a=node_a,
                    node_b=node_b,
                    current_time=event.timestamp,
                )
            else:
                # Already active — transfer messages only
                delivered = self._communication.transfer_messages(
                    node_a=node_a,
                    node_b=node_b,
                    current_time=event.timestamp,
                )

            self._process_delivered_messages(delivered, event.timestamp, results)

    def _handle_node_encounters(
        self,
        event: SimulationEvent,
        results: SimulationResults,
    ) -> None:
        """
        Handle node encounter events — connection-up only for PRoPHET.

        RFC 6693 (Lindgren et al., 2012): PRoPHET encounter updates (the
        P(a,b) equation and transitivity) fire only when a link is first
        established ("connection-up"), not repeatedly while nodes remain
        in range.  Message transfers still occur on all active links at
        each interval.

        Uses the deterministic link availability filter — infrastructure
        damage (Karaman et al., 2026) determines which links exist.
        """
        # Build the current set of link-available edges
        current_links: set[tuple[str, str]] = set()
        for node_a, node_b in self._topology.graph.edges():
            if not self._is_link_available(node_a, node_b):
                continue
            link_key = (min(node_a, node_b), max(node_a, node_b))
            current_links.add(link_key)

        # New links: connection-up → full PRoPHET encounter + transitivity
        new_links = current_links - self._active_links

        for node_a, node_b in new_links:
            delivered = self._communication.process_encounter(
                node_a=node_a,
                node_b=node_b,
                current_time=event.timestamp,
            )
            self._process_delivered_messages(delivered, event.timestamp, results)

        # Existing links: transfer messages only (no encounter/transitivity)
        existing_links = current_links & self._active_links

        for node_a, node_b in existing_links:
            delivered = self._communication.transfer_messages(
                node_a=node_a,
                node_b=node_b,
                current_time=event.timestamp,
            )
            self._process_delivered_messages(delivered, event.timestamp, results)

        # Update active links for next cycle
        self._active_links = current_links

        # Expire old messages
        expired = self._communication.expire_all_messages(event.timestamp)
        results.messages_expired += expired

    def _process_delivered_messages(
        self,
        delivered: list,
        timestamp: float,
        results: SimulationResults,
    ) -> None:
        """Process list of TransmissionResult and update results."""
        for result in delivered:
            if result.reason == "delivered":
                msg = result.message
                results.messages_delivered += 1

                # Calculate delivery time
                creation_time = self._message_creation_times.get(
                    msg.message_id, timestamp
                )
                delivery_time = timestamp - creation_time

                # Find associated task
                task_id = msg.payload.get("task_id") if msg.payload else None
                if task_id:
                    results.delivery_times.append((task_id, delivery_time))

                self._log_event(
                    SimulationEventType.MESSAGE_DELIVERED,
                    timestamp,
                    {
                        "message_id": msg.message_id,
                        "delivery_time": delivery_time,
                    },
                )

    def _get_task(self, task_id: str) -> Task | None:
        """Get task by ID from scenario."""
        for task in self._scenario.tasks:
            if task.task_id == task_id:
                return task
        return None

    def _log_event(
        self,
        event_type: SimulationEventType,
        timestamp: float,
        data: dict | None = None,
    ) -> None:
        """Log a simulation event."""
        event = SimulationEvent(
            event_type=event_type,
            timestamp=timestamp,
            data=data or {},
        )
        self._events.append(event)

    def _finalize_results(self, results: SimulationResults) -> None:
        """Finalize results after simulation."""
        results.events = list(self._events)


class ExperimentRunner:
    """
    Runs complete experiments across all configurations.

    Manages execution of:
    - 2 algorithms × 3 connectivity levels × 30 runs = 180 total runs
    - Results collection and aggregation
    - Progress tracking

    Sources:
        - Law (2015): 30 runs per configuration for statistical significance

    Attributes:
        config: Simulation configuration
        base_seed: Base random seed for reproducibility
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        base_seed: int = 42,
    ):
        """
        Initialize experiment runner.

        Args:
            config: Simulation configuration
            base_seed: Base random seed
        """
        self.config = config or SimulationConfig()
        self.base_seed = base_seed
        self._results: list[SimulationResults] = []

    def run_all(
        self,
        algorithms: list[AlgorithmType] | None = None,
        connectivity_levels: list[float] | None = None,
        runs_per_config: int | None = None,
        progress_callback: Any | None = None,
    ) -> list[SimulationResults]:
        """
        Run all experimental configurations.

        Args:
            algorithms: Algorithms to test (default: both)
            connectivity_levels: Connectivity levels (default: [0.75, 0.40, 0.20])
            runs_per_config: Runs per configuration (default: 30)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of all simulation results
        """
        algorithms = algorithms or [AlgorithmType.ADAPTIVE, AlgorithmType.BASELINE]
        connectivity_levels = (
            connectivity_levels or self.config.network.connectivity_scenarios
        )
        runs_per_config = runs_per_config or self.config.scenario.runs_per_configuration

        total_runs = len(algorithms) * len(connectivity_levels) * runs_per_config
        current_run = 0

        self._results.clear()

        for algorithm in algorithms:
            for connectivity in connectivity_levels:
                for run in range(runs_per_config):
                    # Calculate seed for this run
                    seed = self.base_seed + current_run

                    # Run simulation
                    engine = SimulationEngine(
                        config=self.config,
                        algorithm_type=algorithm,
                        connectivity_level=connectivity,
                        random_seed=seed,
                    )

                    result = engine.run(run_number=run)
                    self._results.append(result)

                    current_run += 1

                    if progress_callback:
                        progress_callback(current_run, total_runs)

        return self._results

    def run_single_configuration(
        self,
        algorithm: AlgorithmType,
        connectivity_level: float,
        runs: int = 30,
    ) -> list[SimulationResults]:
        """
        Run simulations for a single configuration.

        Args:
            algorithm: Algorithm to test
            connectivity_level: Connectivity level
            runs: Number of runs

        Returns:
            List of results for this configuration
        """
        results = []

        for run in range(runs):
            seed = self.base_seed + run

            engine = SimulationEngine(
                config=self.config,
                algorithm_type=algorithm,
                connectivity_level=connectivity_level,
                random_seed=seed,
            )

            result = engine.run(run_number=run)
            results.append(result)

        return results

    def get_results(self) -> list[SimulationResults]:
        """Get all collected results."""
        return list(self._results)

    def get_results_by_algorithm(
        self, algorithm: AlgorithmType
    ) -> list[SimulationResults]:
        """Get results for a specific algorithm."""
        return [r for r in self._results if r.algorithm == algorithm]

    def get_results_by_connectivity(
        self, connectivity_level: float
    ) -> list[SimulationResults]:
        """Get results for a specific connectivity level."""
        return [r for r in self._results if r.connectivity_level == connectivity_level]


def run_simulation(
    algorithm: Literal["adaptive", "baseline"] = "adaptive",
    connectivity_level: float = 0.75,
    random_seed: int | None = None,
) -> SimulationResults:
    """
    Convenience function to run a single simulation.

    Args:
        algorithm: Algorithm type ("adaptive" or "baseline")
        connectivity_level: Network connectivity level
        random_seed: Random seed for reproducibility

    Returns:
        SimulationResults

    Example:
        >>> from ercs.simulation import run_simulation
        >>> results = run_simulation("adaptive", 0.75, random_seed=42)
        >>> print(f"Delivery rate: {results.delivery_rate:.2%}")
    """
    algorithm_type = AlgorithmType(algorithm)

    engine = SimulationEngine(
        algorithm_type=algorithm_type,
        connectivity_level=connectivity_level,
        random_seed=random_seed,
    )

    return engine.run()


def run_experiment(
    runs_per_config: int = 30,
    base_seed: int = 42,
) -> list[SimulationResults]:
    """
    Convenience function to run complete experiment.

    Args:
        runs_per_config: Runs per configuration (default: 30)
        base_seed: Base random seed

    Returns:
        List of all simulation results (180 total by default)

    Example:
        >>> from ercs.simulation import run_experiment
        >>> results = run_experiment(runs_per_config=5)  # Quick test
        >>> print(f"Total runs: {len(results)}")
    """
    runner = ExperimentRunner(base_seed=base_seed)
    return runner.run_all(runs_per_config=runs_per_config)
