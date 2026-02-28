"""
PRoPHET-Inspired Communication Layer (Phase 2).

This module implements store-and-forward message delivery with PRoPHET-inspired
forwarding logic for delay-tolerant networking in emergency coordination.

PRoPHET (Probabilistic Routing Protocol using History of Encounters and Transitivity)
uses delivery predictability values to make forwarding decisions:
- Predictability increases when nodes encounter each other
- Predictability ages (decays) over time
- Transitivity: if A meets B and B can reach C, A gains some predictability to C

Key equations:
    P(a,b) = P(a,b)_old + (1 - P(a,b)_old) × P_init   [encounter update]
    P(a,b) = P(a,b)_old × γ^k                          [aging, k = time units]
    P(a,c) = P(a,c)_old + (1 - P(a,c)_old) × P(a,b) × P(b,c) × β   [transitivity]

Sources:
    - Kumar et al. (2023): PRoPHET protocol parameters (P_init=0.75, β=0.25, γ=0.98)
    - Ullah & Qayyum (2022): Message TTL (300 min), transmit speed (2 Mbps), drop-oldest
    - Castillo et al. (2024): PRoPHET selection rationale
"""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ercs.config.parameters import (
    BufferDropPolicy,
    CommunicationParameters,
    NetworkParameters,
)


class MessageType(str, Enum):
    """Classification of messages in the emergency coordination system."""

    COORDINATION = "coordination"  # Task assignments, status updates
    STATUS = "status"  # Node availability, position reports
    ACKNOWLEDGEMENT = "ack"  # Message receipt confirmations


class MessageStatus(str, Enum):
    """Status of a message in the network."""

    PENDING = "pending"  # Awaiting transmission
    IN_TRANSIT = "in_transit"  # Being transmitted
    DELIVERED = "delivered"  # Successfully delivered
    EXPIRED = "expired"  # TTL exceeded
    DROPPED = "dropped"  # Dropped due to buffer overflow


@dataclass
class Message:
    """
    Represents a message in the DTN communication layer.

    Messages are store-carry-forwarded through the network until they reach
    their destination or expire (TTL).

    Attributes:
        message_id: Unique identifier for the message
        source_id: ID of the originating node
        destination_id: ID of the target node
        message_type: Classification of the message
        payload: Message content (bytes or serialisable data)
        creation_time: Simulation time when message was created (seconds)
        ttl_seconds: Time-to-live in seconds
        size_bytes: Message size in bytes
        urgency_level: Optional urgency level (H/M/L) for prioritisation
        hop_count: Number of forwarding hops taken
        status: Current status of the message
    """

    message_id: str
    source_id: str
    destination_id: str
    message_type: MessageType
    payload: bytes | dict | str
    creation_time: float
    ttl_seconds: int
    size_bytes: int
    urgency_level: str | None = None
    hop_count: int = 0
    status: MessageStatus = MessageStatus.PENDING

    def is_expired(self, current_time: float) -> bool:
        """Check if the message has exceeded its TTL."""
        age = current_time - self.creation_time
        return age >= self.ttl_seconds

    def remaining_ttl(self, current_time: float) -> float:
        """Calculate remaining time-to-live in seconds."""
        age = current_time - self.creation_time
        return max(0.0, self.ttl_seconds - age)

    def age(self, current_time: float) -> float:
        """Calculate message age in seconds."""
        return current_time - self.creation_time

    def increment_hop(self) -> None:
        """Record a forwarding hop."""
        self.hop_count += 1

    def copy(self) -> "Message":
        """Create a copy of this message for forwarding."""
        return Message(
            message_id=self.message_id,
            source_id=self.source_id,
            destination_id=self.destination_id,
            message_type=self.message_type,
            payload=self.payload,
            creation_time=self.creation_time,
            ttl_seconds=self.ttl_seconds,
            size_bytes=self.size_bytes,
            urgency_level=self.urgency_level,
            hop_count=self.hop_count,
            status=self.status,
        )

    def __hash__(self) -> int:
        """Hash by message ID for set operations."""
        return hash(self.message_id)

    def __eq__(self, other: object) -> bool:
        """Compare messages by ID."""
        if not isinstance(other, Message):
            return NotImplemented
        return self.message_id == other.message_id


def create_message(
    source_id: str,
    destination_id: str,
    message_type: MessageType,
    payload: bytes | dict | str,
    creation_time: float,
    ttl_seconds: int,
    size_bytes: int,
    urgency_level: str | None = None,
) -> Message:
    """
    Factory function to create a new message with a unique ID.

    Args:
        source_id: Originating node ID
        destination_id: Target node ID
        message_type: Type of message
        payload: Message content
        creation_time: Simulation time of creation
        ttl_seconds: Time-to-live
        size_bytes: Message size
        urgency_level: Optional urgency (H/M/L)

    Returns:
        New Message instance with unique ID
    """
    return Message(
        message_id=str(uuid.uuid4()),
        source_id=source_id,
        destination_id=destination_id,
        message_type=message_type,
        payload=payload,
        creation_time=creation_time,
        ttl_seconds=ttl_seconds,
        size_bytes=size_bytes,
        urgency_level=urgency_level,
    )


@dataclass
class MessageBuffer:
    """
    Message buffer for a DTN node implementing store-and-forward.

    Manages message storage with capacity limits and drop policies.
    Implements drop-oldest policy when buffer is full (Ullah & Qayyum, 2022).

    Attributes:
        node_id: ID of the node owning this buffer
        capacity_bytes: Maximum buffer capacity in bytes
        drop_policy: Policy for dropping messages when full
    """

    node_id: str
    capacity_bytes: int
    drop_policy: BufferDropPolicy = BufferDropPolicy.DROP_OLDEST

    # Internal storage
    _messages: dict[str, Message] = field(default_factory=dict, repr=False)
    _used_bytes: int = field(default=0, repr=False)
    _dropped_count: int = field(default=0, repr=False)
    _delivered_ids: set[str] = field(default_factory=set, repr=False)

    @property
    def used_bytes(self) -> int:
        """Current buffer usage in bytes."""
        return self._used_bytes

    @property
    def available_bytes(self) -> int:
        """Available buffer space in bytes."""
        return self.capacity_bytes - self._used_bytes

    @property
    def message_count(self) -> int:
        """Number of messages currently in buffer."""
        return len(self._messages)

    @property
    def dropped_count(self) -> int:
        """Total number of messages dropped due to buffer overflow."""
        return self._dropped_count

    @property
    def utilisation(self) -> float:
        """Buffer utilisation as fraction (0-1)."""
        if self.capacity_bytes == 0:
            return 1.0
        return self._used_bytes / self.capacity_bytes

    def has_message(self, message_id: str) -> bool:
        """Check if a message is in the buffer."""
        return message_id in self._messages

    def has_delivered(self, message_id: str) -> bool:
        """Check if we've already delivered a message (prevents duplicates)."""
        return message_id in self._delivered_ids

    def can_store(self, message: Message) -> bool:
        """Check if there's room for a message (before dropping)."""
        return message.size_bytes <= self.available_bytes

    def store(self, message: Message, current_time: float) -> bool:
        """
        Store a message in the buffer.

        If buffer is full, applies drop policy to make room.
        Returns False if message couldn't be stored.

        Args:
            message: Message to store
            current_time: Current simulation time (for expiration check)

        Returns:
            True if message was stored, False otherwise
        """
        # Don't store duplicates
        if self.has_message(message.message_id):
            return True  # Already have it

        # Don't store expired messages
        if message.is_expired(current_time):
            return False

        # Try to make room if needed
        while self.available_bytes < message.size_bytes:
            if not self._drop_message():
                return False  # Can't make room

        # Store the message
        self._messages[message.message_id] = message
        self._used_bytes += message.size_bytes
        return True

    def remove(self, message_id: str) -> Message | None:
        """
        Remove and return a message from the buffer.

        Args:
            message_id: ID of message to remove

        Returns:
            The removed message, or None if not found
        """
        message = self._messages.pop(message_id, None)
        if message is not None:
            self._used_bytes -= message.size_bytes
        return message

    def mark_delivered(self, message_id: str) -> None:
        """Mark a message as delivered (for duplicate detection)."""
        self._delivered_ids.add(message_id)
        # Also remove from buffer if present
        self.remove(message_id)

    def get_message(self, message_id: str) -> Message | None:
        """Get a message by ID without removing it."""
        return self._messages.get(message_id)

    def get_messages_for_destination(self, destination_id: str) -> list[Message]:
        """Get all messages destined for a specific node."""
        return [
            msg
            for msg in self._messages.values()
            if msg.destination_id == destination_id
        ]

    def expire_messages(self, current_time: float) -> list[Message]:
        """
        Remove and return all expired messages.

        Args:
            current_time: Current simulation time

        Returns:
            List of expired messages that were removed
        """
        expired = []
        for msg_id, msg in list(self._messages.items()):
            if msg.is_expired(current_time):
                expired.append(msg)
                self.remove(msg_id)
                msg.status = MessageStatus.EXPIRED
        return expired

    def _drop_message(self) -> bool:
        """
        Drop a message according to the drop policy.

        Returns:
            True if a message was dropped, False if buffer is empty
        """
        if not self._messages:
            return False

        if self.drop_policy == BufferDropPolicy.DROP_OLDEST:
            # Drop oldest by creation time
            oldest = min(self._messages.values(), key=lambda m: m.creation_time)
        else:  # DROP_NEWEST
            oldest = max(self._messages.values(), key=lambda m: m.creation_time)

        self.remove(oldest.message_id)
        oldest.status = MessageStatus.DROPPED
        self._dropped_count += 1
        return True

    def __iter__(self) -> Iterator[Message]:
        """Iterate over all messages in buffer."""
        return iter(self._messages.values())

    def clear(self) -> None:
        """Clear all messages from buffer."""
        self._messages.clear()
        self._used_bytes = 0


class DeliveryPredictabilityMatrix:
    """
    Manages delivery predictability values for PRoPHET routing.

    Each node maintains P(a,b) values representing the probability that
    node 'a' can successfully deliver a message to destination 'b'.

    Predictability updates occur:
    1. On encounter: P increases when nodes meet
    2. Aging: P decays over time when no encounters
    3. Transitivity: If A meets B, A gains some of B's predictabilities

    Sources:
        - Kumar et al. (2023): P_init=0.75, β=0.25, γ=0.98

    Attributes:
        p_init: Initial predictability on first encounter
        beta: Transitivity scaling constant
        gamma: Aging constant (applied per time unit)
        update_interval: Time units between aging updates
    """

    def __init__(
        self,
        p_init: float = 0.75,
        beta: float = 0.25,
        gamma: float = 0.98,
        update_interval: float = 0.1,
    ):
        """
        Initialise the predictability matrix.

        Args:
            p_init: Initial predictability constant (default: 0.75)
            beta: Transitivity scaling constant (default: 0.25)
            gamma: Aging constant (default: 0.98)
            update_interval: Time units for aging (default: 0.1)
        """
        if not (0 < p_init <= 1):
            raise ValueError(f"p_init must be in (0, 1], got {p_init}")
        if not (0 < beta <= 1):
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if not (0 < gamma < 1):
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        if update_interval <= 0:
            raise ValueError(f"update_interval must be positive, got {update_interval}")

        self.p_init = p_init
        self.beta = beta
        self.gamma = gamma
        self.update_interval = update_interval

        # Matrix storage: {node_id: {destination_id: predictability}}
        self._matrix: dict[str, dict[str, float]] = {}

        # Track last aging time per node
        self._last_aging_time: dict[str, float] = {}

    def get_predictability(self, node_id: str, destination_id: str) -> float:
        """
        Get delivery predictability P(node_id, destination_id).

        Returns 0.0 if no predictability exists.
        """
        if node_id not in self._matrix:
            return 0.0
        return self._matrix[node_id].get(destination_id, 0.0)

    def set_predictability(
        self, node_id: str, destination_id: str, value: float
    ) -> None:
        """Set delivery predictability directly (for testing/initialisation)."""
        if node_id not in self._matrix:
            self._matrix[node_id] = {}
        self._matrix[node_id][destination_id] = min(max(value, 0.0), 1.0)

    def update_encounter(self, node_a: str, node_b: str) -> None:
        """
        Update predictability when two nodes encounter each other.

        Applies the PRoPHET encounter equation:
            P(a,b) = P(a,b)_old + (1 - P(a,b)_old) × P_init

        This is applied symmetrically: both nodes update their predictability
        to each other.

        Args:
            node_a: First node in encounter
            node_b: Second node in encounter
        """
        # Update A's predictability to B
        p_old_ab = self.get_predictability(node_a, node_b)
        p_new_ab = p_old_ab + (1 - p_old_ab) * self.p_init
        self.set_predictability(node_a, node_b, p_new_ab)

        # Update B's predictability to A
        p_old_ba = self.get_predictability(node_b, node_a)
        p_new_ba = p_old_ba + (1 - p_old_ba) * self.p_init
        self.set_predictability(node_b, node_a, p_new_ba)

    def update_transitivity(self, node_a: str, node_b: str) -> None:
        """
        Update predictabilities through transitivity.

        When A meets B, A can potentially reach destinations that B can reach.
        Applies the PRoPHET transitivity equation:
            P(a,c) = P(a,c)_old + (1 - P(a,c)_old) × P(a,b) × P(b,c) × β

        This is applied symmetrically for both nodes.

        Args:
            node_a: First node in encounter
            node_b: Second node in encounter
        """
        # A gains from B's destinations
        self._apply_transitivity(node_a, node_b)

        # B gains from A's destinations
        self._apply_transitivity(node_b, node_a)

    def _apply_transitivity(self, receiver: str, giver: str) -> None:
        """Apply transitivity from giver to receiver."""
        if giver not in self._matrix:
            return

        p_receiver_giver = self.get_predictability(receiver, giver)

        for dest, p_giver_dest in self._matrix[giver].items():
            if dest == receiver:
                continue  # Skip self-referential

            p_old = self.get_predictability(receiver, dest)
            p_new = p_old + (1 - p_old) * p_receiver_giver * p_giver_dest * self.beta
            self.set_predictability(receiver, dest, p_new)

    def age_predictabilities(self, node_id: str, current_time: float) -> None:
        """
        Age all predictabilities for a node based on elapsed time.

        Applies the PRoPHET aging equation:
            P(a,b) = P(a,b)_old × γ^k

        Where k is the number of time units since last aging.

        Args:
            node_id: Node whose predictabilities to age
            current_time: Current simulation time
        """
        if node_id not in self._matrix:
            return

        # Calculate number of aging intervals since last update
        last_time = self._last_aging_time.get(node_id, current_time)
        elapsed = current_time - last_time

        if elapsed <= 0:
            return

        # k = number of update intervals
        k = elapsed / self.update_interval

        if k < 1:
            return

        # Apply aging: P = P × γ^k
        aging_factor = self.gamma**k

        for dest in list(self._matrix[node_id].keys()):
            old_p = self._matrix[node_id][dest]
            new_p = old_p * aging_factor

            # Remove very small predictabilities to save memory
            if new_p < 0.001:
                del self._matrix[node_id][dest]
            else:
                self._matrix[node_id][dest] = new_p

        self._last_aging_time[node_id] = current_time

    def get_all_predictabilities(self, node_id: str) -> dict[str, float]:
        """Get all predictability values for a node."""
        if node_id not in self._matrix:
            return {}
        return dict(self._matrix[node_id])

    def get_best_forwarder(
        self,
        current_node: str,
        destination: str,
        candidates: list[str],
    ) -> str | None:
        """
        Find the best forwarding candidate for a message.

        Returns the candidate with highest predictability to the destination,
        but only if that predictability is higher than the current node's.

        Args:
            current_node: Node currently holding the message
            destination: Message destination
            candidates: List of potential forwarding nodes (neighbours)

        Returns:
            Best candidate node ID, or None if no candidate is better
        """
        if not candidates:
            return None

        current_p = self.get_predictability(current_node, destination)
        best_candidate = None
        best_p = current_p

        for candidate in candidates:
            if candidate == destination:
                # Direct delivery possible
                return candidate

            p = self.get_predictability(candidate, destination)
            if p > best_p:
                best_p = p
                best_candidate = candidate

        return best_candidate

    def initialise_node(self, node_id: str, current_time: float = 0.0) -> None:
        """Initialise tracking for a new node."""
        if node_id not in self._matrix:
            self._matrix[node_id] = {}
        if node_id not in self._last_aging_time:
            self._last_aging_time[node_id] = current_time

    def reset(self) -> None:
        """Reset all predictability values."""
        self._matrix.clear()
        self._last_aging_time.clear()

    @property
    def node_count(self) -> int:
        """Number of nodes with predictability data."""
        return len(self._matrix)


@dataclass
class TransmissionResult:
    """Result of a message transmission attempt."""

    message: Message
    success: bool
    transmission_time: float  # Time taken in seconds
    source_node: str
    target_node: str
    reason: str = ""  # Failure reason if not successful


class CommunicationLayer:
    """
    Main communication layer implementing PRoPHET-inspired routing.

    Manages message buffers, delivery predictability, and forwarding decisions
    for all nodes in the network.

    This layer provides:
    - Store-and-forward message handling
    - Delivery predictability matrix
    - Forwarding decisions based on PRoPHET logic
    - Buffer management with drop-oldest policy

    Sources:
        - Kumar et al. (2023): PRoPHET parameters
        - Ullah & Qayyum (2022): Message handling, buffer policy
    """

    def __init__(
        self,
        comm_params: CommunicationParameters,
        network_params: NetworkParameters,
        node_ids: list[str],
    ):
        """
        Initialise the communication layer.

        Args:
            comm_params: Communication layer parameters
            network_params: Network parameters (buffer size, message size)
            node_ids: List of all node IDs in the network
        """
        self.comm_params = comm_params
        self.network_params = network_params

        # Create predictability matrix
        self.predictability = DeliveryPredictabilityMatrix(
            p_init=comm_params.prophet.p_init,
            beta=comm_params.prophet.beta,
            gamma=comm_params.prophet.gamma,
            update_interval=comm_params.update_interval_seconds,
        )

        # Create message buffer for each node
        self.buffers: dict[str, MessageBuffer] = {}
        for node_id in node_ids:
            self.buffers[node_id] = MessageBuffer(
                node_id=node_id,
                capacity_bytes=network_params.buffer_size_bytes,
                drop_policy=comm_params.buffer_drop_policy,
            )
            self.predictability.initialise_node(node_id)

        # Statistics
        self._messages_created = 0
        self._messages_delivered = 0
        self._messages_expired = 0
        self._messages_dropped = 0
        self._total_hops = 0
        self._delivery_times: list[float] = []

    def create_message(
        self,
        source_id: str,
        destination_id: str,
        message_type: MessageType,
        payload: bytes | dict | str,
        current_time: float,
        urgency_level: str | None = None,
    ) -> Message:
        """
        Create a new message and store it in the source node's buffer.

        Args:
            source_id: Originating node ID
            destination_id: Target node ID
            message_type: Type of message
            payload: Message content
            current_time: Current simulation time
            urgency_level: Optional urgency level

        Returns:
            The created Message object

        Raises:
            ValueError: If source node doesn't exist
        """
        if source_id not in self.buffers:
            raise ValueError(f"Unknown source node: {source_id}")

        message = create_message(
            source_id=source_id,
            destination_id=destination_id,
            message_type=message_type,
            payload=payload,
            creation_time=current_time,
            ttl_seconds=self.comm_params.message_ttl_seconds,
            size_bytes=self.network_params.message_size_bytes,
            urgency_level=urgency_level,
        )

        self.buffers[source_id].store(message, current_time)
        self._messages_created += 1

        return message

    def process_encounter(
        self,
        node_a: str,
        node_b: str,
        current_time: float,
    ) -> list[TransmissionResult]:
        """
        Process an encounter between two nodes.

        When nodes meet:
        1. Age predictabilities based on time since last update
        2. Update delivery predictability (encounter + transitivity)
        3. Exchange messages that can be delivered or forwarded

        Args:
            node_a: First node in encounter
            node_b: Second node in encounter
            current_time: Current simulation time

        Returns:
            List of transmission results from this encounter
        """
        results = []

        # Age predictabilities first (based on time since last update)
        self.predictability.age_predictabilities(node_a, current_time)
        self.predictability.age_predictabilities(node_b, current_time)

        # Then update predictability from encounter
        self.predictability.update_encounter(node_a, node_b)
        self.predictability.update_transitivity(node_a, node_b)

        # Exchange messages A -> B
        results.extend(self._transfer_messages(node_a, node_b, current_time))

        # Exchange messages B -> A
        results.extend(self._transfer_messages(node_b, node_a, current_time))

        return results

    def _transfer_messages(
        self,
        from_node: str,
        to_node: str,
        current_time: float,
    ) -> list[TransmissionResult]:
        """
        Transfer appropriate messages from one node to another.

        Transfers messages where:
        - Destination is to_node (direct delivery), OR
        - to_node has higher predictability to destination

        Args:
            from_node: Node sending messages
            to_node: Node receiving messages
            current_time: Current simulation time

        Returns:
            List of transmission results
        """
        results = []
        buffer_from = self.buffers[from_node]
        buffer_to = self.buffers[to_node]

        # Get messages to consider for transfer
        messages_to_transfer = []

        for message in list(buffer_from):
            # Skip expired messages
            if message.is_expired(current_time):
                continue

            # Skip messages already delivered to recipient
            if buffer_to.has_delivered(message.message_id):
                continue

            # Skip messages recipient already has
            if buffer_to.has_message(message.message_id):
                continue

            dest = message.destination_id

            # Direct delivery
            if dest == to_node:
                messages_to_transfer.append((message, "direct"))
                continue

            # Check if forwarding is beneficial
            p_from = self.predictability.get_predictability(from_node, dest)
            p_to = self.predictability.get_predictability(to_node, dest)

            if p_to > p_from:
                messages_to_transfer.append((message, "forward"))

        # Transfer messages
        for message, transfer_type in messages_to_transfer:
            result = self._transmit_message(
                message, from_node, to_node, current_time, transfer_type
            )
            results.append(result)

        return results

    def _transmit_message(
        self,
        message: Message,
        from_node: str,
        to_node: str,
        current_time: float,
        transfer_type: str,
    ) -> TransmissionResult:
        """
        Transmit a single message between nodes.

        Args:
            message: Message to transmit
            from_node: Sending node
            to_node: Receiving node
            current_time: Current simulation time
            transfer_type: "direct" for delivery, "forward" for forwarding

        Returns:
            TransmissionResult indicating success/failure
        """
        # Calculate transmission time
        transmission_time = self._calculate_transmission_time(message.size_bytes)

        buffer_to = self.buffers[to_node]

        if transfer_type == "direct":
            # Direct delivery to destination
            buffer_to.mark_delivered(message.message_id)
            message.status = MessageStatus.DELIVERED

            # Update statistics
            self._messages_delivered += 1
            self._total_hops += message.hop_count
            delivery_time = current_time + transmission_time - message.creation_time
            self._delivery_times.append(delivery_time)

            # Remove from sender's buffer
            self.buffers[from_node].remove(message.message_id)

            return TransmissionResult(
                message=message,
                success=True,
                transmission_time=transmission_time,
                source_node=from_node,
                target_node=to_node,
                reason="delivered",
            )
        else:
            # Forward to intermediate node
            message_copy = message.copy()
            message_copy.increment_hop()
            message_copy.status = MessageStatus.IN_TRANSIT

            if buffer_to.store(message_copy, current_time):
                return TransmissionResult(
                    message=message_copy,
                    success=True,
                    transmission_time=transmission_time,
                    source_node=from_node,
                    target_node=to_node,
                    reason="forwarded",
                )
            else:
                return TransmissionResult(
                    message=message,
                    success=False,
                    transmission_time=0,
                    source_node=from_node,
                    target_node=to_node,
                    reason="buffer_full",
                )

    def _calculate_transmission_time(self, size_bytes: int) -> float:
        """
        Calculate time to transmit a message.

        Args:
            size_bytes: Message size in bytes

        Returns:
            Transmission time in seconds
        """
        bits = size_bytes * 8
        return bits / self.comm_params.transmit_speed_bps

    def expire_all_messages(self, current_time: float) -> int:
        """
        Expire messages across all buffers.

        Args:
            current_time: Current simulation time

        Returns:
            Total number of expired messages
        """
        total_expired = 0
        for buffer in self.buffers.values():
            expired = buffer.expire_messages(current_time)
            total_expired += len(expired)

        self._messages_expired += total_expired
        return total_expired

    def get_buffer(self, node_id: str) -> MessageBuffer:
        """Get the message buffer for a node."""
        return self.buffers[node_id]

    def get_pending_messages(self, node_id: str) -> list[Message]:
        """Get all pending messages in a node's buffer."""
        return list(self.buffers[node_id])

    def get_delivery_predictability(self, node_id: str, destination: str) -> float:
        """Get delivery predictability from a node to a destination."""
        return self.predictability.get_predictability(node_id, destination)

    @property
    def statistics(self) -> dict:
        """Get communication layer statistics."""
        return {
            "messages_created": self._messages_created,
            "messages_delivered": self._messages_delivered,
            "messages_expired": self._messages_expired,
            "messages_dropped": sum(b.dropped_count for b in self.buffers.values()),
            "total_hops": self._total_hops,
            "average_hops": (
                self._total_hops / self._messages_delivered
                if self._messages_delivered > 0
                else 0
            ),
            "delivery_rate": (
                self._messages_delivered / self._messages_created
                if self._messages_created > 0
                else 0
            ),
            "average_delivery_time": (
                np.mean(self._delivery_times) if self._delivery_times else 0
            ),
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._messages_created = 0
        self._messages_delivered = 0
        self._messages_expired = 0
        self._messages_dropped = 0
        self._total_hops = 0
        self._delivery_times.clear()
