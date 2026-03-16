"""
PRoPHETv2 Communication Layer (Phase 2).

This module implements store-and-forward message delivery with PRoPHETv2
forwarding logic for delay-tolerant networking in emergency coordination.

PRoPHETv2 extends the original PRoPHET protocol with:
- Time-based encounter updates (P_enc depends on inter-encounter interval)
- MAX-based transitivity (prevents delivery predictability saturation)
- Greater-or-equal forwarding condition

Key equations:
    P_enc = P_enc_max × min(1, Δt / I_typ)             [time-based encounter]
    P(a,b) = P(a,b)_old + (1 - P(a,b)_old) × P_enc    [encounter update]
    P(a,b) = P(a,b)_old × γ^k                          [aging, k = time units]
    P(a,c) = max(P(a,c)_old, P(a,b) × P(b,c) × β)     [MAX transitivity]

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
    Implements drop-oldest policy when buffer is full.

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
    Manages delivery predictability values for PRoPHETv2 routing.

    Each node maintains P(a,b) values representing the probability that
    node 'a' can successfully deliver a message to destination 'b'.

    PRoPHETv2 differs from original PRoPHET:
    1. Time-based encounter: P_enc depends on inter-encounter interval
    2. MAX-based transitivity: prevents saturation to ~1.0
    3. Greater-or-equal forwarding condition

    Predictability updates occur:
    1. On encounter: P increases (time-scaled P_enc)
    2. Aging: P decays over time when no encounters
    3. Transitivity: MAX(P_old, P(a,b) × P(b,c) × β)

    Attributes:
        p_enc_max: Maximum encounter probability
        i_typ: Typical inter-encounter interval (seconds)
        beta: Transitivity scaling constant
        gamma: Aging constant (applied per time unit)
        update_interval: Time units between aging updates
    """

    def __init__(
        self,
        p_enc_max: float = 0.5,
        i_typ: float = 1800.0,
        beta: float = 0.9,
        gamma: float = 0.999885791,
        update_interval: float = 30.0,
        min_predictability_threshold: float = 0.001,
    ):
        """
        Initialise the predictability matrix with PRoPHETv2 parameters.

        Args:
            p_enc_max: Maximum encounter probability (default: 0.5)
            i_typ: Typical inter-encounter interval in seconds (default: 1800.0)
            beta: Transitivity scaling constant (default: 0.9)
            gamma: Aging constant (default: 0.999885791)
            update_interval: Time units for aging (default: 30.0)
            min_predictability_threshold: Prune P values below this (default: 0.001)
        """
        if not (0 < p_enc_max <= 1):
            raise ValueError(f"p_enc_max must be in (0, 1], got {p_enc_max}")
        if i_typ <= 0:
            raise ValueError(f"i_typ must be positive, got {i_typ}")
        if not (0 < beta <= 1):
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if not (0 < gamma < 1):
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        if update_interval <= 0:
            raise ValueError(f"update_interval must be positive, got {update_interval}")

        self.p_enc_max = p_enc_max
        self.i_typ = i_typ
        self.beta = beta
        self.gamma = gamma
        self.update_interval = update_interval
        self.min_predictability_threshold = min_predictability_threshold

        # Matrix storage: {node_id: {destination_id: predictability}}
        self._matrix: dict[str, dict[str, float]] = {}

        # Track last aging time per node
        self._last_aging_time: dict[str, float] = {}

        # PRoPHETv2: track last encounter time per directed node pair
        self._last_encounter_time: dict[tuple[str, str], float] = {}

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

    def _calculate_p_enc(self, node_a: str, node_b: str, current_time: float) -> float:
        """
        Calculate time-based encounter probability P_enc (PRoPHETv2 Eq. 1).

        If this is the first encounter, P_enc = P_enc_max.
        If time since last encounter < I_typ, P_enc is scaled down.
        Otherwise, P_enc = P_enc_max.

        Args:
            node_a: First node in encounter
            node_b: Second node in encounter
            current_time: Current simulation time

        Returns:
            Calculated P_enc value
        """
        key = (node_a, node_b)
        last_enc_time = self._last_encounter_time.get(key, 0.0)

        if last_enc_time == 0.0:
            # First encounter
            return self.p_enc_max

        delta_t = current_time - last_enc_time
        if delta_t < self.i_typ:
            # Recent encounter — reduce P_enc proportionally
            return self.p_enc_max * (delta_t / self.i_typ)

        # Long time since last encounter — use full P_enc
        return self.p_enc_max

    def update_encounter(self, node_a: str, node_b: str, current_time: float) -> None:
        """
        Update predictability when two nodes encounter each other.

        Applies the PRoPHETv2 time-based encounter equation:
            P_enc = P_enc_max × min(1, Δt / I_typ)
            P(a,b) = P(a,b)_old + (1 - P(a,b)_old) × P_enc

        This is applied symmetrically: both nodes update their predictability
        to each other.

        Args:
            node_a: First node in encounter
            node_b: Second node in encounter
            current_time: Current simulation time
        """
        # Update A's predictability to B
        p_enc_ab = self._calculate_p_enc(node_a, node_b, current_time)
        p_old_ab = self.get_predictability(node_a, node_b)
        p_new_ab = p_old_ab + (1 - p_old_ab) * p_enc_ab
        self.set_predictability(node_a, node_b, p_new_ab)
        self._last_encounter_time[(node_a, node_b)] = current_time

        # Update B's predictability to A
        p_enc_ba = self._calculate_p_enc(node_b, node_a, current_time)
        p_old_ba = self.get_predictability(node_b, node_a)
        p_new_ba = p_old_ba + (1 - p_old_ba) * p_enc_ba
        self.set_predictability(node_b, node_a, p_new_ba)
        self._last_encounter_time[(node_b, node_a)] = current_time

    def update_transitivity(self, node_a: str, node_b: str) -> None:
        """
        Update predictabilities through transitivity (PRoPHETv2 MAX-based).

        When A meets B, A can potentially reach destinations that B can reach.
        PRoPHETv2 uses MAX instead of additive formula:
            P(a,c) = max(P(a,c)_old, P(a,b) × P(b,c) × β)

        This prevents the delivery predictability saturation problem where
        all P values converge to ~1.0 under original PRoPHET.

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
        """Apply MAX-based transitivity from giver to receiver (PRoPHETv2)."""
        if giver not in self._matrix:
            return

        p_receiver_giver = self.get_predictability(receiver, giver)

        for dest, p_giver_dest in self._matrix[giver].items():
            if dest == receiver:
                continue  # Skip self-referential

            p_old = self.get_predictability(receiver, dest)
            p_new = p_receiver_giver * p_giver_dest * self.beta

            # PRoPHETv2: use MAX instead of additive formula
            if p_new > p_old:
                self.set_predictability(receiver, dest, p_new)

    def age_predictabilities(self, node_id: str, current_time: float) -> None:
        """
        Age all predictabilities for a node based on elapsed time.

        Applies the aging equation (same formula, PRoPHETv2 gamma):
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
            if new_p < self.min_predictability_threshold:
                del self._matrix[node_id][dest]
            else:
                self._matrix[node_id][dest] = new_p

        self._last_aging_time[node_id] = current_time

    def get_last_encounter_time(self, node_a: str, node_b: str) -> float:
        """
        Return time of last direct encounter between node_a and node_b.

        Returns 0.0 if the pair has never encountered each other.
        """
        return self._last_encounter_time.get((node_a, node_b), 0.0)

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
        but only if that predictability is >= the current node's (PRoPHETv2).

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
            if p >= best_p and p > 0:
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
        self._last_encounter_time.clear()

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
    reason: str = ""  # "delivered", "forwarded", or "buffer_full"


class CommunicationLayer:
    """
    Main communication layer implementing PRoPHETv2 routing.

    Manages message buffers, delivery predictability, and forwarding decisions
    for all nodes in the network.

    This layer provides:
    - Store-and-forward message handling
    - PRoPHETv2 delivery predictability matrix
    - Forwarding decisions based on PRoPHETv2 logic (>= condition)
    - Buffer management with drop-oldest policy

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

        # Create predictability matrix (PRoPHETv2)
        self.predictability = DeliveryPredictabilityMatrix(
            p_enc_max=comm_params.prophet.p_enc_max,
            i_typ=comm_params.prophet.i_typ,
            beta=comm_params.prophet.beta,
            gamma=comm_params.prophet.gamma,
            update_interval=comm_params.update_interval_seconds,
            min_predictability_threshold=comm_params.min_predictability_threshold,
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

        # Then update predictability from encounter (PRoPHETv2: time-based)
        self.predictability.update_encounter(node_a, node_b, current_time)
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

            # Check if forwarding is beneficial (PRoPHETv2: >= instead of >)
            p_from = self.predictability.get_predictability(from_node, dest)
            p_to = self.predictability.get_predictability(to_node, dest)

            if p_to >= p_from and p_to > 0:
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

    def transfer_messages(
        self,
        node_a: str,
        node_b: str,
        current_time: float,
    ) -> list[TransmissionResult]:
        """
        Transfer messages between two nodes on an active link.

        Unlike process_encounter(), this does NOT update predictability
        (encounter equation or transitivity).  It only ages predictabilities
        and exchanges messages.  Use this for sustained contacts where the
        connection-up event has already been processed.

        RFC 6693: encounter updates happen at
        contact establishment, not continuously during sustained contact.

        Args:
            node_a: First node in active link
            node_b: Second node in active link
            current_time: Current simulation time

        Returns:
            List of transmission results from this transfer
        """
        results = []

        # Age predictabilities (idempotent — no-op if already aged this period)
        self.predictability.age_predictabilities(node_a, current_time)
        self.predictability.age_predictabilities(node_b, current_time)

        # Exchange messages A -> B
        results.extend(self._transfer_messages(node_a, node_b, current_time))

        # Exchange messages B -> A
        results.extend(self._transfer_messages(node_b, node_a, current_time))

        return results

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

    def get_last_encounter_time(self, node_a: str, node_b: str) -> float:
        """Get time of last direct encounter between two nodes (0.0 if never)."""
        return self.predictability.get_last_encounter_time(node_a, node_b)

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
