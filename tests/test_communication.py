"""
Tests for Communication Layer (Phase 2).

These tests verify that the PRoPHET-inspired communication layer correctly
implements the specifications from the Phase 2 documentation:
- PRoPHET parameters: P_init=0.75, β=0.25, γ=0.98
- Message TTL: 300 minutes (18000 seconds)
- Transmit speed: 2 Mbps
- Buffer drop policy: drop oldest

Sources verified:
    - Kumar et al. (2023): PRoPHET protocol parameters
    - Ullah & Qayyum (2022): Message handling parameters
"""

import pytest
import numpy as np

from ercs.config.parameters import (
    CommunicationParameters,
    NetworkParameters,
    PRoPHETParameters,
    BufferDropPolicy,
)
from ercs.communication import (
    CommunicationLayer,
    DeliveryPredictabilityMatrix,
    Message,
    MessageBuffer,
    MessageStatus,
    MessageType,
    TransmissionResult,
    create_message,
)

# =============================================================================
# Test Message Class
# =============================================================================


class TestMessage:
    """Tests for the Message class."""

    def test_message_creation(self):
        """Test basic message creation with all attributes."""
        msg = Message(
            message_id="msg_001",
            source_id="node_a",
            destination_id="node_b",
            message_type=MessageType.COORDINATION,
            payload=b"test payload",
            creation_time=100.0,
            ttl_seconds=18000,
            size_bytes=512000,
        )

        assert msg.message_id == "msg_001"
        assert msg.source_id == "node_a"
        assert msg.destination_id == "node_b"
        assert msg.message_type == MessageType.COORDINATION
        assert msg.creation_time == 100.0
        assert msg.ttl_seconds == 18000
        assert msg.size_bytes == 512000
        assert msg.hop_count == 0
        assert msg.status == MessageStatus.PENDING

    def test_message_not_expired_before_ttl(self):
        """Test message is not expired before TTL."""
        msg = Message(
            message_id="msg_001",
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=18000,
            size_bytes=512000,
        )

        # At creation time
        assert msg.is_expired(0.0) is False

        # Just before expiration (5 hours - 1 second)
        assert msg.is_expired(17999.0) is False

    def test_message_expired_at_ttl(self):
        """Test message expires exactly at TTL."""
        msg = Message(
            message_id="msg_001",
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=18000,  # 300 minutes
            size_bytes=512000,
        )

        # At TTL
        assert msg.is_expired(18000.0) is True

        # After TTL
        assert msg.is_expired(20000.0) is True

    def test_remaining_ttl(self):
        """Test remaining TTL calculation."""
        msg = Message(
            message_id="msg_001",
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=18000,
            size_bytes=512000,
        )

        assert msg.remaining_ttl(0.0) == 18000.0
        assert msg.remaining_ttl(9000.0) == 9000.0
        assert msg.remaining_ttl(18000.0) == 0.0
        assert msg.remaining_ttl(20000.0) == 0.0  # Can't be negative

    def test_message_age(self):
        """Test message age calculation."""
        msg = Message(
            message_id="msg_001",
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=100.0,
            ttl_seconds=18000,
            size_bytes=512000,
        )

        assert msg.age(100.0) == 0.0
        assert msg.age(200.0) == 100.0
        assert msg.age(1000.0) == 900.0

    def test_increment_hop(self):
        """Test hop count increment."""
        msg = Message(
            message_id="msg_001",
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=18000,
            size_bytes=512000,
        )

        assert msg.hop_count == 0
        msg.increment_hop()
        assert msg.hop_count == 1
        msg.increment_hop()
        assert msg.hop_count == 2

    def test_message_copy(self):
        """Test message copy creates independent copy."""
        original = Message(
            message_id="msg_001",
            source_id="a",
            destination_id="b",
            message_type=MessageType.COORDINATION,
            payload="test",
            creation_time=100.0,
            ttl_seconds=18000,
            size_bytes=512000,
            urgency_level="H",
        )

        copy = original.copy()

        # Same values
        assert copy.message_id == original.message_id
        assert copy.source_id == original.source_id
        assert copy.destination_id == original.destination_id
        assert copy.creation_time == original.creation_time

        # Modifications don't affect original
        copy.increment_hop()
        assert copy.hop_count == 1
        assert original.hop_count == 0

    def test_message_hash_and_equality(self):
        """Test message hashing and equality based on ID."""
        msg1 = create_message(
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=18000,
            size_bytes=512000,
        )

        msg2 = Message(
            message_id=msg1.message_id,  # Same ID
            source_id="different",
            destination_id="different",
            message_type=MessageType.COORDINATION,
            payload="different",
            creation_time=999.0,
            ttl_seconds=100,
            size_bytes=100,
        )

        assert msg1 == msg2
        assert hash(msg1) == hash(msg2)


class TestCreateMessage:
    """Tests for the create_message factory function."""

    def test_creates_unique_ids(self):
        """Test that each message gets a unique ID."""
        messages = [
            create_message(
                source_id="a",
                destination_id="b",
                message_type=MessageType.STATUS,
                payload="test",
                creation_time=0.0,
                ttl_seconds=18000,
                size_bytes=512000,
            )
            for _ in range(100)
        ]

        ids = [m.message_id for m in messages]
        assert len(set(ids)) == 100  # All unique


# =============================================================================
# Test MessageBuffer Class
# =============================================================================


class TestMessageBuffer:
    """Tests for the MessageBuffer class."""

    @pytest.fixture
    def buffer(self) -> MessageBuffer:
        """Create a buffer with 5 MB capacity."""
        return MessageBuffer(
            node_id="test_node",
            capacity_bytes=5_242_880,
            drop_policy=BufferDropPolicy.DROP_OLDEST,
        )

    @pytest.fixture
    def small_buffer(self) -> MessageBuffer:
        """Create a small buffer for testing overflow."""
        return MessageBuffer(
            node_id="test_node",
            capacity_bytes=1_500_000,  # ~3 messages at 500 KB each
            drop_policy=BufferDropPolicy.DROP_OLDEST,
        )

    def test_buffer_creation(self, buffer: MessageBuffer):
        """Test buffer initialisation."""
        assert buffer.node_id == "test_node"
        assert buffer.capacity_bytes == 5_242_880
        assert buffer.used_bytes == 0
        assert buffer.available_bytes == 5_242_880
        assert buffer.message_count == 0

    def test_store_message(self, buffer: MessageBuffer):
        """Test storing a message in buffer."""
        msg = create_message(
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=18000,
            size_bytes=512000,
        )

        result = buffer.store(msg, current_time=0.0)

        assert result is True
        assert buffer.message_count == 1
        assert buffer.used_bytes == 512000
        assert buffer.has_message(msg.message_id)

    def test_store_duplicate_ignored(self, buffer: MessageBuffer):
        """Test that duplicate messages are not stored twice."""
        msg = create_message(
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=18000,
            size_bytes=512000,
        )

        buffer.store(msg, current_time=0.0)
        buffer.store(msg, current_time=0.0)  # Duplicate

        assert buffer.message_count == 1

    def test_store_expired_message_rejected(self, buffer: MessageBuffer):
        """Test that expired messages are not stored."""
        msg = create_message(
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=100,  # Short TTL
            size_bytes=512000,
        )

        result = buffer.store(msg, current_time=200.0)  # After expiration

        assert result is False
        assert buffer.message_count == 0

    def test_drop_oldest_policy(self, small_buffer: MessageBuffer):
        """Test drop-oldest policy when buffer is full."""
        # Create messages with different creation times
        messages = []
        for i in range(5):  # More than buffer can hold
            msg = create_message(
                source_id="a",
                destination_id="b",
                message_type=MessageType.STATUS,
                payload=f"test_{i}",
                creation_time=float(i * 10),  # 0, 10, 20, 30, 40
                ttl_seconds=18000,
                size_bytes=512000,
            )
            messages.append(msg)
            small_buffer.store(msg, current_time=50.0)

        # Should have dropped oldest messages
        assert small_buffer.message_count <= 3
        assert small_buffer.dropped_count > 0

        # Newest messages should remain
        assert small_buffer.has_message(messages[-1].message_id)

    def test_remove_message(self, buffer: MessageBuffer):
        """Test removing a message from buffer."""
        msg = create_message(
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=18000,
            size_bytes=512000,
        )

        buffer.store(msg, current_time=0.0)
        removed = buffer.remove(msg.message_id)

        assert removed is not None
        assert removed.message_id == msg.message_id
        assert buffer.message_count == 0
        assert buffer.used_bytes == 0

    def test_remove_nonexistent_returns_none(self, buffer: MessageBuffer):
        """Test removing non-existent message returns None."""
        removed = buffer.remove("nonexistent")
        assert removed is None

    def test_mark_delivered(self, buffer: MessageBuffer):
        """Test marking a message as delivered."""
        msg = create_message(
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=18000,
            size_bytes=512000,
        )

        buffer.store(msg, current_time=0.0)
        buffer.mark_delivered(msg.message_id)

        assert buffer.has_delivered(msg.message_id)
        assert not buffer.has_message(msg.message_id)  # Removed from buffer

    def test_get_messages_for_destination(self, buffer: MessageBuffer):
        """Test filtering messages by destination."""
        for dest in ["node_a", "node_a", "node_b"]:
            msg = create_message(
                source_id="source",
                destination_id=dest,
                message_type=MessageType.STATUS,
                payload="test",
                creation_time=0.0,
                ttl_seconds=18000,
                size_bytes=100000,
            )
            buffer.store(msg, current_time=0.0)

        msgs_to_a = buffer.get_messages_for_destination("node_a")
        msgs_to_b = buffer.get_messages_for_destination("node_b")

        assert len(msgs_to_a) == 2
        assert len(msgs_to_b) == 1

    def test_expire_messages(self, buffer: MessageBuffer):
        """Test expiring old messages."""
        # Create messages with different TTLs
        msg_short = create_message(
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="short",
            creation_time=0.0,
            ttl_seconds=100,
            size_bytes=100000,
        )
        msg_long = create_message(
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="long",
            creation_time=0.0,
            ttl_seconds=1000,
            size_bytes=100000,
        )

        buffer.store(msg_short, current_time=0.0)
        buffer.store(msg_long, current_time=0.0)

        # Expire at time 200 (after short TTL, before long)
        expired = buffer.expire_messages(current_time=200.0)

        assert len(expired) == 1
        assert expired[0].message_id == msg_short.message_id
        assert expired[0].status == MessageStatus.EXPIRED
        assert buffer.message_count == 1

    def test_utilisation(self, buffer: MessageBuffer):
        """Test buffer utilisation calculation."""
        assert buffer.utilisation == 0.0

        msg = create_message(
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=18000,
            size_bytes=buffer.capacity_bytes // 2,
        )
        buffer.store(msg, current_time=0.0)

        assert buffer.utilisation == pytest.approx(0.5, rel=0.01)

    def test_iteration(self, buffer: MessageBuffer):
        """Test iterating over messages in buffer."""
        for i in range(3):
            msg = create_message(
                source_id="a",
                destination_id="b",
                message_type=MessageType.STATUS,
                payload=f"test_{i}",
                creation_time=0.0,
                ttl_seconds=18000,
                size_bytes=100000,
            )
            buffer.store(msg, current_time=0.0)

        messages = list(buffer)
        assert len(messages) == 3


# =============================================================================
# Test DeliveryPredictabilityMatrix
# =============================================================================


class TestDeliveryPredictabilityMatrix:
    """Tests for the PRoPHET delivery predictability matrix."""

    @pytest.fixture
    def matrix(self) -> DeliveryPredictabilityMatrix:
        """Create matrix with standard PRoPHET parameters."""
        return DeliveryPredictabilityMatrix(
            p_init=0.75,
            beta=0.25,
            gamma=0.98,
            update_interval=0.1,
        )

    def test_matrix_creation_with_default_params(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test matrix creation with standard parameters (Kumar et al., 2023)."""
        assert matrix.p_init == 0.75
        assert matrix.beta == 0.25
        assert matrix.gamma == 0.98
        assert matrix.update_interval == 0.1

    def test_initial_predictability_is_zero(self, matrix: DeliveryPredictabilityMatrix):
        """Test that initial predictability is zero for unknown destinations."""
        assert matrix.get_predictability("node_a", "node_b") == 0.0

    def test_encounter_update(self, matrix: DeliveryPredictabilityMatrix):
        """Test predictability update on encounter.

        Equation: P(a,b) = P(a,b)_old + (1 - P(a,b)_old) × P_init
        With P_init=0.75 and P(a,b)_old=0:
            P(a,b) = 0 + (1 - 0) × 0.75 = 0.75
        """
        matrix.update_encounter("node_a", "node_b")

        # First encounter should set predictability to P_init
        assert matrix.get_predictability("node_a", "node_b") == 0.75
        assert matrix.get_predictability("node_b", "node_a") == 0.75

    def test_repeated_encounters_increase_predictability(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test that repeated encounters increase predictability.

        After first encounter: P = 0.75
        After second encounter: P = 0.75 + (1 - 0.75) × 0.75 = 0.9375
        """
        matrix.update_encounter("node_a", "node_b")
        first_p = matrix.get_predictability("node_a", "node_b")

        matrix.update_encounter("node_a", "node_b")
        second_p = matrix.get_predictability("node_a", "node_b")

        assert second_p > first_p
        assert second_p == pytest.approx(0.9375, rel=0.001)

    def test_predictability_approaches_one(self, matrix: DeliveryPredictabilityMatrix):
        """Test that predictability approaches 1.0 with many encounters."""
        for _ in range(10):
            matrix.update_encounter("node_a", "node_b")

        p = matrix.get_predictability("node_a", "node_b")
        assert p > 0.99

    def test_transitivity_update(self, matrix: DeliveryPredictabilityMatrix):
        """Test transitivity update.

        If A knows B (P(A,B)) and B knows C (P(B,C)),
        then A gains predictability to C:
        P(A,C) = P(A,C)_old + (1 - P(A,C)_old) × P(A,B) × P(B,C) × β
        """
        # A encounters B
        matrix.update_encounter("node_a", "node_b")

        # B has seen C before
        matrix.set_predictability("node_b", "node_c", 0.75)

        # Apply transitivity when A meets B
        matrix.update_transitivity("node_a", "node_b")

        # A should now have some predictability to C
        p_ac = matrix.get_predictability("node_a", "node_c")

        # P(A,C) = 0 + (1-0) × 0.75 × 0.75 × 0.25 = 0.140625
        expected = 0.75 * 0.75 * 0.25
        assert p_ac == pytest.approx(expected, rel=0.001)

    def test_aging_reduces_predictability(self, matrix: DeliveryPredictabilityMatrix):
        """Test that predictability ages (decays) over time.

        Equation: P = P_old × γ^k where k = time_units
        With γ=0.98 and k=10: P = 0.75 × 0.98^10 ≈ 0.611
        """
        matrix.initialise_node("node_a", current_time=0.0)
        matrix.set_predictability("node_a", "node_b", 0.75)

        # Age by 1 second (10 intervals of 0.1)
        matrix.age_predictabilities("node_a", current_time=1.0)

        p = matrix.get_predictability("node_a", "node_b")
        expected = 0.75 * (0.98**10)
        assert p == pytest.approx(expected, rel=0.001)

    def test_aging_removes_small_predictabilities(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test that very small predictabilities are removed to save memory."""
        matrix.initialise_node("node_a", current_time=0.0)
        matrix.set_predictability("node_a", "node_b", 0.001)  # Very small

        # Age significantly
        matrix.age_predictabilities("node_a", current_time=100.0)

        # Should be removed (threshold is 0.001)
        assert matrix.get_predictability("node_a", "node_b") == 0.0

    def test_get_best_forwarder_direct_delivery(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test that destination is returned if among candidates."""
        result = matrix.get_best_forwarder(
            current_node="node_a",
            destination="node_c",
            candidates=["node_b", "node_c", "node_d"],
        )
        assert result == "node_c"

    def test_get_best_forwarder_by_predictability(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test forwarding to node with highest predictability."""
        matrix.set_predictability("node_a", "dest", 0.3)
        matrix.set_predictability("node_b", "dest", 0.5)
        matrix.set_predictability("node_c", "dest", 0.8)  # Highest

        result = matrix.get_best_forwarder(
            current_node="node_a",
            destination="dest",
            candidates=["node_b", "node_c"],
        )
        assert result == "node_c"

    def test_get_best_forwarder_none_if_not_better(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test that None is returned if no candidate is better."""
        matrix.set_predictability("node_a", "dest", 0.8)  # Current is best
        matrix.set_predictability("node_b", "dest", 0.3)
        matrix.set_predictability("node_c", "dest", 0.5)

        result = matrix.get_best_forwarder(
            current_node="node_a",
            destination="dest",
            candidates=["node_b", "node_c"],
        )
        assert result is None

    def test_get_best_forwarder_empty_candidates(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test with no candidates."""
        result = matrix.get_best_forwarder(
            current_node="node_a",
            destination="dest",
            candidates=[],
        )
        assert result is None

    def test_get_all_predictabilities(self, matrix: DeliveryPredictabilityMatrix):
        """Test getting all predictabilities for a node."""
        matrix.set_predictability("node_a", "dest_1", 0.5)
        matrix.set_predictability("node_a", "dest_2", 0.7)
        matrix.set_predictability("node_a", "dest_3", 0.3)

        all_p = matrix.get_all_predictabilities("node_a")

        assert len(all_p) == 3
        assert all_p["dest_1"] == 0.5
        assert all_p["dest_2"] == 0.7
        assert all_p["dest_3"] == 0.3

    def test_reset_clears_all_data(self, matrix: DeliveryPredictabilityMatrix):
        """Test that reset clears all predictability data."""
        matrix.set_predictability("node_a", "dest", 0.75)
        matrix.initialise_node("node_a")

        matrix.reset()

        assert matrix.node_count == 0
        assert matrix.get_predictability("node_a", "dest") == 0.0

    def test_invalid_parameters_raise_errors(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            DeliveryPredictabilityMatrix(p_init=0)  # Must be > 0

        with pytest.raises(ValueError):
            DeliveryPredictabilityMatrix(p_init=1.5)  # Must be <= 1

        with pytest.raises(ValueError):
            DeliveryPredictabilityMatrix(beta=-0.1)  # Must be > 0

        with pytest.raises(ValueError):
            DeliveryPredictabilityMatrix(gamma=1.0)  # Must be < 1

        with pytest.raises(ValueError):
            DeliveryPredictabilityMatrix(update_interval=0)  # Must be > 0


# =============================================================================
# Test CommunicationLayer
# =============================================================================


class TestCommunicationLayer:
    """Tests for the main CommunicationLayer class."""

    @pytest.fixture
    def comm_params(self) -> CommunicationParameters:
        """Communication parameters matching Phase 2 spec."""
        return CommunicationParameters()

    @pytest.fixture
    def network_params(self) -> NetworkParameters:
        """Network parameters from Phase 1."""
        return NetworkParameters()

    @pytest.fixture
    def node_ids(self) -> list[str]:
        """Simple node ID list for testing."""
        return ["coord_0", "coord_1", "mobile_0", "mobile_1", "mobile_2"]

    @pytest.fixture
    def comm_layer(
        self,
        comm_params: CommunicationParameters,
        network_params: NetworkParameters,
        node_ids: list[str],
    ) -> CommunicationLayer:
        """Create a communication layer instance."""
        return CommunicationLayer(comm_params, network_params, node_ids)

    def test_layer_initialisation(
        self,
        comm_layer: CommunicationLayer,
        node_ids: list[str],
    ):
        """Test communication layer initialisation."""
        # Should have buffers for all nodes
        assert len(comm_layer.buffers) == len(node_ids)
        for node_id in node_ids:
            assert node_id in comm_layer.buffers

        # Predictability should be initialised
        assert comm_layer.predictability.node_count == len(node_ids)

    def test_create_message(self, comm_layer: CommunicationLayer):
        """Test creating a message through the layer."""
        msg = comm_layer.create_message(
            source_id="mobile_0",
            destination_id="coord_0",
            message_type=MessageType.COORDINATION,
            payload={"task": "test"},
            current_time=0.0,
            urgency_level="H",
        )

        assert msg.source_id == "mobile_0"
        assert msg.destination_id == "coord_0"
        assert msg.urgency_level == "H"

        # Should be in source buffer
        buffer = comm_layer.get_buffer("mobile_0")
        assert buffer.has_message(msg.message_id)

    def test_create_message_unknown_source_raises(self, comm_layer: CommunicationLayer):
        """Test that creating message from unknown source raises error."""
        with pytest.raises(ValueError, match="Unknown source node"):
            comm_layer.create_message(
                source_id="unknown_node",
                destination_id="coord_0",
                message_type=MessageType.STATUS,
                payload="test",
                current_time=0.0,
            )

    def test_process_encounter_updates_predictability(
        self, comm_layer: CommunicationLayer
    ):
        """Test that encounters update predictability."""
        # Initially no predictability
        p_before = comm_layer.get_delivery_predictability("mobile_0", "mobile_1")
        assert p_before == 0.0

        # Process encounter
        comm_layer.process_encounter("mobile_0", "mobile_1", current_time=0.0)

        # Should now have predictability
        p_after = comm_layer.get_delivery_predictability("mobile_0", "mobile_1")
        assert p_after == 0.75  # P_init

    def test_direct_delivery(self, comm_layer: CommunicationLayer):
        """Test direct message delivery when destination is encountered."""
        # Create message to mobile_1
        msg = comm_layer.create_message(
            source_id="mobile_0",
            destination_id="mobile_1",
            message_type=MessageType.COORDINATION,
            payload="test",
            current_time=0.0,
        )

        # Encounter with destination
        results = comm_layer.process_encounter("mobile_0", "mobile_1", current_time=1.0)

        # Should have delivered
        delivered = [r for r in results if r.reason == "delivered"]
        assert len(delivered) == 1
        assert delivered[0].message.message_id == msg.message_id

        # Check statistics
        stats = comm_layer.statistics
        assert stats["messages_delivered"] == 1

    def test_message_forwarding(self, comm_layer: CommunicationLayer):
        """Test message forwarding to node with higher predictability."""
        # Set up predictabilities
        # mobile_0 has low predictability to coord_0
        comm_layer.predictability.set_predictability("mobile_0", "coord_0", 0.2)
        # mobile_1 has high predictability to coord_0
        comm_layer.predictability.set_predictability("mobile_1", "coord_0", 0.9)

        # Create message
        msg = comm_layer.create_message(
            source_id="mobile_0",
            destination_id="coord_0",
            message_type=MessageType.COORDINATION,
            payload="test",
            current_time=0.0,
        )

        # Encounter with mobile_1 (better forwarder)
        results = comm_layer.process_encounter("mobile_0", "mobile_1", current_time=1.0)

        # Should have forwarded
        forwarded = [r for r in results if r.reason == "forwarded"]
        assert len(forwarded) == 1

        # mobile_1 should now have the message
        buffer = comm_layer.get_buffer("mobile_1")
        assert buffer.has_message(msg.message_id)

    def test_no_forward_to_worse_node(self, comm_layer: CommunicationLayer):
        """Test that messages are not forwarded to nodes with lower predictability."""
        # mobile_0 has high predictability
        comm_layer.predictability.set_predictability("mobile_0", "coord_0", 0.9)
        # mobile_1 has low predictability
        comm_layer.predictability.set_predictability("mobile_1", "coord_0", 0.2)

        msg = comm_layer.create_message(
            source_id="mobile_0",
            destination_id="coord_0",
            message_type=MessageType.COORDINATION,
            payload="test",
            current_time=0.0,
        )

        # Encounter with mobile_1 (worse forwarder)
        results = comm_layer.process_encounter("mobile_0", "mobile_1", current_time=1.0)

        # Should not have forwarded (message stays with mobile_0)
        forwarded = [r for r in results if r.reason == "forwarded"]
        assert len(forwarded) == 0

        # mobile_1 should NOT have the message
        buffer = comm_layer.get_buffer("mobile_1")
        assert not buffer.has_message(msg.message_id)

    def test_expire_all_messages(self, comm_layer: CommunicationLayer):
        """Test expiring messages across all buffers."""
        # Create messages with short TTL
        for source in ["mobile_0", "mobile_1"]:
            comm_layer.create_message(
                source_id=source,
                destination_id="coord_0",
                message_type=MessageType.STATUS,
                payload="test",
                current_time=0.0,
            )

        # Modify TTL directly for testing (normally done at creation)
        for buffer in comm_layer.buffers.values():
            for msg in buffer:
                msg.ttl_seconds = 100

        # Expire at time 200
        expired_count = comm_layer.expire_all_messages(current_time=200.0)

        assert expired_count == 2

    def test_statistics(self, comm_layer: CommunicationLayer):
        """Test statistics collection."""
        # Create and deliver some messages
        for i in range(3):
            comm_layer.create_message(
                source_id="mobile_0",
                destination_id="mobile_1",
                message_type=MessageType.STATUS,
                payload=f"test_{i}",
                current_time=float(i),
            )

        comm_layer.process_encounter("mobile_0", "mobile_1", current_time=10.0)

        stats = comm_layer.statistics
        assert stats["messages_created"] == 3
        assert stats["messages_delivered"] == 3
        assert stats["delivery_rate"] == 1.0

    def test_transmission_time_calculation(self, comm_layer: CommunicationLayer):
        """Test transmission time calculation (2 Mbps).

        500 KB = 500,000 bytes = 4,000,000 bits
        At 2 Mbps: 4,000,000 / 2,000,000 = 2 seconds
        """
        size_bytes = 500_000
        expected_time = (size_bytes * 8) / 2_000_000  # 2 seconds

        time = comm_layer._calculate_transmission_time(size_bytes)
        assert time == expected_time


# =============================================================================
# Test Parameter Verification (Phase 2 Spec)
# =============================================================================


class TestPhase2Parameters:
    """Verify Phase 2 parameters match specification."""

    def test_prophet_p_init(self):
        """Verify P_init = 0.75 (Kumar et al., 2023)."""
        params = PRoPHETParameters()
        assert params.p_init == 0.75

    def test_prophet_beta(self):
        """Verify β = 0.25 (Kumar et al., 2023)."""
        params = PRoPHETParameters()
        assert params.beta == 0.25

    def test_prophet_gamma(self):
        """Verify γ = 0.98 (Kumar et al., 2023)."""
        params = PRoPHETParameters()
        assert params.gamma == 0.98

    def test_message_ttl(self):
        """Verify message TTL = 300 minutes (Ullah & Qayyum, 2022)."""
        params = CommunicationParameters()
        assert params.message_ttl_seconds == 18000  # 300 * 60

    def test_transmit_speed(self):
        """Verify transmit speed = 2 Mbps (Ullah & Qayyum, 2022)."""
        params = CommunicationParameters()
        assert params.transmit_speed_bps == 2_000_000

    def test_update_interval(self):
        """Verify update interval = 30.0s (aging time unit, Kumar et al., 2023)."""
        params = CommunicationParameters()
        assert params.update_interval_seconds == 30.0

    def test_buffer_drop_policy(self):
        """Verify buffer drop policy = drop_oldest (Ullah & Qayyum, 2022)."""
        params = CommunicationParameters()
        assert params.buffer_drop_policy == BufferDropPolicy.DROP_OLDEST


# =============================================================================
# Integration Tests
# =============================================================================


class TestCommunicationIntegration:
    """Integration tests for communication layer scenarios."""

    @pytest.fixture
    def full_comm_layer(self) -> CommunicationLayer:
        """Create a fully configured communication layer."""
        comm_params = CommunicationParameters()
        network_params = NetworkParameters()

        # Create realistic node set
        node_ids = ["coord_0", "coord_1"] + [f"mobile_{i}" for i in range(10)]

        return CommunicationLayer(comm_params, network_params, node_ids)

    def test_multi_hop_delivery(self, full_comm_layer: CommunicationLayer):
        """Test message delivery through multiple hops."""
        layer = full_comm_layer

        # Build predictability chain: mobile_0 -> mobile_1 -> coord_0
        # Set high predictabilities that will remain valid after encounter updates
        layer.predictability.set_predictability("mobile_0", "coord_0", 0.1)
        layer.predictability.set_predictability("mobile_1", "coord_0", 0.8)
        layer.predictability.set_predictability("mobile_2", "coord_0", 0.95)

        # Create message at mobile_0
        msg = layer.create_message(
            source_id="mobile_0",
            destination_id="coord_0",
            message_type=MessageType.COORDINATION,
            payload="urgent",
            current_time=0.0,
        )

        # Hop 1: mobile_0 meets mobile_1 (at time 0, no aging)
        layer.process_encounter("mobile_0", "mobile_1", current_time=0.0)

        # Message should be at mobile_1 (forwarded because mobile_1 has higher P to coord_0)
        assert layer.get_buffer("mobile_1").has_message(msg.message_id)

        # Hop 2: mobile_1 meets mobile_2 (at time 0, no aging)
        layer.process_encounter("mobile_1", "mobile_2", current_time=0.0)

        # Message should be at mobile_2 (forwarded because mobile_2 has higher P to coord_0)
        assert layer.get_buffer("mobile_2").has_message(msg.message_id)

        # Hop 3: mobile_2 meets coord_0 (direct delivery)
        results = layer.process_encounter("mobile_2", "coord_0", current_time=0.0)

        # Should be delivered
        delivered = [r for r in results if r.reason == "delivered"]
        assert len(delivered) == 1

        # Check statistics
        assert layer.statistics["messages_delivered"] == 1

    def test_multiple_messages_same_encounter(
        self, full_comm_layer: CommunicationLayer
    ):
        """Test handling multiple messages in a single encounter."""
        layer = full_comm_layer

        # Create multiple messages
        for i in range(5):
            layer.create_message(
                source_id="mobile_0",
                destination_id="mobile_1",
                message_type=MessageType.STATUS,
                payload=f"msg_{i}",
                current_time=float(i),
            )

        # All should be delivered in one encounter
        results = layer.process_encounter("mobile_0", "mobile_1", current_time=100.0)
        delivered = [r for r in results if r.reason == "delivered"]

        assert len(delivered) == 5
        assert layer.statistics["messages_delivered"] == 5

    def test_predictability_evolution_over_time(
        self, full_comm_layer: CommunicationLayer
    ):
        """Test predictability changes through encounters and aging."""
        layer = full_comm_layer

        # First encounter at time 0
        layer.process_encounter("mobile_0", "mobile_1", current_time=0.0)
        p1 = layer.get_delivery_predictability("mobile_0", "mobile_1")
        assert p1 == 0.75

        # Second encounter at same time (should increase, no aging)
        layer.process_encounter("mobile_0", "mobile_1", current_time=0.0)
        p2 = layer.get_delivery_predictability("mobile_0", "mobile_1")
        assert p2 > p1  # Should increase due to encounter update

        # Age directly without encounter (should decrease)
        # Elapsed must exceed update_interval (30s) for aging to apply (k >= 1)
        layer.predictability._last_aging_time["mobile_0"] = 0.0
        layer.predictability.age_predictabilities("mobile_0", current_time=60.0)
        p3 = layer.get_delivery_predictability("mobile_0", "mobile_1")
        assert p3 < p2  # Should decrease due to aging

    def test_buffer_overflow_scenario(self, full_comm_layer: CommunicationLayer):
        """Test buffer overflow with many messages."""
        layer = full_comm_layer
        buffer = layer.get_buffer("mobile_0")

        # Calculate how many messages fit in buffer
        msg_size = layer.network_params.message_size_bytes
        buffer_size = layer.network_params.buffer_size_bytes
        max_messages = buffer_size // msg_size

        # Create more messages than buffer can hold
        for i in range(max_messages + 5):
            layer.create_message(
                source_id="mobile_0",
                destination_id="coord_0",
                message_type=MessageType.STATUS,
                payload=f"test_{i}",
                current_time=float(i),
            )

        # Buffer should have dropped some
        assert buffer.dropped_count >= 5
        assert buffer.message_count <= max_messages
