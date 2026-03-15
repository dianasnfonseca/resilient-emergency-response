"""
Tests for Communication Layer (Phase 2) — PRoPHETv2.

These tests verify that the PRoPHETv2 communication layer correctly
implements the specification:
- Time-based encounter updates (P_enc depends on inter-encounter interval)
- MAX-based transitivity (prevents saturation)
- Greater-or-equal forwarding condition
- Parameter defaults: P_enc_max=0.5, I_typ=1800, β=0.9, γ=0.999885791

Additional parameters:
- Message TTL: 300 minutes (18000 seconds)
- Transmit speed: 2 Mbps
- Buffer drop policy: drop oldest
"""

import pytest
import numpy as np

from ercs.config.parameters import (
    CommunicationParameters,
    NetworkParameters,
    PRoPHETParameters,
    BufferDropPolicy,
)
from conftest import (
    AGING_INTERVAL_S,
    BETA,
    BUFFER_SIZE_BYTES,
    GAMMA,
    I_TYP,
    MESSAGE_SIZE_BYTES,
    MESSAGE_TTL_S,
    P_ENC_MAX,
    TRANSMIT_SPEED_BPS,
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
            ttl_seconds=MESSAGE_TTL_S,
            size_bytes=MESSAGE_SIZE_BYTES,
        )

        assert msg.message_id == "msg_001"
        assert msg.source_id == "node_a"
        assert msg.destination_id == "node_b"
        assert msg.message_type == MessageType.COORDINATION
        assert msg.creation_time == 100.0
        assert msg.ttl_seconds == MESSAGE_TTL_S
        assert msg.size_bytes == MESSAGE_SIZE_BYTES
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
            ttl_seconds=MESSAGE_TTL_S,
            size_bytes=MESSAGE_SIZE_BYTES,
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
            ttl_seconds=MESSAGE_TTL_S,  # 300 minutes
            size_bytes=MESSAGE_SIZE_BYTES,
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
            ttl_seconds=MESSAGE_TTL_S,
            size_bytes=MESSAGE_SIZE_BYTES,
        )

        assert msg.remaining_ttl(0.0) == MESSAGE_TTL_S
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
            ttl_seconds=MESSAGE_TTL_S,
            size_bytes=MESSAGE_SIZE_BYTES,
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
            ttl_seconds=MESSAGE_TTL_S,
            size_bytes=MESSAGE_SIZE_BYTES,
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
            ttl_seconds=MESSAGE_TTL_S,
            size_bytes=MESSAGE_SIZE_BYTES,
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
            ttl_seconds=MESSAGE_TTL_S,
            size_bytes=MESSAGE_SIZE_BYTES,
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
                ttl_seconds=MESSAGE_TTL_S,
                size_bytes=MESSAGE_SIZE_BYTES,
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
            capacity_bytes=BUFFER_SIZE_BYTES,
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
        assert buffer.capacity_bytes == BUFFER_SIZE_BYTES
        assert buffer.used_bytes == 0
        assert buffer.available_bytes == BUFFER_SIZE_BYTES
        assert buffer.message_count == 0

    def test_store_message(self, buffer: MessageBuffer):
        """Test storing a message in buffer."""
        msg = create_message(
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=MESSAGE_TTL_S,
            size_bytes=MESSAGE_SIZE_BYTES,
        )

        result = buffer.store(msg, current_time=0.0)

        assert result is True
        assert buffer.message_count == 1
        assert buffer.used_bytes == MESSAGE_SIZE_BYTES
        assert buffer.has_message(msg.message_id)

    def test_store_duplicate_ignored(self, buffer: MessageBuffer):
        """Test that duplicate messages are not stored twice."""
        msg = create_message(
            source_id="a",
            destination_id="b",
            message_type=MessageType.STATUS,
            payload="test",
            creation_time=0.0,
            ttl_seconds=MESSAGE_TTL_S,
            size_bytes=MESSAGE_SIZE_BYTES,
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
            size_bytes=MESSAGE_SIZE_BYTES,
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
                ttl_seconds=MESSAGE_TTL_S,
                size_bytes=MESSAGE_SIZE_BYTES,
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
            ttl_seconds=MESSAGE_TTL_S,
            size_bytes=MESSAGE_SIZE_BYTES,
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
            ttl_seconds=MESSAGE_TTL_S,
            size_bytes=MESSAGE_SIZE_BYTES,
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
                ttl_seconds=MESSAGE_TTL_S,
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
            ttl_seconds=MESSAGE_TTL_S,
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
                ttl_seconds=MESSAGE_TTL_S,
                size_bytes=100000,
            )
            buffer.store(msg, current_time=0.0)

        messages = list(buffer)
        assert len(messages) == 3


# =============================================================================
# Test DeliveryPredictabilityMatrix — PRoPHETv2
# =============================================================================


class TestDeliveryPredictabilityMatrix:
    """Tests for the PRoPHETv2 delivery predictability matrix."""

    @pytest.fixture
    def matrix(self) -> DeliveryPredictabilityMatrix:
        """Create matrix with PRoPHETv2 parameters."""
        return DeliveryPredictabilityMatrix(
            p_enc_max=P_ENC_MAX,
            i_typ=I_TYP,
            beta=BETA,
            gamma=GAMMA,
            update_interval=AGING_INTERVAL_S,
        )

    def test_matrix_creation_with_prophetv2_params(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test matrix creation with PRoPHETv2 parameters."""
        assert matrix.p_enc_max == P_ENC_MAX
        assert matrix.i_typ == I_TYP
        assert matrix.beta == BETA
        assert matrix.gamma == GAMMA
        assert matrix.update_interval == AGING_INTERVAL_S

    def test_initial_predictability_is_zero(self, matrix: DeliveryPredictabilityMatrix):
        """Test that initial predictability is zero for unknown destinations."""
        assert matrix.get_predictability("node_a", "node_b") == 0.0

    def test_first_encounter_uses_p_enc_max(self, matrix: DeliveryPredictabilityMatrix):
        """Test first encounter uses full P_enc_max.

        PRoPHETv2: First encounter → P_enc = P_enc_max
        P(a,b) = 0 + (1 - 0) × 0.5 = 0.5
        """
        matrix.update_encounter("node_a", "node_b", current_time=100.0)

        p_ab = matrix.get_predictability("node_a", "node_b")
        p_ba = matrix.get_predictability("node_b", "node_a")

        assert p_ab == pytest.approx(P_ENC_MAX, rel=0.001)
        assert p_ba == pytest.approx(P_ENC_MAX, rel=0.001)

    def test_rapid_re_encounter_reduces_p_enc(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test that re-encountering quickly reduces P_enc (PRoPHETv2 Eq. 1).

        First encounter at t=100: P_enc = 0.5 → P = 0.5
        Second encounter at t=200 (Δt=100 < I_typ=1800):
            P_enc = 0.5 × (100/1800) ≈ 0.0278
            P = 0.5 + (1 - 0.5) × 0.0278 ≈ 0.5139
        """
        matrix.update_encounter("node_a", "node_b", current_time=100.0)
        p_first = matrix.get_predictability("node_a", "node_b")

        matrix.update_encounter("node_a", "node_b", current_time=200.0)
        p_second = matrix.get_predictability("node_a", "node_b")

        assert p_second > p_first  # Still increases
        # But only slightly — P_enc was scaled down
        delta_t = 100.0
        expected_p_enc = P_ENC_MAX * (delta_t / I_TYP)
        expected_p = p_first + (1 - p_first) * expected_p_enc
        assert p_second == pytest.approx(expected_p, rel=0.001)

    def test_long_gap_encounter_uses_full_p_enc(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test that encounter after I_typ seconds uses full P_enc_max.

        First encounter at t=0: P = 0.5
        Second encounter at t=2000 (Δt=2000 > I_typ=1800):
            P_enc = P_enc_max = 0.5
            P = 0.5 + (1 - 0.5) × 0.5 = 0.75
        """
        matrix.update_encounter("node_a", "node_b", current_time=0.0)
        p_first = matrix.get_predictability("node_a", "node_b")

        matrix.update_encounter("node_a", "node_b", current_time=2000.0)
        p_second = matrix.get_predictability("node_a", "node_b")

        expected = p_first + (1 - p_first) * P_ENC_MAX
        assert p_second == pytest.approx(expected, rel=0.001)

    def test_saturation_prevention(self, matrix: DeliveryPredictabilityMatrix):
        """Test PRoPHETv2 prevents delivery predictability saturation.

        With rapid re-encounters (Δt << I_typ), P_enc is scaled down
        significantly, preventing P from converging to ~1.0.
        """
        # Simulate 50 rapid encounters (every 10 seconds)
        for i in range(50):
            matrix.update_encounter("node_a", "node_b", current_time=float(i * 10))

        p = matrix.get_predictability("node_a", "node_b")

        # PRoPHETv2 should NOT saturate — P should stay well below 1.0
        # With 10s intervals and I_typ=1800, P_enc ≈ 0.5*(10/1800) ≈ 0.0028
        assert p < 0.9, f"P-value {p:.4f} saturated despite PRoPHETv2"

    def test_max_transitivity(self, matrix: DeliveryPredictabilityMatrix):
        """Test MAX-based transitivity update (PRoPHETv2 Eq. 2).

        If A knows B (P(A,B)) and B knows C (P(B,C)),
        then: P(A,C) = max(P(A,C)_old, P(A,B) × P(B,C) × β)

        With P(A,B)=0.5, P(B,C)=0.5, β=0.9:
            P(A,C) = max(0, 0.5 × 0.5 × 0.9) = 0.225
        """
        matrix.update_encounter("node_a", "node_b", current_time=100.0)
        matrix.set_predictability("node_b", "node_c", 0.5)

        matrix.update_transitivity("node_a", "node_b")

        p_ac = matrix.get_predictability("node_a", "node_c")
        expected = P_ENC_MAX * 0.5 * BETA  # 0.5 × 0.5 × 0.9 = 0.225
        assert p_ac == pytest.approx(expected, rel=0.001)

    def test_max_transitivity_does_not_decrease(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test MAX-based transitivity never decreases existing predictability.

        If P(A,C) is already high, a low transitive value should not reduce it.
        """
        matrix.set_predictability("node_a", "node_c", 0.8)  # Already high
        matrix.set_predictability("node_a", "node_b", 0.3)  # Low A→B
        matrix.set_predictability("node_b", "node_c", 0.3)  # Low B→C

        matrix.update_transitivity("node_a", "node_b")

        p_ac = matrix.get_predictability("node_a", "node_c")
        # transitive = 0.3 × 0.3 × 0.9 = 0.081 < 0.8 → no change
        assert p_ac == pytest.approx(0.8, rel=0.001)

    def test_max_transitivity_increases_when_higher(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test MAX-based transitivity increases P when transitive value is higher."""
        matrix.set_predictability("node_a", "node_c", 0.1)  # Low existing
        matrix.set_predictability("node_a", "node_b", 0.8)  # High A→B
        matrix.set_predictability("node_b", "node_c", 0.8)  # High B→C

        matrix.update_transitivity("node_a", "node_b")

        p_ac = matrix.get_predictability("node_a", "node_c")
        # transitive = 0.8 × 0.8 × 0.9 = 0.576 > 0.1 → update
        expected = 0.8 * 0.8 * BETA
        assert p_ac == pytest.approx(expected, rel=0.001)

    def test_aging_reduces_predictability(self, matrix: DeliveryPredictabilityMatrix):
        """Test that predictability ages (decays) over time.

        Equation: P = P_old × γ^k where k = elapsed / update_interval
        With γ=0.999885791 and 60s elapsed (k=2 at 30s interval):
            P = 0.5 × 0.999885791^2 ≈ 0.49989
        """
        matrix.initialise_node("node_a", current_time=0.0)
        matrix.set_predictability("node_a", "node_b", P_ENC_MAX)

        # Age by 60s (k=2 at 30s interval)
        matrix.age_predictabilities("node_a", current_time=60.0)

        p = matrix.get_predictability("node_a", "node_b")
        k = 60.0 / AGING_INTERVAL_S
        expected = P_ENC_MAX * (GAMMA ** k)
        assert p == pytest.approx(expected, rel=0.001)

    def test_aging_slow_decay_with_prophetv2_gamma(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test PRoPHETv2 gamma decays much slower than original PRoPHET.

        PRoPHETv2 γ=0.999885791 vs original γ=0.98.
        After 300s (k=10): PRoPHETv2 P ≈ 0.5 × 0.99886 ≈ 0.4994
        vs original: P ≈ 0.5 × 0.817 ≈ 0.409
        """
        matrix.initialise_node("node_a", current_time=0.0)
        matrix.set_predictability("node_a", "node_b", P_ENC_MAX)

        matrix.age_predictabilities("node_a", current_time=300.0)

        p = matrix.get_predictability("node_a", "node_b")
        # Should still be close to original value due to slow decay
        assert p > 0.49, f"P={p:.4f} decayed too fast for PRoPHETv2 gamma"

    def test_aging_removes_small_predictabilities(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test that very small predictabilities are removed to save memory."""
        matrix.initialise_node("node_a", current_time=0.0)
        matrix.set_predictability("node_a", "node_b", 0.001)  # Very small

        # With PRoPHETv2 gamma, need very long time to decay below threshold
        # k = elapsed / 30; need 0.001 × γ^k < 0.001 → γ^k < 1
        # Any aging will reduce it, but we need enough time
        matrix.age_predictabilities("node_a", current_time=100000.0)

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

    def test_get_best_forwarder_equal_predictability(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test PRoPHETv2 >= forwarding: equal P triggers forwarding."""
        matrix.set_predictability("node_a", "dest", 0.5)
        matrix.set_predictability("node_b", "dest", 0.5)  # Equal

        result = matrix.get_best_forwarder(
            current_node="node_a",
            destination="dest",
            candidates=["node_b"],
        )
        # PRoPHETv2 uses >= so equal P should trigger forwarding
        assert result == "node_b"

    def test_get_best_forwarder_none_if_not_better(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test that None is returned if no candidate is better or equal."""
        matrix.set_predictability("node_a", "dest", 0.8)  # Current is best
        matrix.set_predictability("node_b", "dest", 0.3)
        matrix.set_predictability("node_c", "dest", 0.5)

        result = matrix.get_best_forwarder(
            current_node="node_a",
            destination="dest",
            candidates=["node_b", "node_c"],
        )
        assert result is None

    def test_get_best_forwarder_zero_predictability_not_forwarded(
        self, matrix: DeliveryPredictabilityMatrix
    ):
        """Test that zero predictability doesn't trigger forwarding."""
        # Both have P=0 — should not forward
        result = matrix.get_best_forwarder(
            current_node="node_a",
            destination="dest",
            candidates=["node_b"],
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
        """Test that reset clears all predictability data including encounter times."""
        matrix.set_predictability("node_a", "dest", 0.5)
        matrix.initialise_node("node_a")
        matrix.update_encounter("node_a", "dest", current_time=100.0)

        matrix.reset()

        assert matrix.node_count == 0
        assert matrix.get_predictability("node_a", "dest") == 0.0
        assert len(matrix._last_encounter_time) == 0

    def test_invalid_parameters_raise_errors(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            DeliveryPredictabilityMatrix(p_enc_max=0)  # Must be > 0

        with pytest.raises(ValueError):
            DeliveryPredictabilityMatrix(p_enc_max=1.5)  # Must be <= 1

        with pytest.raises(ValueError):
            DeliveryPredictabilityMatrix(i_typ=0)  # Must be > 0

        with pytest.raises(ValueError):
            DeliveryPredictabilityMatrix(beta=-0.1)  # Must be > 0

        with pytest.raises(ValueError):
            DeliveryPredictabilityMatrix(gamma=1.0)  # Must be < 1

        with pytest.raises(ValueError):
            DeliveryPredictabilityMatrix(update_interval=0)  # Must be > 0

    def test_encounter_time_tracking(self, matrix: DeliveryPredictabilityMatrix):
        """Test that last encounter times are tracked per directed pair."""
        matrix.update_encounter("node_a", "node_b", current_time=100.0)

        assert matrix._last_encounter_time[("node_a", "node_b")] == 100.0
        assert matrix._last_encounter_time[("node_b", "node_a")] == 100.0

        matrix.update_encounter("node_a", "node_b", current_time=200.0)

        assert matrix._last_encounter_time[("node_a", "node_b")] == 200.0


# =============================================================================
# Test CommunicationLayer
# =============================================================================


class TestCommunicationLayer:
    """Tests for the main CommunicationLayer class."""

    @pytest.fixture
    def comm_params(self) -> CommunicationParameters:
        """Communication parameters matching PRoPHETv2 spec."""
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
        """Test that encounters update predictability (PRoPHETv2)."""
        p_before = comm_layer.get_delivery_predictability("mobile_0", "mobile_1")
        assert p_before == 0.0

        comm_layer.process_encounter("mobile_0", "mobile_1", current_time=100.0)

        p_after = comm_layer.get_delivery_predictability("mobile_0", "mobile_1")
        assert p_after == pytest.approx(P_ENC_MAX, rel=0.001)

    def test_direct_delivery(self, comm_layer: CommunicationLayer):
        """Test direct message delivery when destination is encountered."""
        msg = comm_layer.create_message(
            source_id="mobile_0",
            destination_id="mobile_1",
            message_type=MessageType.COORDINATION,
            payload="test",
            current_time=0.0,
        )

        results = comm_layer.process_encounter("mobile_0", "mobile_1", current_time=1.0)

        delivered = [r for r in results if r.reason == "delivered"]
        assert len(delivered) == 1
        assert delivered[0].message.message_id == msg.message_id

        stats = comm_layer.statistics
        assert stats["messages_delivered"] == 1

    def test_message_forwarding(self, comm_layer: CommunicationLayer):
        """Test message forwarding to node with higher predictability."""
        comm_layer.predictability.set_predictability("mobile_0", "coord_0", 0.2)
        comm_layer.predictability.set_predictability("mobile_1", "coord_0", 0.9)

        msg = comm_layer.create_message(
            source_id="mobile_0",
            destination_id="coord_0",
            message_type=MessageType.COORDINATION,
            payload="test",
            current_time=0.0,
        )

        results = comm_layer.process_encounter("mobile_0", "mobile_1", current_time=1.0)

        forwarded = [r for r in results if r.reason == "forwarded"]
        assert len(forwarded) == 1

        buffer = comm_layer.get_buffer("mobile_1")
        assert buffer.has_message(msg.message_id)

    def test_forwarding_with_equal_predictability(self, comm_layer: CommunicationLayer):
        """Test PRoPHETv2 >= forwarding with equal predictability."""
        comm_layer.predictability.set_predictability("mobile_0", "coord_0", 0.5)
        comm_layer.predictability.set_predictability("mobile_1", "coord_0", 0.5)

        msg = comm_layer.create_message(
            source_id="mobile_0",
            destination_id="coord_0",
            message_type=MessageType.COORDINATION,
            payload="test",
            current_time=0.0,
        )

        results = comm_layer.process_encounter("mobile_0", "mobile_1", current_time=1.0)

        # PRoPHETv2: equal P should forward (>= condition)
        forwarded = [r for r in results if r.reason == "forwarded"]
        assert len(forwarded) == 1

    def test_no_forward_to_worse_node(self, comm_layer: CommunicationLayer):
        """Test that messages are not forwarded to nodes with lower predictability."""
        comm_layer.predictability.set_predictability("mobile_0", "coord_0", 0.9)
        comm_layer.predictability.set_predictability("mobile_1", "coord_0", 0.2)

        msg = comm_layer.create_message(
            source_id="mobile_0",
            destination_id="coord_0",
            message_type=MessageType.COORDINATION,
            payload="test",
            current_time=0.0,
        )

        results = comm_layer.process_encounter("mobile_0", "mobile_1", current_time=1.0)

        forwarded = [r for r in results if r.reason == "forwarded"]
        assert len(forwarded) == 0

        buffer = comm_layer.get_buffer("mobile_1")
        assert not buffer.has_message(msg.message_id)

    def test_no_forward_when_both_zero(self, comm_layer: CommunicationLayer):
        """Test no forwarding when both nodes have zero predictability."""
        msg = comm_layer.create_message(
            source_id="mobile_0",
            destination_id="coord_0",
            message_type=MessageType.COORDINATION,
            payload="test",
            current_time=0.0,
        )

        # Before any encounters, both have P=0 to coord_0
        # The encounter will set P(mobile_0, mobile_1) but NOT P(mobile_x, coord_0)
        # So both still have P=0 to coord_0 — should NOT forward
        results = comm_layer.process_encounter("mobile_0", "mobile_1", current_time=1.0)

        forwarded = [r for r in results if r.reason == "forwarded"]
        assert len(forwarded) == 0

    def test_expire_all_messages(self, comm_layer: CommunicationLayer):
        """Test expiring messages across all buffers."""
        for source in ["mobile_0", "mobile_1"]:
            comm_layer.create_message(
                source_id=source,
                destination_id="coord_0",
                message_type=MessageType.STATUS,
                payload="test",
                current_time=0.0,
            )

        for buffer in comm_layer.buffers.values():
            for msg in buffer:
                msg.ttl_seconds = 100

        expired_count = comm_layer.expire_all_messages(current_time=200.0)

        assert expired_count == 2

    def test_statistics(self, comm_layer: CommunicationLayer):
        """Test statistics collection."""
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
        expected_time = (size_bytes * 8) / TRANSMIT_SPEED_BPS  # 2 seconds

        time = comm_layer._calculate_transmission_time(size_bytes)
        assert time == expected_time


# =============================================================================
# Test Parameter Verification (PRoPHETv2 Spec)
# =============================================================================


class TestPRoPHETv2Parameters:
    """Verify PRoPHETv2 parameters match specification."""

    def test_prophet_p_enc_max(self):
        """Verify P_enc_max = 0.5."""
        params = PRoPHETParameters()
        assert params.p_enc_max == P_ENC_MAX

    def test_prophet_i_typ(self):
        """Verify I_typ = 1800.0."""
        params = PRoPHETParameters()
        assert params.i_typ == I_TYP

    def test_prophet_beta(self):
        """Verify β = 0.9."""
        params = PRoPHETParameters()
        assert params.beta == BETA

    def test_prophet_gamma(self):
        """Verify γ = 0.999885791."""
        params = PRoPHETParameters()
        assert params.gamma == GAMMA

    def test_message_ttl(self):
        """Verify message TTL = 300 minutes."""
        params = CommunicationParameters()
        assert params.message_ttl_seconds == MESSAGE_TTL_S

    def test_transmit_speed(self):
        """Verify transmit speed = 2 Mbps."""
        params = CommunicationParameters()
        assert params.transmit_speed_bps == TRANSMIT_SPEED_BPS

    def test_update_interval(self):
        """Verify update interval = 30.0s (aging time unit)."""
        params = CommunicationParameters()
        assert params.update_interval_seconds == AGING_INTERVAL_S

    def test_buffer_drop_policy(self):
        """Verify buffer drop policy = drop_oldest."""
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

        node_ids = ["coord_0", "coord_1"] + [f"mobile_{i}" for i in range(10)]

        return CommunicationLayer(comm_params, network_params, node_ids)

    def test_multi_hop_delivery(self, full_comm_layer: CommunicationLayer):
        """Test message delivery through multiple hops."""
        layer = full_comm_layer

        layer.predictability.set_predictability("mobile_0", "coord_0", 0.1)
        layer.predictability.set_predictability("mobile_1", "coord_0", 0.8)
        layer.predictability.set_predictability("mobile_2", "coord_0", 0.95)

        msg = layer.create_message(
            source_id="mobile_0",
            destination_id="coord_0",
            message_type=MessageType.COORDINATION,
            payload="urgent",
            current_time=0.0,
        )

        # Hop 1: mobile_0 meets mobile_1
        layer.process_encounter("mobile_0", "mobile_1", current_time=100.0)
        assert layer.get_buffer("mobile_1").has_message(msg.message_id)

        # Hop 2: mobile_1 meets mobile_2
        layer.process_encounter("mobile_1", "mobile_2", current_time=200.0)
        assert layer.get_buffer("mobile_2").has_message(msg.message_id)

        # Hop 3: mobile_2 meets coord_0 (direct delivery)
        results = layer.process_encounter("mobile_2", "coord_0", current_time=300.0)

        delivered = [r for r in results if r.reason == "delivered"]
        assert len(delivered) == 1
        assert layer.statistics["messages_delivered"] == 1

    def test_multiple_messages_same_encounter(
        self, full_comm_layer: CommunicationLayer
    ):
        """Test handling multiple messages in a single encounter."""
        layer = full_comm_layer

        for i in range(5):
            layer.create_message(
                source_id="mobile_0",
                destination_id="mobile_1",
                message_type=MessageType.STATUS,
                payload=f"msg_{i}",
                current_time=float(i),
            )

        results = layer.process_encounter("mobile_0", "mobile_1", current_time=100.0)
        delivered = [r for r in results if r.reason == "delivered"]

        assert len(delivered) == 5
        assert layer.statistics["messages_delivered"] == 5

    def test_predictability_evolution_over_time(
        self, full_comm_layer: CommunicationLayer
    ):
        """Test predictability changes through encounters and aging."""
        layer = full_comm_layer

        # First encounter
        layer.process_encounter("mobile_0", "mobile_1", current_time=100.0)
        p1 = layer.get_delivery_predictability("mobile_0", "mobile_1")
        assert p1 == pytest.approx(P_ENC_MAX, rel=0.001)

        # Second encounter after long gap (> I_typ) → full P_enc_max again
        layer.process_encounter("mobile_0", "mobile_1", current_time=2100.0)
        p2 = layer.get_delivery_predictability("mobile_0", "mobile_1")
        assert p2 > p1

        # Age directly without encounter (should decrease)
        layer.predictability._last_aging_time["mobile_0"] = 2100.0
        layer.predictability.age_predictabilities("mobile_0", current_time=10000.0)
        p3 = layer.get_delivery_predictability("mobile_0", "mobile_1")
        assert p3 < p2

    def test_buffer_overflow_scenario(self, full_comm_layer: CommunicationLayer):
        """Test buffer overflow with many messages."""
        layer = full_comm_layer
        buffer = layer.get_buffer("mobile_0")

        msg_size = layer.network_params.message_size_bytes
        buffer_size = layer.network_params.buffer_size_bytes
        max_messages = buffer_size // msg_size

        for i in range(max_messages + 5):
            layer.create_message(
                source_id="mobile_0",
                destination_id="coord_0",
                message_type=MessageType.STATUS,
                payload=f"test_{i}",
                current_time=float(i),
            )

        assert buffer.dropped_count >= 5
        assert buffer.message_count <= max_messages

    def test_prophetv2_anti_saturation_in_layer(
        self, full_comm_layer: CommunicationLayer
    ):
        """Integration test: verify PRoPHETv2 prevents saturation across layer.

        Simulate many rapid encounters and verify P-values remain varied
        rather than all converging to ~1.0.
        """
        layer = full_comm_layer

        # Simulate 100 encounters between mobile_0 and mobile_1 every 10s
        for i in range(100):
            layer.process_encounter(
                "mobile_0", "mobile_1", current_time=float(i * 10)
            )

        p = layer.get_delivery_predictability("mobile_0", "mobile_1")

        # PRoPHETv2 should prevent saturation
        assert p < 0.9, f"P-value {p:.4f} saturated despite PRoPHETv2"
        assert p > 0.0, "P-value should not be zero after encounters"
