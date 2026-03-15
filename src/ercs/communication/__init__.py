"""
Communication Layer (Phase 2).

This module provides the PRoPHET-inspired DTN communication layer for
store-and-forward message delivery under intermittent connectivity.

Classes:
    Message: Individual message in the network
    MessageBuffer: Store-and-forward buffer for a node
    DeliveryPredictabilityMatrix: PRoPHET predictability tracking
    CommunicationLayer: Main communication layer coordinator

Factory Functions:
    create_message: Create a new message with unique ID

Enums:
    MessageType: Classification of messages
    MessageStatus: Status of message delivery

"""

from ercs.communication.prophet import (
    CommunicationLayer,
    DeliveryPredictabilityMatrix,
    Message,
    MessageBuffer,
    MessageStatus,
    MessageType,
    TransmissionResult,
    create_message,
)

__all__ = [
    "CommunicationLayer",
    "DeliveryPredictabilityMatrix",
    "Message",
    "MessageBuffer",
    "MessageStatus",
    "MessageType",
    "TransmissionResult",
    "create_message",
]
