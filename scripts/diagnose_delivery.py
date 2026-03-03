#!/usr/bin/env python3
"""
Diagnostic: 100% Delivery Rate Investigation.

After fixing PRoPHET encounter updates to connection-up only,
Adaptive achieves 100% delivery at 20% connectivity. This script
investigates whether that is genuine or caused by a bug.

Usage:
    python scripts/diagnose_delivery.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ercs.communication.prophet import CommunicationLayer, TransmissionResult
from ercs.config.parameters import AlgorithmType, SimulationConfig
from ercs.simulation.engine import SimulationEngine, SimulationEventType


def header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def subheader(title: str) -> None:
    print(f"\n  --- {title} ---")


# ============================================================================
# Instrumented Engine
# ============================================================================

class DeliveryTracer(SimulationEngine):
    """Engine that traces every encounter and transfer call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track per-call data
        self.encounter_calls = []  # (time, source, node_a, node_b, n_results)
        self.transfer_calls = []   # (time, source, node_a, node_b, n_results)
        # Track forwarding events
        self.forward_events = []   # (time, msg_id, from, to, reason)
        # Track message copies in buffers
        self.buffer_snapshots = []  # (time, {msg_id: count_in_buffers})

    def _handle_mobility_update(self, event, results):
        """Trace mobility encounter/transfer calls."""
        delta_time = event.data.get("delta_time", 1.0)
        moved_nodes = self._mobility.step(
            current_time=event.timestamp,
            delta_time=delta_time,
        )
        if not moved_nodes:
            return

        all_positions = self._mobility.get_all_positions()
        for node_id, (x, y) in all_positions.items():
            self._topology.update_node_position(node_id, x, y)

        new_connections = self._topology.update_edges_from_positions()

        for node_a, node_b in new_connections:
            if not self._is_link_available(node_a, node_b):
                continue

            link_key = (min(node_a, node_b), max(node_a, node_b))

            if link_key not in self._active_links:
                self._active_links.add(link_key)
                delivered = self._communication.process_encounter(
                    node_a=node_a,
                    node_b=node_b,
                    current_time=event.timestamp,
                )
                self.encounter_calls.append(
                    (event.timestamp, "mobility", node_a, node_b, len(delivered)))
                self._trace_results(delivered, event.timestamp)
            else:
                delivered = self._communication.transfer_messages(
                    node_a=node_a,
                    node_b=node_b,
                    current_time=event.timestamp,
                )
                self.transfer_calls.append(
                    (event.timestamp, "mobility", node_a, node_b, len(delivered)))
                self._trace_results(delivered, event.timestamp)

            self._process_delivered_messages(delivered, event.timestamp, results)

    def _handle_node_encounters(self, event, results):
        """Trace node encounter/transfer calls."""
        current_links: set[tuple[str, str]] = set()
        for node_a, node_b in self._topology.graph.edges():
            if not self._is_link_available(node_a, node_b):
                continue
            link_key = (min(node_a, node_b), max(node_a, node_b))
            current_links.add(link_key)

        new_links = current_links - self._active_links
        existing_links = current_links & self._active_links

        for node_a, node_b in new_links:
            delivered = self._communication.process_encounter(
                node_a=node_a,
                node_b=node_b,
                current_time=event.timestamp,
            )
            self.encounter_calls.append(
                (event.timestamp, "periodic_new", node_a, node_b, len(delivered)))
            self._trace_results(delivered, event.timestamp)
            self._process_delivered_messages(delivered, event.timestamp, results)

        for node_a, node_b in existing_links:
            delivered = self._communication.transfer_messages(
                node_a=node_a,
                node_b=node_b,
                current_time=event.timestamp,
            )
            self.transfer_calls.append(
                (event.timestamp, "periodic_existing", node_a, node_b, len(delivered)))
            self._trace_results(delivered, event.timestamp)
            self._process_delivered_messages(delivered, event.timestamp, results)

        self._active_links = current_links

        expired = self._communication.expire_all_messages(event.timestamp)
        results.messages_expired += expired

        # Snapshot buffer state every 300s
        if event.timestamp % 300 < 10.1:
            self._snapshot_buffers(event.timestamp)

    def _trace_results(self, results, timestamp):
        """Record forwarding events."""
        for r in results:
            self.forward_events.append((
                timestamp,
                r.message.message_id,
                r.source_node,
                r.target_node,
                r.reason,
            ))

    def _snapshot_buffers(self, timestamp):
        """Count copies of each message across all buffers."""
        msg_counts = Counter()
        for buffer in self._communication.buffers.values():
            for msg in buffer:
                msg_counts[msg.message_id] += 1
        self.buffer_snapshots.append((timestamp, dict(msg_counts)))


# ============================================================================
# Diagnostics
# ============================================================================

def diagnose_encounter_vs_transfer(engine: DeliveryTracer) -> None:
    header("PART 1: ENCOUNTER vs TRANSFER CALL ANALYSIS")

    enc_by_source = Counter(src for _, src, _, _, _ in engine.encounter_calls)
    tr_by_source = Counter(src for _, src, _, _, _ in engine.transfer_calls)

    subheader("1a. Call Counts by Source")
    print(f"  process_encounter() calls:")
    for src, count in sorted(enc_by_source.items()):
        print(f"    {src:25s}: {count:6d}")
    print(f"    {'TOTAL':25s}: {len(engine.encounter_calls):6d}")
    print(f"\n  transfer_messages() calls:")
    for src, count in sorted(tr_by_source.items()):
        print(f"    {src:25s}: {count:6d}")
    print(f"    {'TOTAL':25s}: {len(engine.transfer_calls):6d}")

    # Check for same pair getting both encounter and transfer at same timestamp
    subheader("1b. Double Processing Check (same pair, same timestamp)")
    enc_keys = set()
    for t, src, a, b, _ in engine.encounter_calls:
        pair = (t, min(a, b), max(a, b))
        enc_keys.add(pair)

    double_count = 0
    for t, src, a, b, _ in engine.transfer_calls:
        pair = (t, min(a, b), max(a, b))
        if pair in enc_keys:
            double_count += 1

    print(f"  Same (time, pair) in BOTH encounter and transfer: {double_count}")
    if double_count > 0:
        print(f"  *** DOUBLE PROCESSING DETECTED ***")
    else:
        print(f"  No double processing — sets are disjoint. OK")

    # Check encounter updates per unique contact episode
    subheader("1c. Encounter Updates per Unique Node Pair")
    pair_encounter_counts = Counter()
    for _, _, a, b, _ in engine.encounter_calls:
        pair = (min(a, b), max(a, b))
        pair_encounter_counts[pair] += 1

    counts = list(pair_encounter_counts.values())
    if counts:
        print(f"  Unique pairs with encounter updates: {len(counts)}")
        print(f"  Encounter updates per pair: mean={np.mean(counts):.1f}, "
              f"median={np.median(counts):.0f}, max={max(counts)}")
        # Show top pairs
        top = pair_encounter_counts.most_common(5)
        for (a, b), c in top:
            print(f"    {a}-{b}: {c} encounter updates")


def diagnose_forwarding(engine: DeliveryTracer) -> None:
    header("PART 2: FORWARDING ANALYSIS")

    by_reason = Counter(reason for _, _, _, _, reason in engine.forward_events)
    print(f"  Total forwarding/delivery events: {len(engine.forward_events)}")
    for reason, count in sorted(by_reason.items()):
        print(f"    {reason:15s}: {count:6d}")

    # Copies per message
    subheader("2a. Message Copies (from forwarding events)")
    msg_copies = Counter()
    for _, msg_id, _, _, reason in engine.forward_events:
        if reason == "forwarded":
            msg_copies[msg_id] += 1

    copy_counts = list(msg_copies.values()) if msg_copies else [0]
    print(f"  Messages that were forwarded at least once: {len(msg_copies)}")
    print(f"  Forwards per message: mean={np.mean(copy_counts):.1f}, "
          f"max={max(copy_counts)}")

    # Copies in buffers from snapshots
    subheader("2b. Message Copies in Buffers Over Time")
    for t, counts in engine.buffer_snapshots:
        if counts:
            vals = list(counts.values())
            print(f"  t={t:6.0f}s: {len(counts):4d} unique msgs, "
                  f"copies/msg: mean={np.mean(vals):.1f}, max={max(vals)}")
        else:
            print(f"  t={t:6.0f}s: 0 messages in buffers")

    # Hops to delivery
    subheader("2c. Hops to Delivery")
    delivery_hops = []
    for _, msg_id, _, _, reason in engine.forward_events:
        if reason == "delivered":
            # Count forwards for this message
            n_forwards = sum(1 for _, mid, _, _, r in engine.forward_events
                            if mid == msg_id and r == "forwarded")
            delivery_hops.append(n_forwards)

    if delivery_hops:
        print(f"  Delivered messages: {len(delivery_hops)}")
        print(f"  Hops: mean={np.mean(delivery_hops):.1f}, "
              f"median={np.median(delivery_hops):.0f}, "
              f"max={max(delivery_hops)}")
        hop_dist = Counter(delivery_hops)
        for hops in sorted(hop_dist.keys())[:10]:
            print(f"    {hops} hops: {hop_dist[hops]} messages")


def diagnose_algorithm_comparison(config: SimulationConfig, seed: int) -> None:
    header("PART 3: ALGORITHM COMPARISON")

    for algo_name, algo_type in [("Adaptive", AlgorithmType.ADAPTIVE),
                                  ("Baseline", AlgorithmType.BASELINE)]:
        engine = DeliveryTracer(
            config=config,
            algorithm_type=algo_type,
            connectivity_level=0.20,
            random_seed=seed,
        )
        result = engine.run()

        subheader(f"3. {algo_name} at 20% connectivity")
        print(f"  Tasks: {result.total_tasks}, "
              f"Assigned: {result.tasks_assigned} ({result.assignment_rate:.1%})")
        print(f"  Msgs created: {result.messages_created}")
        print(f"  Msgs delivered: {result.messages_delivered} "
              f"({result.delivery_rate:.1%})")
        print(f"  Msgs expired: {result.messages_expired}")

        avg_dt = result.average_delivery_time
        print(f"  Avg delivery time: {avg_dt:.1f}s" if avg_dt else
              "  Avg delivery time: N/A")

        by_reason = Counter(r for _, _, _, _, r in engine.forward_events)
        print(f"  Forward events: {by_reason.get('forwarded', 0)}")
        print(f"  Direct deliveries: {by_reason.get('delivered', 0)}")
        print(f"  Buffer full: {by_reason.get('buffer_full', 0)}")
        print(f"  process_encounter calls: {len(engine.encounter_calls)}")
        print(f"  transfer_messages calls: {len(engine.transfer_calls)}")

        # Check assignments
        assignments = engine._manager._all_assignments
        if assignments:
            p_vals = [a.predictability for a in assignments
                      if a.predictability is not None]
            if p_vals:
                print(f"  Assignment P values: mean={np.mean(p_vals):.4f}, "
                      f"min={np.min(p_vals):.4f}, max={np.max(p_vals):.4f}")

            # Check responder roles
            role_counts = Counter()
            for a in assignments:
                state = engine._mobility._node_states.get(a.responder_id)
                if state and state.role:
                    role_counts[state.role.value] += 1
            if role_counts:
                print(f"  Assigned to roles: {dict(role_counts)}")


def diagnose_delivery_paths(engine: DeliveryTracer, results) -> None:
    header("PART 4: DELIVERY PATH ANALYSIS")

    # Group forward events by message_id
    msg_events = defaultdict(list)
    for t, msg_id, src, tgt, reason in engine.forward_events:
        msg_events[msg_id].append((t, src, tgt, reason))

    # Find coordination messages (from task assignments)
    coord_msg_ids = set(engine._task_message_map.values())

    delivered_msgs = []
    undelivered_msgs = []

    for msg_id in coord_msg_ids:
        events = msg_events.get(msg_id, [])
        delivered = any(r == "delivered" for _, _, _, r in events)
        if delivered:
            delivered_msgs.append((msg_id, events))
        else:
            undelivered_msgs.append((msg_id, events))

    subheader("4a. Delivery Summary")
    print(f"  Coordination messages: {len(coord_msg_ids)}")
    print(f"  Delivered: {len(delivered_msgs)}")
    print(f"  Undelivered: {len(undelivered_msgs)}")

    # Show sample delivery paths
    subheader("4b. Sample Delivery Paths (first 5 delivered)")
    for msg_id, events in delivered_msgs[:5]:
        forwards = [(t, s, tgt, r) for t, s, tgt, r in events if r == "forwarded"]
        delivery = [(t, s, tgt, r) for t, s, tgt, r in events if r == "delivered"]
        print(f"\n  Message {msg_id[:12]}:")
        print(f"    Forwards: {len(forwards)}, Delivery: {len(delivery)}")
        for t, s, tgt, r in sorted(events, key=lambda x: x[0]):
            print(f"      t={t:7.1f}s: {s:12s} → {tgt:12s}  [{r}]")

    # Show undelivered messages if any
    if undelivered_msgs:
        subheader("4c. Undelivered Messages (first 5)")
        for msg_id, events in undelivered_msgs[:5]:
            forwards = [(t, s, tgt, r) for t, s, tgt, r in events
                        if r == "forwarded"]
            print(f"\n  Message {msg_id[:12]}:")
            print(f"    Forwards: {len(forwards)}")
            # Check where message ended up
            for buf in engine._communication.buffers.values():
                if buf.has_message(msg_id):
                    print(f"    Still in buffer: {buf.node_id}")


def diagnose_multiple_seeds(config: SimulationConfig) -> None:
    header("PART 5: DELIVERY RATE ACROSS SEEDS")

    subheader("5a. Adaptive at 20% connectivity, seeds 42-51")
    for seed in range(42, 52):
        engine = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=0.20,
            random_seed=seed,
        )
        result = engine.run()
        avg_dt = result.average_delivery_time
        dt_str = f"{avg_dt:.0f}s" if avg_dt else "N/A"
        print(f"  seed={seed}: delivery={result.delivery_rate:.1%} "
              f"({result.messages_delivered}/{result.messages_created}), "
              f"avg_dt={dt_str}")

    subheader("5b. Baseline at 20% connectivity, seeds 42-51")
    for seed in range(42, 52):
        engine = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.BASELINE,
            connectivity_level=0.20,
            random_seed=seed,
        )
        result = engine.run()
        avg_dt = result.average_delivery_time
        dt_str = f"{avg_dt:.0f}s" if avg_dt else "N/A"
        print(f"  seed={seed}: delivery={result.delivery_rate:.1%} "
              f"({result.messages_delivered}/{result.messages_created}), "
              f"avg_dt={dt_str}")


# ============================================================================
# Main
# ============================================================================

def main():
    config = SimulationConfig()
    seed = 42

    print("ERCS Delivery Rate Diagnostic")
    print(f"Config: warmup={config.scenario.warmup_period_seconds}s, "
          f"active={config.scenario.simulation_duration_seconds}s, "
          f"total={config.total_simulation_duration}s")
    print(f"Message TTL: {config.communication.message_ttl_seconds}s "
          f"({config.communication.message_ttl_seconds / 60:.0f} min)")

    # Run instrumented Adaptive simulation
    engine = DeliveryTracer(
        config=config,
        algorithm_type=AlgorithmType.ADAPTIVE,
        connectivity_level=0.20,
        random_seed=seed,
    )
    result = engine.run()

    print(f"\nAdaptive @ 20% connectivity: "
          f"{result.messages_delivered}/{result.messages_created} delivered "
          f"({result.delivery_rate:.1%})")

    diagnose_encounter_vs_transfer(engine)
    diagnose_forwarding(engine)
    diagnose_delivery_paths(engine, result)
    diagnose_algorithm_comparison(config, seed)
    diagnose_multiple_seeds(config)


if __name__ == "__main__":
    main()
