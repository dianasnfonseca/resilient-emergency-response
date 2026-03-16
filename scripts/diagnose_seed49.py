#!/usr/bin/env python3
"""
Diagnóstico: Why does seed 49 produce 0% delivery at 20% connectivity?

The seed validator confirmed ALL 200 seeds produce connected topologies —
transport nodes encounter coordination nodes within 300s.  So the 0% delivery
is NOT due to topology disconnection.  This script traces every message's
lifecycle to identify the actual cause.

Parts:
  1. Message Lifecycle Tracking (forwarding, rejection, buffer residence)
  2. Seed 49 vs good-seed comparison
  3. Expired message deep-dive (encounter overlap analysis)
  4. Link availability analysis for specific task assignments
"""

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np

from ercs.communication.prophet import (
    MessageType,
    TransmissionResult,
)
from ercs.config.parameters import (
    AlgorithmType,
    ResponderRole,
    SimulationConfig,
)
from ercs.network.mobility import _assign_roles
from ercs.simulation.engine import (
    SimulationEngine,
    SimulationEventType,
)

# ═══════════════════════════════════════════════════════════════════════
# Data structures for tracing
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ForwardEvent:
    time: float
    from_node: str
    to_node: str
    p_from_to_dest: float
    p_to_to_dest: float
    transfer_type: str  # "forward" or "direct"


@dataclass
class RejectedForward:
    time: float
    from_node: str
    to_node: str
    dest: str
    p_from: float
    p_to: float


@dataclass
class BufferEntry:
    node: str
    enter_time: float
    exit_time: float | None = None
    exit_reason: str = "still_held"


@dataclass
class MessageTrace:
    message_id: str
    task_id: str
    source: str
    destination: str
    destination_role: str
    creation_time: float
    ttl_seconds: float
    final_status: str = "in_buffer"
    delivery_time: float | None = None

    forward_events: list[ForwardEvent] = field(default_factory=list)
    rejected_forwards: list[RejectedForward] = field(default_factory=list)
    buffer_residence: list[BufferEntry] = field(default_factory=list)
    holders: set[str] = field(default_factory=set)


@dataclass
class EncounterRecord:
    """Records each encounter between two nodes (link-available edges)."""

    time: float
    node_a: str
    node_b: str
    encounter_type: str  # "new" or "existing"


# ═══════════════════════════════════════════════════════════════════════
# Instrumented engine that traces message lifecycle
# ═══════════════════════════════════════════════════════════════════════


class TracingEngine(SimulationEngine):
    """SimulationEngine subclass that instruments message forwarding."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.message_traces: dict[str, MessageTrace] = {}
        self.encounter_log: list[EncounterRecord] = []
        self._role_map: dict[str, str] = {}

    def _initialize_components(self):
        super()._initialize_components()

        # Build role map
        mobile_ids = self._topology.get_mobile_responder_ids()
        roles = _assign_roles(len(mobile_ids))
        self._role_map = {
            node_id: roles[idx].value for idx, node_id in enumerate(mobile_ids)
        }

    # ── Override _handle_coordination_cycle to trace message creation ──

    def _handle_coordination_cycle(self, event, results):
        """Override to capture message creation with destination role."""
        coord_nodes = self._topology.get_coordination_node_ids()
        coord_node = coord_nodes[0] if coord_nodes else "coord_0"

        assignments = self._manager.run_coordination_cycle(
            responder_locator=self._adapter,
            network_state=self._adapter,
            coordination_node=coord_node,
            current_time=event.timestamp,
        )

        for i, assignment in enumerate(assignments):
            results.tasks_assigned += 1
            response_time = event.timestamp - assignment.task.creation_time
            results.response_times.append((assignment.task_id, response_time))

            source_node = coord_nodes[i % len(coord_nodes)]
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

            self._task_message_map[assignment.task_id] = message.message_id
            self._message_creation_times[message.message_id] = event.timestamp
            results.messages_created += 1

            # ── TRACE: register message ──
            dest_role = self._role_map.get(assignment.responder_id, "unknown")
            trace = MessageTrace(
                message_id=message.message_id,
                task_id=assignment.task_id,
                source=source_node,
                destination=assignment.responder_id,
                destination_role=dest_role,
                creation_time=event.timestamp,
                ttl_seconds=message.ttl_seconds,
            )
            trace.holders.add(source_node)
            trace.buffer_residence.append(
                BufferEntry(node=source_node, enter_time=event.timestamp)
            )
            self.message_traces[message.message_id] = trace

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
                {"message_id": message.message_id, "task_id": assignment.task_id},
            )

    # ── Override encounter handlers to trace forwarding ──

    def _handle_mobility_update(self, event, results):
        """Override to trace forwarding during mobility encounters."""
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
                self._trace_transfers(delivered, node_a, node_b, event.timestamp, "new")
                self.encounter_log.append(
                    EncounterRecord(
                        event.timestamp,
                        node_a,
                        node_b,
                        "new",
                    )
                )
            else:
                delivered = self._communication.transfer_messages(
                    node_a=node_a,
                    node_b=node_b,
                    current_time=event.timestamp,
                )
                self._trace_transfers(
                    delivered, node_a, node_b, event.timestamp, "existing"
                )

            self._process_delivered_messages(delivered, event.timestamp, results)

    def _handle_node_encounters(self, event, results):
        """Override to trace forwarding during periodic encounters."""
        current_links: set[tuple[str, str]] = set()
        for node_a, node_b in self._topology.graph.edges():
            if not self._is_link_available(node_a, node_b):
                continue
            link_key = (min(node_a, node_b), max(node_a, node_b))
            current_links.add(link_key)

        new_links = current_links - self._active_links
        for node_a, node_b in new_links:
            delivered = self._communication.process_encounter(
                node_a=node_a,
                node_b=node_b,
                current_time=event.timestamp,
            )
            self._trace_transfers(delivered, node_a, node_b, event.timestamp, "new")
            self.encounter_log.append(
                EncounterRecord(
                    event.timestamp,
                    node_a,
                    node_b,
                    "new",
                )
            )
            self._process_delivered_messages(delivered, event.timestamp, results)

        existing_links = current_links & self._active_links
        for node_a, node_b in existing_links:
            delivered = self._communication.transfer_messages(
                node_a=node_a,
                node_b=node_b,
                current_time=event.timestamp,
            )
            self._trace_transfers(
                delivered, node_a, node_b, event.timestamp, "existing"
            )
            self._process_delivered_messages(delivered, event.timestamp, results)

        self._active_links = current_links

        expired = self._communication.expire_all_messages(event.timestamp)
        results.messages_expired += expired

    def _trace_transfers(
        self,
        results: list[TransmissionResult],
        encounter_a: str,
        encounter_b: str,
        time: float,
        encounter_type: str,
    ):
        """Record forwarding/delivery events for each message trace."""
        for r in results:
            msg_id = r.message.message_id
            trace = self.message_traces.get(msg_id)
            if trace is None:
                continue

            if r.reason == "delivered":
                trace.final_status = "delivered"
                trace.delivery_time = time
                trace.forward_events.append(
                    ForwardEvent(
                        time=time,
                        from_node=r.source_node,
                        to_node=r.target_node,
                        p_from_to_dest=0.0,
                        p_to_to_dest=1.0,
                        transfer_type="direct",
                    )
                )
                # Close buffer entry for the delivering node
                for entry in trace.buffer_residence:
                    if entry.node == r.source_node and entry.exit_time is None:
                        entry.exit_time = time
                        entry.exit_reason = "delivered_from_here"

            elif r.reason == "forwarded":
                dest = trace.destination
                p_from = self._communication.get_delivery_predictability(
                    r.source_node, dest
                )
                p_to = self._communication.get_delivery_predictability(
                    r.target_node, dest
                )
                trace.forward_events.append(
                    ForwardEvent(
                        time=time,
                        from_node=r.source_node,
                        to_node=r.target_node,
                        p_from_to_dest=p_from,
                        p_to_to_dest=p_to,
                        transfer_type="forward",
                    )
                )
                trace.holders.add(r.target_node)
                trace.buffer_residence.append(
                    BufferEntry(node=r.target_node, enter_time=time)
                )

    # ── Collect rejected forwards by scanning encounters ──
    # This is expensive, so we do it post-hoc for specific messages
    def collect_rejections_for_message(self, msg_trace: MessageTrace):
        """Post-hoc: for each encounter during message lifetime, check if
        forwarding was rejected (p_to <= p_from)."""
        # We can't replay encounters, but we can infer from the encounter log
        # which nodes met during the message's lifetime
        pass  # Handled in analysis section


# ═══════════════════════════════════════════════════════════════════════
# Helper: is_link_available (standalone, matches engine logic)
# ═══════════════════════════════════════════════════════════════════════


def is_link_available(node_a: str, node_b: str, connectivity: float, seed: int) -> bool:
    pair_key = tuple(sorted([node_a, node_b]))
    pair_hash = hash((pair_key, seed)) % 10000
    threshold = int(connectivity * 10000)
    return pair_hash < threshold


# ═══════════════════════════════════════════════════════════════════════
# Run a traced simulation
# ═══════════════════════════════════════════════════════════════════════


def run_traced(
    seed: int, connectivity: float, algorithm: AlgorithmType
) -> TracingEngine:
    config = SimulationConfig()
    engine = TracingEngine(
        config=config,
        algorithm_type=algorithm,
        connectivity_level=connectivity,
        random_seed=seed,
    )
    engine.run(run_number=0)
    return engine


# ═══════════════════════════════════════════════════════════════════════
# Analysis functions
# ═══════════════════════════════════════════════════════════════════════


def analyze_seed(engine: TracingEngine, label: str):
    """Print detailed analysis of a traced simulation."""
    traces = engine.message_traces
    results_summary = {
        "total_messages": len(traces),
        "delivered": sum(1 for t in traces.values() if t.final_status == "delivered"),
        "expired": sum(1 for t in traces.values() if t.final_status != "delivered"),
    }

    delivered_count = results_summary["delivered"]
    total = results_summary["total_messages"]
    pct = 100 * delivered_count / total if total else 0

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Messages created:   {total}")
    print(f"  Delivered:          {delivered_count} ({pct:.1f}%)")
    print(f"  Not delivered:      {total - delivered_count}")

    # Forwarding stats
    fwd_counts = [len(t.forward_events) for t in traces.values()]
    zero_fwd = sum(1 for c in fwd_counts if c == 0)
    holder_counts = [len(t.holders) for t in traces.values()]

    print("\n  Forwarding statistics:")
    print(f"    Messages with 0 forwards (stuck in source buffer): {zero_fwd}/{total}")
    print(f"    Avg forwards per message: {np.mean(fwd_counts):.1f}")
    print(f"    Max forwards per message: {max(fwd_counts) if fwd_counts else 0}")
    print(f"    Avg unique holders per message: {np.mean(holder_counts):.1f}")
    print(f"    Max unique holders: {max(holder_counts) if holder_counts else 0}")

    # Role distribution of destinations
    role_counts = defaultdict(int)
    role_delivered = defaultdict(int)
    for t in traces.values():
        role_counts[t.destination_role] += 1
        if t.final_status == "delivered":
            role_delivered[t.destination_role] += 1

    print("\n  Assignment by destination role:")
    for role in ["rescue", "transport", "liaison"]:
        assigned = role_counts.get(role, 0)
        deliv = role_delivered.get(role, 0)
        pct_r = 100 * deliv / assigned if assigned else 0
        print(
            f"    {role:12s}: {assigned:3d} assigned, {deliv:3d} delivered ({pct_r:.0f}%)"
        )

    # Encounter analysis
    new_encounters = sum(1 for e in engine.encounter_log if e.encounter_type == "new")
    coord_ids = set(engine._topology.get_coordination_node_ids())
    coord_encounters = sum(
        1
        for e in engine.encounter_log
        if e.encounter_type == "new"
        and (e.node_a in coord_ids or e.node_b in coord_ids)
    )
    print("\n  Encounter statistics:")
    print(f"    Total new-link encounters: {new_encounters}")
    print(f"    Coord-involved encounters: {coord_encounters}")

    return results_summary


def analyze_expired_messages(engine: TracingEngine):
    """Deep analysis of messages that were never delivered."""
    traces = engine.message_traces
    expired = [t for t in traces.values() if t.final_status != "delivered"]
    set(engine._topology.get_coordination_node_ids())

    if not expired:
        print("\n  No expired messages — all delivered!")
        return

    print(f"\n{'─' * 70}")
    print(f"  EXPIRED MESSAGE ANALYSIS ({len(expired)} messages)")
    print(f"{'─' * 70}")

    # Build encounter index: for each node, when did it encounter other nodes?
    # node -> list of (time, other_node)
    encounter_index: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for e in engine.encounter_log:
        if e.encounter_type == "new":
            encounter_index[e.node_a].append((e.time, e.node_b))
            encounter_index[e.node_b].append((e.time, e.node_a))

    # Summary counters
    could_have_delivered = 0
    never_reached_neighbor = 0
    no_encounter_with_dest = 0

    for msg in expired[:10]:  # Limit detailed output to 10 messages
        print(f"\n  === Message {msg.message_id[:12]}... ===")
        print(
            f"    Source: {msg.source} → Dest: {msg.destination} (role: {msg.destination_role})"
        )
        print(
            f"    Created: {msg.creation_time:.0f}s, Expires: {msg.creation_time + msg.ttl_seconds:.0f}s"
        )
        print(f"    Forward events: {len(msg.forward_events)}")
        print(f"    Unique holders: {len(msg.holders)}")

        if msg.forward_events:
            last_fwd = msg.forward_events[-1]
            print(
                f"    Last forward: t={last_fwd.time:.0f}s {last_fwd.from_node}→{last_fwd.to_node}"
            )
        else:
            print(f"    !! Never forwarded — message stayed in {msg.source} buffer")

        # Which nodes encountered the destination during message lifetime?
        msg_start = msg.creation_time
        msg_end = msg.creation_time + msg.ttl_seconds
        dest = msg.destination

        nodes_that_met_dest = set()
        for enc_time, other in encounter_index.get(dest, []):
            if msg_start <= enc_time <= msg_end:
                nodes_that_met_dest.add(other)

        # Also check: dest encountered these nodes
        # (encounter_index is bidirectional from our logging)

        overlap = nodes_that_met_dest & msg.holders
        print(
            f"    Nodes that met dest during msg lifetime: {len(nodes_that_met_dest)}"
        )
        print(f"    Of those, also held this message: {len(overlap)}")

        if len(overlap) > 0:
            could_have_delivered += 1
            print("    !!! Message COULD HAVE BEEN DELIVERED but wasn't")
            for node in list(overlap)[:5]:
                # When did this node hold the message?
                held_entries = [b for b in msg.buffer_residence if b.node == node]
                # When did this node meet the destination?
                met_times = [
                    t
                    for t, other in encounter_index.get(dest, [])
                    if other == node and msg_start <= t <= msg_end
                ]
                print(
                    f"      {node}: held msg from t={held_entries[0].enter_time:.0f}s, "
                    f"met dest at {[f'{t:.0f}' for t in met_times[:5]]}"
                )
                # Check: was it holding msg WHEN it met dest?
                for met_t in met_times:
                    was_holding = any(
                        b.enter_time <= met_t
                        and (b.exit_time is None or b.exit_time > met_t)
                        for b in held_entries
                    )
                    if was_holding:
                        # Why wasn't it delivered? Must be link_available or buffer issue
                        link_ok = is_link_available(
                            node, dest, engine.connectivity_level, engine.random_seed
                        )
                        print(
                            f"        At t={met_t:.0f}: was holding msg, link_available={link_ok}"
                        )
        elif len(nodes_that_met_dest) == 0:
            no_encounter_with_dest += 1
            print("    Destination node had NO encounters during msg lifetime!")
            # Check: does the dest node appear in ANY encounter?
            total_dest_enc = len(encounter_index.get(dest, []))
            print(
                f"    Total encounters for {dest} across simulation: {total_dest_enc}"
            )
        else:
            never_reached_neighbor += 1
            print("    Message never reached a node that encounters the destination")

    print(f"\n  Summary of {len(expired)} expired messages:")
    print(f"    Could have delivered (holder met dest): {could_have_delivered}")
    print(
        f"    Dest had encounters but msg didn't reach neighbor: {never_reached_neighbor}"
    )
    print(f"    Dest had NO encounters during msg lifetime: {no_encounter_with_dest}")
    remaining = (
        len(expired)
        - could_have_delivered
        - never_reached_neighbor
        - no_encounter_with_dest
    )
    if remaining > 0:
        print(
            f"    (remaining {remaining} not analyzed in detail — only first 10 shown)"
        )


def analyze_link_availability(engine: TracingEngine):
    """Analyze which links are available for the assigned tasks."""
    traces = engine.message_traces
    coord_ids = engine._topology.get_coordination_node_ids()
    mobile_ids = engine._topology.get_mobile_responder_ids()
    roles = _assign_roles(len(mobile_ids))
    role_map = {nid: roles[i] for i, nid in enumerate(mobile_ids)}

    transport_ids = [n for n, r in role_map.items() if r == ResponderRole.TRANSPORT]
    liaison_ids = [n for n, r in role_map.items() if r == ResponderRole.LIAISON]
    bridge_ids = transport_ids + liaison_ids

    seed = engine.random_seed
    conn = engine.connectivity_level

    print(f"\n{'─' * 70}")
    print(f"  LINK AVAILABILITY ANALYSIS (seed={seed}, conn={conn})")
    print(f"{'─' * 70}")

    # 1. Coord to bridge links
    coord_bridge_links = 0
    coord_bridge_total = 0
    for c in coord_ids:
        for b in bridge_ids:
            coord_bridge_total += 1
            if is_link_available(c, b, conn, seed):
                coord_bridge_links += 1

    print(
        f"\n  Coord → Bridge links: {coord_bridge_links}/{coord_bridge_total} available"
    )

    # 2. Per-destination analysis for messages that failed
    failed = [t for t in traces.values() if t.final_status != "delivered"]

    if not failed:
        print("  All messages delivered — no failed paths to analyze.")
        return

    no_path_count = 0
    for msg in failed[:20]:
        dest = msg.destination
        dest_role = msg.destination_role

        # Bridge nodes with available link to this destination
        bridges_to_dest = [
            b for b in bridge_ids if is_link_available(b, dest, conn, seed)
        ]
        # Of those, which have available link FROM coord?
        full_path = []
        for b in bridges_to_dest:
            for c in coord_ids:
                if is_link_available(c, b, conn, seed):
                    full_path.append((c, b))
                    break

        if len(full_path) == 0:
            no_path_count += 1

        if len(failed) <= 20:
            print(f"\n  Msg → {dest} ({dest_role}):")
            print(
                f"    Bridges with link to dest: {len(bridges_to_dest)}/{len(bridge_ids)}"
            )
            print(f"    Full path coord→bridge→dest: {len(full_path)}")
            if full_path:
                print(f"    Example paths: {full_path[:3]}")
            else:
                print("    !!! NO 2-hop path available")

    print(
        f"\n  Messages with NO 2-hop coord→bridge→dest path: {no_path_count}/{len(failed)}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  SEED 49 DELIVERY FAILURE DIAGNOSTIC")
    print("=" * 70)

    # ── Part 1 & 2: Run both seeds ──
    print("\nRunning seed 49 at 20% connectivity (adaptive)...")
    engine_bad = run_traced(
        seed=49, connectivity=0.20, algorithm=AlgorithmType.ADAPTIVE
    )

    # Find a good seed for comparison — try seed 42 first
    print("Running seed 42 at 20% connectivity (adaptive)...")
    engine_good = run_traced(
        seed=42, connectivity=0.20, algorithm=AlgorithmType.ADAPTIVE
    )

    # If seed 42 also has low delivery, try another
    good_delivery = sum(
        1 for t in engine_good.message_traces.values() if t.final_status == "delivered"
    )
    good_total = len(engine_good.message_traces)
    good_label = (
        f"SEED 42 (20% conn, adaptive) — {good_delivery}/{good_total} delivered"
    )

    bad_summary = analyze_seed(engine_bad, "SEED 49 (20% conn, adaptive)")
    good_summary = analyze_seed(engine_good, good_label)

    # ── Comparison table ──
    print(f"\n{'─' * 70}")
    print("  COMPARISON: SEED 49 vs SEED 42")
    print(f"{'─' * 70}")
    print(f"  {'':30s}  {'Seed 49':>10s}  {'Seed 42':>10s}")
    print(
        f"  {'Messages created':30s}  {bad_summary['total_messages']:>10d}  {good_summary['total_messages']:>10d}"
    )
    print(
        f"  {'Delivered':30s}  {bad_summary['delivered']:>10d}  {good_summary['delivered']:>10d}"
    )
    print(
        f"  {'Not delivered':30s}  {bad_summary['expired']:>10d}  {good_summary['expired']:>10d}"
    )

    # Forwarding comparison
    bad_fwd = [len(t.forward_events) for t in engine_bad.message_traces.values()]
    good_fwd = [len(t.forward_events) for t in engine_good.message_traces.values()]
    bad_zero = sum(1 for c in bad_fwd if c == 0)
    good_zero = sum(1 for c in good_fwd if c == 0)
    print(f"  {'Zero-forward messages':30s}  {bad_zero:>10d}  {good_zero:>10d}")
    print(
        f"  {'Avg forwards/msg':30s}  {np.mean(bad_fwd):>10.1f}  {np.mean(good_fwd):>10.1f}"
    )

    bad_holders = [len(t.holders) for t in engine_bad.message_traces.values()]
    good_holders = [len(t.holders) for t in engine_good.message_traces.values()]
    print(
        f"  {'Avg holders/msg':30s}  {np.mean(bad_holders):>10.1f}  {np.mean(good_holders):>10.1f}"
    )

    bad_new_enc = sum(1 for e in engine_bad.encounter_log if e.encounter_type == "new")
    good_new_enc = sum(
        1 for e in engine_good.encounter_log if e.encounter_type == "new"
    )
    print(f"  {'New-link encounters':30s}  {bad_new_enc:>10d}  {good_new_enc:>10d}")

    bad_coord_ids = set(engine_bad._topology.get_coordination_node_ids())
    good_coord_ids = set(engine_good._topology.get_coordination_node_ids())
    bad_coord_enc = sum(
        1
        for e in engine_bad.encounter_log
        if e.encounter_type == "new"
        and (e.node_a in bad_coord_ids or e.node_b in bad_coord_ids)
    )
    good_coord_enc = sum(
        1
        for e in engine_good.encounter_log
        if e.encounter_type == "new"
        and (e.node_a in good_coord_ids or e.node_b in good_coord_ids)
    )
    print(f"  {'Coord encounters':30s}  {bad_coord_enc:>10d}  {good_coord_enc:>10d}")

    # ── Part 3: Expired message deep-dive for seed 49 ──
    analyze_expired_messages(engine_bad)

    # ── Part 4: Link availability analysis ──
    analyze_link_availability(engine_bad)
    analyze_link_availability(engine_good)

    # ── HYPOTHESIS ──
    traces_bad = engine_bad.message_traces
    total_bad = len(traces_bad)
    zero_fwd_bad = sum(1 for t in traces_bad.values() if len(t.forward_events) == 0)
    delivered_bad = sum(1 for t in traces_bad.values() if t.final_status == "delivered")

    print(f"\n{'=' * 70}")
    print("  HYPOTHESIS")
    print(f"{'=' * 70}")

    if zero_fwd_bad == total_bad:
        print("  A) Messages never leave coord buffer")
        print("     → No encounter with any node that has p_to > p_from to destination")
    elif zero_fwd_bad > total_bad * 0.8:
        print("  A) Most messages never leave coord buffer")
        print("     → Very few encounters with forward-eligible nodes")
    elif delivered_bad == 0 and np.mean(bad_fwd) > 0:
        # Messages forwarded but never delivered
        # Check if link to dest is the problem
        failed_msgs = [t for t in traces_bad.values() if t.final_status != "delivered"]
        mobile_ids = engine_bad._topology.get_mobile_responder_ids()
        roles = _assign_roles(len(mobile_ids))
        role_map = {nid: roles[i] for i, nid in enumerate(mobile_ids)}
        bridge_ids = [
            n
            for n, r in role_map.items()
            if r in (ResponderRole.TRANSPORT, ResponderRole.LIAISON)
        ]

        no_2hop = 0
        for msg in failed_msgs:
            dest = msg.destination
            bridges_to_dest = [
                b for b in bridge_ids if is_link_available(b, dest, 0.20, 49)
            ]
            has_full_path = any(
                is_link_available(c, b, 0.20, 49)
                for b in bridges_to_dest
                for c in engine_bad._topology.get_coordination_node_ids()
            )
            if not has_full_path:
                no_2hop += 1

        if no_2hop > total_bad * 0.5:
            print(
                "  C) Messages reach relay nodes but link to destination is unavailable"
            )
            print(
                f"     → {no_2hop}/{total_bad} messages have NO 2-hop path (coord→bridge→dest)"
            )
        else:
            print(
                "  B) Messages forwarded but never reach a node that encounters destination"
            )
            print("  E) PRoPHET P-values create forwarding dead-ends")
            print(
                f"     → Messages forwarded (avg {np.mean(bad_fwd):.1f}) but never delivered"
            )
    else:
        print("  F) Mixed causes — examine detailed output above")


if __name__ == "__main__":
    main()
