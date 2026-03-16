#!/usr/bin/env python3
"""
Diagnostic: measure encounter frequency and message delivery bottlenecks.

Answers the key question: can messages actually leave the coordination
nodes and reach mobile responders?
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ercs.config.parameters import SimulationConfig
from ercs.coordination.algorithms import AlgorithmType
from ercs.simulation.engine import (
    SimulationEngine,
)


class DiagnosticEngine(SimulationEngine):
    """SimulationEngine with encounter tracking instrumentation."""

    def _initialize_components(self):
        super()._initialize_components()

        # Identify node types (available after super init)
        self._coord_ids = set(self._topology.get_coordination_node_ids())
        self._mobile_ids = set(self._topology.get_mobile_responder_ids())

        # Tracking state
        self._encounter_counts = Counter()
        self._encounter_times = defaultdict(list)
        self._min_distances = {
            cid: (float("inf"), 0.0) for cid in self._coord_ids
        }  # (dist, time)

        # Wrap process_encounter to count by type
        self._original_process = self._communication.process_encounter

        def tracked_process(node_a, node_b, current_time):
            a_type = "coord" if node_a in self._coord_ids else "mobile"
            b_type = "coord" if node_b in self._coord_ids else "mobile"
            key = "-".join(sorted([a_type, b_type]))
            self._encounter_counts[key] += 1
            if "coord" in key:
                self._encounter_times[key].append(current_time)
            return self._original_process(node_a, node_b, current_time)

        self._communication.process_encounter = tracked_process

    def _handle_mobility_update(self, event, results):
        super()._handle_mobility_update(event, results)
        # Track closest mobile approach to each coord node
        for cid in self._coord_ids:
            cpos = self._topology.get_node_position(cid)
            for mid in self._mobile_ids:
                mpos = self._topology.get_node_position(mid)
                dx = cpos[0] - mpos[0]
                dy = cpos[1] - mpos[1]
                dist = (dx**2 + dy**2) ** 0.5
                if dist < self._min_distances[cid][0]:
                    self._min_distances[cid] = (dist, event.timestamp)


def run_diagnostic(duration=6000, connectivity=0.75, seed=42, algorithm="adaptive"):
    config = SimulationConfig()
    config = config.model_copy(
        update={
            "scenario": config.scenario.model_copy(
                update={"simulation_duration_seconds": duration}
            )
        }
    )

    algo = AlgorithmType.ADAPTIVE if algorithm == "adaptive" else AlgorithmType.BASELINE

    engine = DiagnosticEngine(
        config=config,
        algorithm_type=algo,
        connectivity_level=connectivity,
        random_seed=seed,
    )

    # Run simulation
    print(
        f"Running simulation ({duration}s, connectivity={connectivity}, seed={seed}, {algorithm})..."
    )
    results = engine.run()

    # -- Report --
    coord_ids = engine._coord_ids
    mobile_ids = engine._mobile_ids

    print(f"\nNodes: {len(coord_ids)} coord + {len(mobile_ids)} mobile")
    for cid in sorted(coord_ids):
        pos = engine._topology.get_node_position(cid)
        print(f"  {cid} at ({pos[0]:.0f}, {pos[1]:.0f})")

    print("\n" + "=" * 60)
    print("ENCOUNTER COUNTS")
    print("=" * 60)
    for key, count in sorted(engine._encounter_counts.items()):
        print(f"  {key:20s}: {count:6d}")

    total = sum(engine._encounter_counts.values())
    coord_mobile = engine._encounter_counts.get("coord-mobile", 0)
    if total:
        print(f"\n  Total:           {total}")
        print(f"  Coord-mobile:    {coord_mobile} ({coord_mobile/total*100:.1f}%)")

    if engine._encounter_times.get("coord-mobile"):
        times = engine._encounter_times["coord-mobile"]
        print(f"\n  First coord-mobile encounter: t={min(times):.0f}s")
        print(f"  Last:  t={max(times):.0f}s")
        print(f"  Count: {len(times)}")
    else:
        print("\n  >>> ZERO coord-mobile encounters! <<<")

    print("\n" + "=" * 60)
    print("CLOSEST MOBILE APPROACH TO COORD NODES")
    print("=" * 60)
    radio = config.network.radio_range_m
    for cid in sorted(coord_ids):
        d, t = engine._min_distances[cid]
        in_range = (
            "WITHIN RANGE" if d <= radio else f"{d/radio:.0f}x beyond radio range"
        )
        print(f"  {cid}: {d:.0f}m at t={t:.0f}s ({in_range})")

    print("\n" + "=" * 60)
    print("MESSAGE DELIVERY")
    print("=" * 60)
    print(f"  Tasks created:      {results.total_tasks}")
    print(f"  Tasks assigned:     {results.tasks_assigned}")
    print(f"  Messages created:   {results.messages_created}")
    print(f"  Messages delivered: {results.messages_delivered}")
    print(f"  Messages expired:   {results.messages_expired}")
    print(f"  Delivery rate:      {results.delivery_rate:.1%}")
    print(f"  Assignment rate:    {results.assignment_rate:.1%}")

    print("\n" + "=" * 60)
    print("COORD NODE BUFFERS AT END")
    print("=" * 60)
    for cid in sorted(coord_ids):
        buf = engine._communication.buffers[cid]
        print(f"  {cid}: {buf.message_count} messages ({buf.utilisation:.0%} full)")
        shown = 0
        for msg in buf:
            if shown < 3:
                print(f"    msg {msg.message_id[:12]}... → {msg.destination_id}")
                shown += 1
        if buf.message_count > 3:
            print(f"    ... and {buf.message_count - 3} more")

    print("\n" + "=" * 60)
    print("COORD→MOBILE PREDICTABILITY AT END")
    print("=" * 60)
    for cid in sorted(coord_ids):
        preds = engine._communication.predictability.get_all_predictabilities(cid)
        nonzero = [(nid, p) for nid, p in preds.items() if p > 0.001]
        nonzero.sort(key=lambda x: -x[1])
        print(f"  {cid}: {len(nonzero)} nodes with P > 0.001")
        for nid, p in nonzero[:5]:
            print(f"    → {nid}: P = {p:.6f}")
        if not nonzero:
            print("    ALL predictabilities decayed to ~0!")

    # Summary diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    if coord_mobile == 0:
        print("  PROBLEM: Zero encounters between coord and mobile nodes.")
        print("  Messages created in coord node buffers can NEVER be forwarded.")
        all_dists = [engine._min_distances[c][0] for c in coord_ids]
        closest = min(all_dists)
        print(f"  Closest any mobile node got to a coord node: {closest:.0f}m")
        print(f"  Radio range: {radio:.0f}m")
        print(f"  Gap: {closest - radio:.0f}m")
        if closest > radio:
            print(
                "\n  ROOT CAUSE: Mobile nodes never come within radio range of coord nodes."
            )
            print("  The 2200m gap between incident zone and coordination zone")
            print("  is too large for Random Waypoint to bridge reliably.")
    else:
        print(f"  Coord-mobile encounters: {coord_mobile}")
        if results.messages_delivered == 0:
            print("  But zero messages delivered — messages may be forwarded")
            print("  but never reach their final destination.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=6000)
    parser.add_argument("--connectivity", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--algorithm", choices=["adaptive", "baseline"], default="adaptive"
    )
    args = parser.parse_args()
    run_diagnostic(args.duration, args.connectivity, args.seed, args.algorithm)
