#!/usr/bin/env python3
"""
Diagnostic: Role-Based Mobility + PRoPHET Predictability Saturation.

Runs a short simulation and produces detailed diagnostics to identify
why Adaptive and Baseline produce identical results.

Usage:
    python scripts/diagnose_mobility_prophet.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ercs.communication.prophet import DeliveryPredictabilityMatrix
from ercs.config.parameters import (
    NetworkParameters,
    ResponderRole,
    SimulationConfig,
    ZoneConfig,
)
from ercs.coordination.algorithms import AlgorithmType
from ercs.network.mobility import (
    MobilityManager,
    MobilityState,
    ROLE_CONFIGS,
    ROLE_DISTRIBUTION,
    _assign_roles,
)
from ercs.network.topology import generate_topology
from ercs.simulation.engine import SimulationEngine, SimulationEventType


# ============================================================================
# Helpers
# ============================================================================

def header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def subheader(title: str) -> None:
    print(f"\n  --- {title} ---")


# ============================================================================
# Part 1: Verify Roles and Mobility
# ============================================================================

def diagnose_roles_and_mobility(config: SimulationConfig, seed: int = 42) -> None:
    header("PART 1: ROLE ASSIGNMENT & MOBILITY")

    params = config.network

    # 1a. Role distribution
    subheader("1a. Role Distribution")
    n_mobile = params.mobile_responder_count
    roles = _assign_roles(n_mobile)
    role_counts = Counter(roles)
    print(f"  Total mobile nodes: {n_mobile}")
    for role in ResponderRole:
        expected_pct = ROLE_DISTRIBUTION[role] * 100
        actual = role_counts.get(role, 0)
        actual_pct = actual / n_mobile * 100
        print(f"    {role.value:12s}: {actual:3d} ({actual_pct:5.1f}%)  expected ~{expected_pct:.0f}%")

    # 1b. Verify waypoints per role
    subheader("1b. Waypoint Zone Verification (100 waypoints per node)")

    topology = generate_topology(
        parameters=params,
        connectivity_level=None,  # Full connectivity for position check
        random_seed=seed,
    )

    mobility = MobilityManager(parameters=params)
    mobile_ids = topology.get_mobile_responder_ids()
    initial_positions = {}
    for nid in mobile_ids:
        pos = topology.get_node_position(nid)
        if pos:
            initial_positions[nid] = pos

    mobility.initialize(mobile_ids, initial_positions, random_seed=seed)

    # Incident zone bounds
    iz = params.incident_zone
    iz_x_min, iz_x_max = iz.origin_x, iz.origin_x + iz.width_m
    iz_y_min, iz_y_max = iz.origin_y, iz.origin_y + iz.height_m

    # Coordination zone bounds
    cz = params.coordination_zone
    cz_x_min, cz_x_max = cz.origin_x, cz.origin_x + cz.width_m
    cz_y_min, cz_y_max = cz.origin_y, cz.origin_y + cz.height_m

    print(f"  Incident zone:     x=[{iz_x_min:.0f}, {iz_x_max:.0f}], y=[{iz_y_min:.0f}, {iz_y_max:.0f}]")
    print(f"  Coordination zone: x=[{cz_x_min:.0f}, {cz_x_max:.0f}], y=[{cz_y_min:.0f}, {cz_y_max:.0f}]")

    def in_zone(x, y, zone):
        return (zone.origin_x <= x <= zone.origin_x + zone.width_m and
                zone.origin_y <= y <= zone.origin_y + zone.height_m)

    # Force many waypoint assignments and check zones
    waypoint_checks = {role: {"in_incident": 0, "in_coord": 0, "total": 0}
                       for role in ResponderRole}

    for nid, state in mobility._node_states.items():
        role = state.role
        if role is None:
            continue
        for _ in range(100):
            mobility._assign_new_waypoint(state)
            wp = state.waypoint
            waypoint_checks[role]["total"] += 1
            if in_zone(wp.x, wp.y, iz):
                waypoint_checks[role]["in_incident"] += 1
            if in_zone(wp.x, wp.y, cz):
                waypoint_checks[role]["in_coord"] += 1

    for role in ResponderRole:
        checks = waypoint_checks[role]
        total = checks["total"]
        in_iz = checks["in_incident"]
        in_cz = checks["in_coord"]
        in_iz_pct = 100 * in_iz / total if total else 0
        in_cz_pct = 100 * in_cz / total if total else 0
        zone_mode = ROLE_CONFIGS[role].zone_mode
        print(f"\n  {role.value:12s} (zone_mode={zone_mode}):")
        print(f"    Waypoints in incident zone: {in_iz}/{total} ({in_iz_pct:.0f}%)")
        print(f"    Waypoints in coord zone:    {in_cz}/{total} ({in_cz_pct:.0f}%)")

    # 1c. Run mobility for 3000s and check positions
    subheader("1c. Node Positions After Mobility (t=3000s)")

    # Re-initialize
    mobility2 = MobilityManager(parameters=params)
    mobility2.initialize(mobile_ids, initial_positions, random_seed=seed)

    # Run 3000 steps of 1s each
    for t in range(3000):
        mobility2.step(current_time=float(t), delta_time=1.0)

    positions = mobility2.get_all_positions()
    in_iz_count = 0
    in_cz_count = 0
    role_in_iz = Counter()
    role_in_cz = Counter()

    for nid, (x, y) in positions.items():
        state = mobility2._node_states[nid]
        if in_zone(x, y, iz):
            in_iz_count += 1
            role_in_iz[state.role] += 1
        if in_zone(x, y, cz):
            in_cz_count += 1
            role_in_cz[state.role] += 1

    print(f"  Nodes in incident zone: {in_iz_count}/{n_mobile}")
    print(f"  Nodes in coord zone:    {in_cz_count}/{n_mobile}")
    for role in ResponderRole:
        n_role = role_counts[role]
        n_iz = role_in_iz.get(role, 0)
        n_cz = role_in_cz.get(role, 0)
        print(f"    {role.value:12s}: {n_iz}/{n_role} in incident, {n_cz}/{n_role} in coord")

    # 1d. Distance of each node to coord zone centre
    subheader("1d. Distance to Coordination Zone Centre at t=3000s")
    cz_cx = cz.origin_x + cz.width_m / 2
    cz_cy = cz.origin_y + cz.height_m / 2

    role_distances = {role: [] for role in ResponderRole}
    for nid, (x, y) in positions.items():
        state = mobility2._node_states[nid]
        dist = np.sqrt((x - cz_cx) ** 2 + (y - cz_cy) ** 2)
        role_distances[state.role].append(dist)

    for role in ResponderRole:
        dists = role_distances[role]
        if dists:
            print(f"  {role.value:12s}: mean={np.mean(dists):.0f}m, "
                  f"min={np.min(dists):.0f}m, max={np.max(dists):.0f}m, "
                  f"within 100m: {sum(1 for d in dists if d <= 100)}/{len(dists)}")


# ============================================================================
# Part 2: PRoPHET Aging Verification
# ============================================================================

def diagnose_prophet_aging(config: SimulationConfig) -> None:
    header("PART 2: PRoPHET AGING VERIFICATION")

    comm = config.communication

    # 2a. Check update_interval value
    subheader("2a. Update Interval Configuration")
    print(f"  config.communication.update_interval_seconds = {comm.update_interval_seconds}")
    print(f"  DeliveryPredictabilityMatrix default = 0.1")
    print(f"  CommunicationLayer passes: comm_params.update_interval_seconds = {comm.update_interval_seconds}")

    # 2b. Manual aging test
    subheader("2b. Manual Aging Test (P_enc_max=0.5, γ=0.999885791, interval=30s)")

    matrix = DeliveryPredictabilityMatrix(
        p_enc_max=comm.prophet.p_enc_max,
        i_typ=comm.prophet.i_typ,
        beta=comm.prophet.beta,
        gamma=comm.prophet.gamma,
        update_interval=comm.update_interval_seconds,
    )
    matrix.initialise_node("A", current_time=0.0)
    matrix.set_predictability("A", "B", 0.5)

    test_times = [10, 20, 30, 60, 300, 600, 1800]
    for t in test_times:
        # Reset for each test
        matrix2 = DeliveryPredictabilityMatrix(
            p_enc_max=comm.prophet.p_enc_max,
            i_typ=comm.prophet.i_typ,
            beta=comm.prophet.beta,
            gamma=comm.prophet.gamma,
            update_interval=comm.update_interval_seconds,
        )
        matrix2.initialise_node("A", current_time=0.0)
        matrix2.set_predictability("A", "B", 0.5)
        matrix2.age_predictabilities("A", float(t))
        p = matrix2.get_predictability("A", "B")
        k = t / comm.update_interval_seconds
        expected = 0.5 * comm.prophet.gamma ** k if k >= 1 else 0.5
        status = "OK" if abs(p - expected) < 0.001 else "MISMATCH!"
        print(f"  t={t:5d}s: k={k:6.1f}  P={p:.6f}  expected={expected:.6f}  {status}")

    # 2c. Aging vs encounter race condition (PRoPHETv2 time-based)
    subheader("2c. PRoPHETv2 Encounter Saturation Test (10s interval, 30s aging)")
    print("  Simulating: encounter every 10s, aging applied at each encounter")

    matrix3 = DeliveryPredictabilityMatrix(
        p_enc_max=comm.prophet.p_enc_max,
        i_typ=comm.prophet.i_typ,
        beta=comm.prophet.beta,
        gamma=comm.prophet.gamma,
        update_interval=comm.update_interval_seconds,
    )
    matrix3.initialise_node("A", current_time=0.0)

    p_history = []
    for t in range(0, 1801, 10):
        # This is what process_encounter does: age then encounter
        matrix3.age_predictabilities("A", float(t))
        matrix3.update_encounter("A", "B", float(t))
        p = matrix3.get_predictability("A", "B")
        p_history.append((t, p))

    # Print at key intervals
    sample_times = [0, 10, 20, 30, 60, 120, 300, 600, 1200, 1800]
    for t, p in p_history:
        if t in sample_times:
            print(f"    t={t:5d}s: P(A,B) = {p:.6f}")

    final_p = p_history[-1][1]
    print(f"\n  Final P after 1800s of encounters every 10s: {final_p:.6f}")
    high_p = [t for t, p in p_history if p > 0.9]
    if high_p:
        print(f"  P exceeded 0.9 at t={high_p[0]}s")
    else:
        print(f"  P never exceeded 0.9 — PRoPHETv2 anti-saturation working!")

    # 2d. What if encounters are less frequent?
    subheader("2d. Encounter Frequency Sensitivity (PRoPHETv2)")
    for interval in [10, 30, 60, 120, 300, 600]:
        m = DeliveryPredictabilityMatrix(
            p_enc_max=comm.prophet.p_enc_max,
            i_typ=comm.prophet.i_typ,
            beta=comm.prophet.beta,
            gamma=comm.prophet.gamma,
            update_interval=comm.update_interval_seconds,
        )
        m.initialise_node("A", current_time=0.0)
        for t in range(0, 1801, interval):
            m.age_predictabilities("A", float(t))
            m.update_encounter("A", "B", float(t))
        p = m.get_predictability("A", "B")
        n_encounters = 1800 // interval + 1
        print(f"  Encounter every {interval:4d}s ({n_encounters:3d} total): P = {p:.6f}")


# ============================================================================
# Part 3: Encounter Frequency in Actual Simulation
# ============================================================================

def diagnose_encounter_frequency(config: SimulationConfig, seed: int = 42) -> None:
    header("PART 3: ENCOUNTER FREQUENCY (ACTUAL SIMULATION)")

    class EncounterTracker(SimulationEngine):
        """Track every encounter that happens."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.encounter_log = []  # (time, node_a, node_b, source)
            self.edge_counts = []  # (time, n_edges)

        def _handle_mobility_update(self, event, results):
            """Track new-connection encounters from mobility."""
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
                self.encounter_log.append(
                    (event.timestamp, node_a, node_b, "mobility_new"))
                delivered = self._communication.process_encounter(
                    node_a=node_a, node_b=node_b, current_time=event.timestamp)
                self._process_delivered_messages(delivered, event.timestamp, results)

        def _handle_node_encounters(self, event, results):
            """Track edge-based encounters."""
            edges = list(self._topology.graph.edges())
            self.edge_counts.append((event.timestamp, len(edges)))

            for node_a, node_b in edges:
                if not self._is_link_available(node_a, node_b):
                    continue
                self.encounter_log.append(
                    (event.timestamp, node_a, node_b, "periodic"))
                delivered = self._communication.process_encounter(
                    node_a=node_a, node_b=node_b, current_time=event.timestamp)
                self._process_delivered_messages(delivered, event.timestamp, results)

            expired = self._communication.expire_all_messages(event.timestamp)
            results.messages_expired += expired

    # Run short simulation (warmup only, 600s) at 20% connectivity
    short_config = SimulationConfig(
        scenario=config.scenario.model_copy(update={
            "warmup_period_seconds": 600,
            "simulation_duration_seconds": 1,  # Almost no active sim
        }),
    )

    engine = EncounterTracker(
        config=short_config,
        algorithm_type=AlgorithmType.ADAPTIVE,
        connectivity_level=0.20,
        random_seed=seed,
    )
    engine.run()

    # Analyse encounters
    encounter_log = engine.encounter_log
    print(f"  Total encounters logged: {len(encounter_log)}")

    # Count by source
    by_source = Counter(src for _, _, _, src in encounter_log)
    print(f"  From mobility (new connections): {by_source.get('mobility_new', 0)}")
    print(f"  From periodic (existing edges):  {by_source.get('periodic', 0)}")

    # Count unique pairs
    pair_encounters = Counter()
    for _, a, b, _ in encounter_log:
        pair = tuple(sorted([a, b]))
        pair_encounters[pair] += 1

    print(f"  Unique pairs that encountered: {len(pair_encounters)}")

    # Get role info
    roles_map = {}
    for nid, state in engine._mobility._node_states.items():
        roles_map[nid] = state.role

    coord_nodes = set(engine._topology.get_coordination_node_ids())

    # Classify encounters
    def classify_pair(a, b):
        a_is_coord = a in coord_nodes
        b_is_coord = b in coord_nodes
        a_role = roles_map.get(a)
        b_role = roles_map.get(b)
        if a_is_coord or b_is_coord:
            mobile = b if a_is_coord else a
            m_role = roles_map.get(mobile)
            return f"coord-{m_role.value if m_role else 'unknown'}"
        if a_role and b_role:
            roles_sorted = tuple(sorted([a_role.value, b_role.value]))
            return f"{roles_sorted[0]}-{roles_sorted[1]}"
        return "other"

    category_encounters = defaultdict(list)
    for pair, count in pair_encounters.items():
        cat = classify_pair(pair[0], pair[1])
        category_encounters[cat].append(count)

    subheader("3a. Encounters by Category (600s, connectivity=0.20)")
    for cat in sorted(category_encounters.keys()):
        counts = category_encounters[cat]
        print(f"  {cat:25s}: {len(counts):4d} pairs, "
              f"mean={np.mean(counts):5.1f}, "
              f"total={sum(counts):6d} encounters")

    # Edge counts over time
    subheader("3b. Graph Edge Count Over Time")
    if engine.edge_counts:
        for t, n in engine.edge_counts[::6]:  # Every 60s
            print(f"  t={t:6.0f}s: {n:4d} edges")

    # 3c. How many link-available pairs exist?
    subheader("3c. Link Availability Filter")
    all_node_ids = engine._topology.get_all_node_ids()
    total_pairs = 0
    available_pairs = 0
    for i, a in enumerate(all_node_ids):
        for b in all_node_ids[i + 1:]:
            total_pairs += 1
            if engine._is_link_available(a, b):
                available_pairs += 1

    print(f"  Total possible node pairs: {total_pairs}")
    print(f"  Link-available pairs:      {available_pairs} "
          f"({100 * available_pairs / total_pairs:.1f}%)")
    print(f"  Expected at 20%:           {int(total_pairs * 0.20)}")


# ============================================================================
# Part 4: Predictability Matrix at Warm-up End
# ============================================================================

def diagnose_predictability(config: SimulationConfig, seed: int = 42) -> None:
    header("PART 4: PREDICTABILITY AT WARM-UP END")

    class PredTracker(SimulationEngine):
        """Capture P values at intervals during warm-up."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.p_snapshots = {}  # time -> {(coord, mobile): P}

        def _handle_node_encounters(self, event, results):
            super()._handle_node_encounters(event, results)
            t = event.timestamp
            # Snapshot every 300s
            if t % 300 < 10.1 and t not in self.p_snapshots:
                self._capture_p_snapshot(t)

        def _handle_warmup_end(self, event, results):
            super()._handle_warmup_end(event, results)
            self._capture_p_snapshot(event.timestamp)

        def _capture_p_snapshot(self, t):
            snapshot = {}
            coord_ids = self._topology.get_coordination_node_ids()
            mobile_ids = self._topology.get_mobile_responder_ids()
            for cid in coord_ids:
                for mid in mobile_ids:
                    p = self._communication.get_delivery_predictability(cid, mid)
                    snapshot[(cid, mid)] = p
            self.p_snapshots[t] = snapshot

    engine = PredTracker(
        config=config,
        algorithm_type=AlgorithmType.ADAPTIVE,
        connectivity_level=0.20,
        random_seed=seed,
    )
    engine.run()

    # Get role info
    roles_map = {}
    for nid, state in engine._mobility._node_states.items():
        roles_map[nid] = state.role

    subheader("4a. P-Value Evolution (coord_0 → all mobile nodes)")
    sorted_times = sorted(engine.p_snapshots.keys())
    for t in sorted_times:
        snapshot = engine.p_snapshots[t]
        # Filter to coord_0
        coord0_vals = [p for (c, m), p in snapshot.items() if c == "coord_0"]
        if not coord0_vals:
            continue
        arr = np.array(coord0_vals)
        nonzero = np.sum(arr > 0)
        print(f"  t={t:6.0f}s: mean={arr.mean():.4f}, std={arr.std():.4f}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}, "
              f"P>0: {nonzero}/{len(arr)}, P>0.9: {np.sum(arr > 0.9)}/{len(arr)}")

    # At warmup end, break down by role
    warmup_t = config.scenario.warmup_period_seconds
    if warmup_t in engine.p_snapshots:
        subheader(f"4b. P-Values by Role at t={warmup_t}s (warm-up end)")
        snapshot = engine.p_snapshots[warmup_t]

        role_p_values = {role: [] for role in ResponderRole}
        for (cid, mid), p in snapshot.items():
            role = roles_map.get(mid)
            if role:
                role_p_values[role].append(p)

        for role in ResponderRole:
            vals = role_p_values[role]
            if vals:
                arr = np.array(vals)
                print(f"  {role.value:12s}: n={len(vals):3d}, "
                      f"mean={arr.mean():.4f}, std={arr.std():.4f}, "
                      f"min={arr.min():.4f}, max={arr.max():.4f}, "
                      f"P>0.9: {np.sum(arr > 0.9)}/{len(vals)}")

    # 4c. Check what P values the matrix has for mobile-to-mobile pairs
    subheader("4c. Mobile-to-Mobile P Values (sample)")
    pred_matrix = engine._communication.predictability
    mobile_ids = engine._topology.get_mobile_responder_ids()
    sample_pairs = []
    for i in range(min(5, len(mobile_ids))):
        for j in range(i + 1, min(10, len(mobile_ids))):
            a, b = mobile_ids[i], mobile_ids[j]
            p = pred_matrix.get_predictability(a, b)
            ra = roles_map.get(a, "?")
            rb = roles_map.get(b, "?")
            sample_pairs.append((a, b, ra, rb, p))

    for a, b, ra, rb, p in sample_pairs[:15]:
        print(f"  P({a}, {b}) [{ra.value if hasattr(ra, 'value') else ra}"
              f"-{rb.value if hasattr(rb, 'value') else rb}] = {p:.6f}")


# ============================================================================
# Part 5: Coordination Algorithm Comparison
# ============================================================================

def diagnose_coordination(config: SimulationConfig, seed: int = 42) -> None:
    header("PART 5: COORDINATION ALGORITHM COMPARISON")

    class CoordDiag(SimulationEngine):
        """Capture coordination cycle details."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cycle_details = []

        def _handle_coordination_cycle(self, event, results):
            coord_nodes = self._topology.get_coordination_node_ids()
            coord_node = coord_nodes[0] if coord_nodes else "coord_0"
            mobile_ids = self._adapter.get_all_responder_ids()

            # Snapshot before assignment
            responder_p = {}
            for mid in mobile_ids:
                p = self._adapter.get_delivery_predictability(coord_node, mid)
                pos = self._adapter.get_responder_position(mid)
                responder_p[mid] = {"p": p, "pos": pos}

            pending_count = len(self._manager._pending_tasks)

            # Run actual cycle
            super()._handle_coordination_cycle(event, results)

            # Record
            p_vals = [v["p"] for v in responder_p.values()]
            p_arr = np.array(p_vals)
            nonzero_count = np.sum(p_arr > 0)
            above_09 = np.sum(p_arr > 0.9)

            self.cycle_details.append({
                "time": event.timestamp,
                "pending": pending_count,
                "assigned": results.tasks_assigned,
                "p_mean": p_arr.mean(),
                "p_std": p_arr.std(),
                "p_min": p_arr.min(),
                "p_max": p_arr.max(),
                "p_nonzero": int(nonzero_count),
                "p_above_09": int(above_09),
                "total_responders": len(mobile_ids),
            })

    # Run both algorithms
    for algo_name, algo_type in [("Adaptive", AlgorithmType.ADAPTIVE),
                                  ("Baseline", AlgorithmType.BASELINE)]:
        engine = CoordDiag(
            config=config,
            algorithm_type=algo_type,
            connectivity_level=0.20,
            random_seed=seed,
        )
        result = engine.run()

        subheader(f"5a. {algo_name} Coordination Cycles (connectivity=0.20)")
        for cycle in engine.cycle_details:
            print(f"  t={cycle['time']:6.0f}s: pending={cycle['pending']:3d}, "
                  f"assigned={cycle['assigned']:3d}, "
                  f"P: mean={cycle['p_mean']:.4f} std={cycle['p_std']:.4f} "
                  f"[{cycle['p_min']:.4f}, {cycle['p_max']:.4f}] "
                  f"P>0={cycle['p_nonzero']}/{cycle['total_responders']} "
                  f"P>0.9={cycle['p_above_09']}/{cycle['total_responders']}")

        print(f"\n  {algo_name} results:")
        print(f"    Tasks total:    {result.total_tasks}")
        print(f"    Tasks assigned: {result.tasks_assigned} ({result.assignment_rate:.1%})")
        print(f"    Msgs created:   {result.messages_created}")
        print(f"    Msgs delivered: {result.messages_delivered} ({result.delivery_rate:.1%})")
        avg_rt = result.average_response_time
        print(f"    Avg resp time:  {avg_rt:.1f}s" if avg_rt else "    Avg resp time:  N/A")


# ============================================================================
# Part 6: Connectivity Inversion Check
# ============================================================================

def diagnose_connectivity_inversion(config: SimulationConfig, seed: int = 42) -> None:
    header("PART 6: CONNECTIVITY LEVEL INVERSION CHECK")

    for conn in [0.75, 0.40, 0.20]:
        engine = SimulationEngine(
            config=config,
            algorithm_type=AlgorithmType.ADAPTIVE,
            connectivity_level=conn,
            random_seed=seed,
        )
        result = engine.run()

        print(f"\n  Connectivity={conn:.2f}:")
        print(f"    Tasks: {result.total_tasks}, "
              f"Assigned: {result.tasks_assigned} ({result.assignment_rate:.1%})")
        print(f"    Messages: {result.messages_created}, "
              f"Delivered: {result.messages_delivered} ({result.delivery_rate:.1%})")
        avg_dt = result.average_delivery_time
        print(f"    Avg delivery time: {avg_dt:.1f}s" if avg_dt else
              "    Avg delivery time: N/A")


# ============================================================================
# Part 7: Root Cause Summary
# ============================================================================

def print_root_cause_summary() -> None:
    header("SUSPECTED ROOT CAUSES")
    print("""
  Based on the diagnostic output above, check for these issues:

  1. P-VALUE SATURATION
     If P > 0.9 for ALL coord→mobile pairs at warm-up end:
     → Encounters are too frequent relative to aging rate
     → With encounter every 10s and aging every 30s, P saturates in ~30s
     → Role-based mobility cannot help if transitivity propagates P to all nodes

  2. TRANSITIVITY AMPLIFICATION
     If rescue nodes (never near coord) still have high P:
     → Transport/liaison nodes act as "P bridges"
     → P(coord, transport) ≈ 1.0 + P(transport, rescue) ≈ 1.0
     → Transitivity: P(coord, rescue) += (1-P) * P(c,t) * P(t,r) * β
     → Even with β=0.25, this converges to ~1.0 over 1800s

  3. ENCOUNTER INTERVAL TOO SHORT
     If periodic encounters (every 10s on ALL edges) dominate:
     → Every pair within 100m exchanges P updates 6x per minute
     → P_init=0.75 means 3 encounters → P > 0.98
     → Aging γ=0.98 per 30s is too weak to counteract

  4. LINK AVAILABILITY NOT DIFFERENTIATING
     If the hash-based filter gives ~20% of ALL pairs:
     → Some rescue-coord pairs may still be link-available
     → Through transitivity, even non-available pairs get high P

  5. DELIVERY RATE INVERSION (higher at lower connectivity)
     Possible causes:
     → Fewer encounters → fewer messages forwarded → less congestion
     → Messages to unreachable nodes never delivered → not counted?
     → Check if messages_created differs across connectivity levels
""")


# ============================================================================
# Main
# ============================================================================

def main():
    config = SimulationConfig()
    seed = 42

    print(f"ERCS Mobility + PRoPHET Diagnostic")
    print(f"Config: warmup={config.scenario.warmup_period_seconds}s, "
          f"duration={config.scenario.simulation_duration_seconds}s, "
          f"total={config.total_simulation_duration}s")
    print(f"PRoPHETv2: P_enc_max={config.communication.prophet.p_enc_max}, "
          f"I_typ={config.communication.prophet.i_typ}, "
          f"β={config.communication.prophet.beta}, "
          f"γ={config.communication.prophet.gamma}, "
          f"update_interval={config.communication.update_interval_seconds}s")
    print(f"Network: {config.network.mobile_responder_count} mobile nodes, "
          f"radio_range={config.network.radio_range_m}m")
    print(f"Mobility: {config.network.mobility_model.value}")

    diagnose_roles_and_mobility(config, seed)
    diagnose_prophet_aging(config)
    diagnose_encounter_frequency(config, seed)
    diagnose_predictability(config, seed)
    diagnose_coordination(config, seed)
    diagnose_connectivity_inversion(config, seed)
    print_root_cause_summary()


if __name__ == "__main__":
    main()
