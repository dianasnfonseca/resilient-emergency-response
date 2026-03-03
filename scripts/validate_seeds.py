#!/usr/bin/env python3
"""
Seed Connectivity Validator for ERCS Experiments.

Screens candidate seeds to ensure that, at each connectivity level, at least
one TRANSPORT or LIAISON node can encounter a coordination node through an
available link during the warm-up period.

The hash-based link availability filter (``_is_link_available`` in engine.py)
can deterministically disable ALL links between coordination nodes and the
bridge nodes (TRANSPORT/LIAISON) for certain seed values.  This makes message
delivery impossible regardless of algorithm, injecting topology artefacts into
the experimental results.

This script:
  1. For each candidate seed (1..200) and each connectivity level (0.75, 0.40, 0.20):
     - Creates minimal simulation setup (topology + mobility + link filter)
     - Runs 300 seconds of mobility to let nodes move into range
     - Checks whether ANY transport or liaison node encounters a coord node
       through an available link
  2. Outputs ``config/valid_seeds.json`` with seeds that pass ALL levels

Usage:
    python scripts/validate_seeds.py

Output:
    config/valid_seeds.json
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np

from ercs.config.parameters import SimulationConfig
from ercs.network.mobility import MobilityManager, _assign_roles
from ercs.config.parameters import ResponderRole
from ercs.network.topology import generate_topology


def is_link_available(
    node_a: str, node_b: str, connectivity_level: float, random_seed: int
) -> bool:
    """Replicate the engine's deterministic link availability check."""
    pair_key = tuple(sorted([node_a, node_b]))
    pair_hash = hash((pair_key, random_seed)) % 10000
    threshold = int(connectivity_level * 10000)
    return pair_hash < threshold


def validate_seed(
    seed: int,
    connectivity_level: float,
    config: SimulationConfig,
    warmup_seconds: float = 300.0,
) -> bool:
    """
    Check whether a seed produces a connected topology at the given
    connectivity level.

    Returns True if at least one TRANSPORT or LIAISON node encounters
    a coordination node through an available link within ``warmup_seconds``.
    """
    topology = generate_topology(
        parameters=config.network,
        connectivity_level=connectivity_level,
        random_seed=seed,
    )

    coord_ids = set(topology.get_coordination_node_ids())
    mobile_ids = topology.get_mobile_responder_ids()

    # Initialize mobility
    mobility = MobilityManager(
        parameters=config.network,
        speed_min=1.0,
        speed_max=20.0,
        pause_min=0.0,
        pause_max=30.0,
    )

    initial_positions = {}
    for node_id in mobile_ids:
        pos = topology.get_node_position(node_id)
        if pos is not None:
            initial_positions[node_id] = pos

    mobility.initialize(
        mobile_node_ids=mobile_ids,
        initial_positions=initial_positions,
        random_seed=seed,
    )

    # Identify transport/liaison nodes via deterministic role assignment
    roles = _assign_roles(len(mobile_ids))
    bridge_nodes = set()
    for idx, node_id in enumerate(mobile_ids):
        if roles[idx] in (ResponderRole.TRANSPORT, ResponderRole.LIAISON):
            bridge_nodes.add(node_id)

    # If no bridge nodes, seed is invalid
    if not bridge_nodes:
        return False

    # Check at t=0 first (before any mobility steps)
    radio_range = config.network.radio_range_m

    def check_encounters() -> bool:
        """Check if any bridge node is within range of a coord node via available link."""
        for bridge_id in bridge_nodes:
            bridge_pos = topology.get_node_position(bridge_id)
            if bridge_pos is None:
                continue
            for coord_id in coord_ids:
                coord_pos = topology.get_node_position(coord_id)
                if coord_pos is None:
                    continue
                dx = bridge_pos[0] - coord_pos[0]
                dy = bridge_pos[1] - coord_pos[1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= radio_range:
                    if is_link_available(bridge_id, coord_id, connectivity_level, seed):
                        return True
        return False

    if check_encounters():
        return True

    # Run mobility for warmup_seconds in 1-second steps
    delta = 1.0
    t = delta
    while t <= warmup_seconds:
        moved = mobility.step(current_time=t, delta_time=delta)

        if moved:
            all_pos = mobility.get_all_positions()
            for nid, (x, y) in all_pos.items():
                topology.update_node_position(nid, x, y)
            topology.update_edges_from_positions()

        if check_encounters():
            return True

        t += delta

    return False


def main():
    config = SimulationConfig()
    connectivity_levels = config.network.connectivity_scenarios  # [0.75, 0.40, 0.20]
    candidate_range = range(1, 201)  # seeds 1..200

    print("=" * 60)
    print("ERCS Seed Connectivity Validator")
    print("=" * 60)
    print(f"Candidate seeds: {candidate_range.start}..{candidate_range.stop - 1}")
    print(f"Connectivity levels: {connectivity_levels}")
    print(f"Warmup duration: 300s")
    print()

    # Results: seed -> {level: pass/fail}
    valid_per_level: dict[float, list[int]] = {level: [] for level in connectivity_levels}
    valid_all_levels: list[int] = []

    start_time = time.time()

    for i, seed in enumerate(candidate_range):
        passes_all = True

        for level in connectivity_levels:
            ok = validate_seed(seed, level, config, warmup_seconds=300.0)
            if ok:
                valid_per_level[level].append(seed)
            else:
                passes_all = False

        if passes_all:
            valid_all_levels.append(seed)

        # Progress every 20 seeds
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            pct = 100 * (i + 1) / len(candidate_range)
            print(
                f"  [{pct:5.1f}%] Seed {seed:3d} — "
                f"{len(valid_all_levels)} valid so far "
                f"({elapsed:.0f}s elapsed)"
            )

    elapsed = time.time() - start_time
    print()
    print("-" * 60)
    print(f"Results ({elapsed:.1f}s):")
    for level in connectivity_levels:
        print(f"  Connectivity {level:.2f}: {len(valid_per_level[level])}/{len(candidate_range)} seeds pass")
    print(f"  ALL levels:       {len(valid_all_levels)}/{len(candidate_range)} seeds pass")
    print()

    # Build output
    output = {
        "description": (
            "Pre-screened seeds that guarantee at least one transport/liaison "
            "node can reach a coordination node at every connectivity level. "
            "Generated by scripts/validate_seeds.py."
        ),
        "connectivity_levels": connectivity_levels,
        "warmup_seconds": 300,
        "candidate_range": [candidate_range.start, candidate_range.stop - 1],
        "valid_seeds": valid_all_levels,
        "valid_per_level": {str(k): v for k, v in valid_per_level.items()},
    }

    # Write output
    output_dir = project_root / "config"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "valid_seeds.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Written to {output_path}")
    print(f"Valid seeds (first 30): {valid_all_levels[:30]}")


if __name__ == "__main__":
    main()
