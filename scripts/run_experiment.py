#!/usr/bin/env python3
"""
ERCS Experiment Runner.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/default.yaml --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ercs.config.schemas import ExperimentConfig, load_experiment_config


def setup_logging(log_level: str, verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else getattr(logging, log_level.upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_experiment_plan(config: ExperimentConfig) -> None:
    """Print the experiment execution plan."""
    print("\n" + "=" * 60)
    print("ERCS EXPERIMENT PLAN")
    print("=" * 60)
    print(f"\nExperiment: {config.experiment_name}")
    if config.description:
        print(f"Description: {config.description}")

    print(f"\nAlgorithms:")
    for alg in config.enabled_algorithms:
        print(f"  - {alg.value}")

    print(f"\nConnectivity scenarios:")
    for scenario in config.connectivity_scenarios:
        print(f"  - {scenario.connectivity_level * 100:.0f}%: {scenario.num_runs} runs")

    print(f"\nTotal runs: {config.total_runs}")
    print(f"  = {len(config.enabled_algorithms)} algorithms × {len(config.connectivity_scenarios)} scenarios × {config.connectivity_scenarios[0].num_runs} runs")

    print(f"\nKey parameters:")
    print(f"  Nodes: {config.network.primary_node_count}")
    print(f"  Duration: {config.scenario.simulation_duration_seconds}s")
    print(f"  Coordination interval: {config.coordination.update_interval_seconds}s")
    print(f"\nOutput: {config.output_directory}")
    print("=" * 60)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ERCS experiments")
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dry-run", "-n", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    setup_logging(config.log_level, args.verbose)
    logger = logging.getLogger("ercs")
    logger.info(f"Loaded configuration from: {args.config}")

    print_experiment_plan(config)

    if args.dry_run:
        print("\n✓ Configuration validated successfully (dry run)")
        return 0

    print("\n⚠ Simulation execution not yet implemented")
    print("  Next: Implement Phases 1-6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
