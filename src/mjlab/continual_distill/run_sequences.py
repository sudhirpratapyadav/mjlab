#!/usr/bin/env python3
"""
Runner script for executing multiple continual learning experiments.

This script reads task sequences from task_sequence.yaml and runs
continual_distill.py for each sequence defined in the file.

Usage:
    python run_sequences.py --sequence-config config/task_sequence.yaml [options]

    # With custom tasks config:
    python run_sequences.py --sequence-config config/task_sequence.yaml --tasks-config config/tasks.yaml

    # Pass additional arguments to continual_distill.py:
    python run_sequences.py --sequence-config config/task_sequence.yaml --learning-rate 1e-5 --si-coeff 2.0
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import yaml


def load_sequences(sequence_config_path: Path) -> Dict[str, List[str]]:
    """Load task sequences from YAML file.

    Args:
        sequence_config_path: Path to task_sequence.yaml

    Returns:
        Dictionary mapping sequence names to lists of task names
    """
    if not sequence_config_path.exists():
        raise FileNotFoundError(f"Sequence config not found: {sequence_config_path}")

    with sequence_config_path.open("r") as f:
        config = yaml.safe_load(f)

    sequences = config.get("sequences", {})
    if not sequences:
        raise ValueError(f"No sequences found in {sequence_config_path}")

    return sequences


def run_continual_distill(
    tasks_config: Path,
    task_sequence: List[str],
    sequence_name: str,
    extra_args: List[str],
) -> int:
    """Run continual_distill.py for a single sequence.

    Args:
        tasks_config: Path to tasks.yaml
        task_sequence: List of task names for this sequence
        sequence_name: Name of the sequence (for run naming)
        extra_args: Additional arguments to pass to continual_distill.py

    Returns:
        Return code from subprocess
    """
    # Build command
    cmd = [
        sys.executable,
        "-m",
        "mjlab.continual_distill.continual_distill",
        "--tasks-config",
        str(tasks_config),
        "--task-sequence",
        *task_sequence,
    ]

    # Add run name if not already specified in extra_args
    if "--run-name" not in extra_args:
        run_name = f"{sequence_name}_{int(time.time())}"
        cmd.extend(["--run-name", run_name])

    # Add any extra arguments
    cmd.extend(extra_args)

    print("=" * 80)
    print(f"Running sequence: {sequence_name}")
    print(f"Tasks: {' -> '.join(task_sequence)}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)
    print()

    # Run the command
    result = subprocess.run(cmd)
    return result.returncode


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multiple continual learning experiments from sequence definitions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sequence-config",
        type=Path,
        required=True,
        help="Path to task_sequence.yaml defining sequences to run.",
    )
    parser.add_argument(
        "--tasks-config",
        type=Path,
        default=None,
        help="Path to tasks.yaml (default: config/tasks.yaml in same directory as sequence-config).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop running sequences if one fails.",
    )

    # Parse known args - everything else will be passed to continual_distill.py
    args, unknown = parser.parse_known_args()

    # Store unknown args to pass through
    args.extra_args = unknown

    return args


def main() -> None:
    args = parse_args()

    # Load sequences
    sequences = load_sequences(args.sequence_config)
    print(f"Loaded {len(sequences)} sequence(s) from {args.sequence_config}")

    # Run all sequences
    sequences_to_run = sequences

    if not sequences_to_run:
        print("Error: No sequences to run!")
        sys.exit(1)

    # Determine tasks config path
    if args.tasks_config:
        tasks_config = args.tasks_config
    else:
        # Default: tasks.yaml in same directory as sequence config
        tasks_config = args.sequence_config.parent / "tasks.yaml"

    if not tasks_config.exists():
        print(f"Error: Tasks config not found: {tasks_config}")
        sys.exit(1)

    print(f"Using tasks config: {tasks_config}")
    print(f"Running {len(sequences_to_run)} sequence(s): {', '.join(sequences_to_run.keys())}")
    print()

    # Run each sequence
    results = {}
    for seq_name, task_list in sequences_to_run.items():
        return_code = run_continual_distill(
            tasks_config=tasks_config,
            task_sequence=task_list,
            sequence_name=seq_name,
            extra_args=args.extra_args,
        )

        results[seq_name] = return_code

        if return_code != 0:
            print(f"\nSequence '{seq_name}' failed with return code {return_code}")
            if args.stop_on_error:
                print("Stopping due to --stop-on-error flag.")
                break
        else:
            print(f"\nSequence '{seq_name}' completed successfully!")

        print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for seq_name, return_code in results.items():
        status = "SUCCESS" if return_code == 0 else f"FAILED (code {return_code})"
        print(f"  {seq_name}: {status}")

    # Exit with non-zero if any sequence failed
    if any(rc != 0 for rc in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
