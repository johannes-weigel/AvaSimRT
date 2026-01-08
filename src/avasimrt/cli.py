from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from .app import run
from .config import SimConfig


@dataclass(frozen=True, slots=True)
class CliArgs:
    config: str | None
    run_id: str | None
    output: str
    delete_existing: bool


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="avasimrt",
        description=(
            "Toolchain to simulate avalanche-driven movement of a node and compute complex "
            "UWB channel evaluations from stationary anchors."
        )
    )

    p.add_argument(
        "--config",
        type=str,
        help="Path to a YAML config file. If set, all other CLI arguments are ignored.",
    )
    
    p.add_argument("--run-id", type=str, help="Optional run identifier (ignored with --config).")
    p.add_argument("--output", type=str, default="output", help="Base output directory (ignored with --config).")
    p.add_argument("--delete-existing", action="store_true", help="Delete existing non-empty output directory.")
    return p


def parse_args(argv: list[str] | None = None) -> CliArgs:
    ns = build_parser().parse_args(argv)
    return CliArgs(
        config=ns.config,
        run_id=ns.run_id,
        output=ns.output,
        delete_existing=ns.delete_existing,
    )


def resolve_config(args: CliArgs) -> SimConfig:
    """Pure config resolution logic for unit testing."""
    if args.config:
        return SimConfig.from_yaml(args.config)

    if args.run_id is not None:
        return SimConfig(run_id=args.run_id, output=args.output, delete_existing=args.delete_existing)

    return SimConfig(output=args.output, delete_existing=args.delete_existing)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        config = resolve_config(args)
        result = run(config)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not result.successful:
        print(f"ERROR: {result.message}", file=sys.stderr)
        return 1

    print("Run completed successfully.")
    print(f"Run ID: {result.run_id}")
    if result.output_dir is not None:
        print(f"Output directory: {result.output_dir}")
    return 0
