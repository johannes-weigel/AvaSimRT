from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .app import run
from .config import AnchorConfig, NodeConfig, SimConfig, VisualizationConfig, ReportingConfig

from .channelstate.config import ChannelStateConfig, SceneConfig as ChannelSceneConfig, ChannelConfig, RenderConfig
from .motion.config import MotionConfig, MotionDebug, MotionPhysics, MotionTime


@dataclass(frozen=True, slots=True)
class CliArgs:
    config: str | None
    run_id: str | None
    output: str
    delete_existing: bool

    debug: bool

    node: str | None
    anchors: list[str]
    scene_xml: str | None

    # motion
    sim_time: float
    sampling_rate: float
    time_step: float

    # channelstate
    freq_center: float
    sc_num: int
    sc_spacing: float
    reflection_depth: int
    seed: int

    render: bool
    render_every_n: int

    # visualization/reporting
    interactive_plots: bool
    save_all_plots: bool
    no_csv: bool


def _parse_node(s: str) -> NodeConfig:
    # "x,y,z[,size]" where z may be "none"
    parts = [p.strip() for p in s.split(",")]
    if len(parts) not in (3, 4):
        raise ValueError("node must be 'x,y,z[,size]'")
    x = float(parts[0])
    y = float(parts[1])
    z = None if parts[2].lower() in ("none", "null") else float(parts[2])
    size = float(parts[3]) if len(parts) == 4 else 0.2
    return NodeConfig(x=x, y=y, z=z, size=size)


def _parse_anchor(s: str) -> AnchorConfig:
    # "id,x,y,z[,size]" where z may be "none"
    parts = [p.strip() for p in s.split(",")]
    if len(parts) not in (4, 5):
        raise ValueError("anchor must be 'id,x,y,z[,size]'")
    a_id = parts[0]
    x = float(parts[1])
    y = float(parts[2])
    z = None if parts[3].lower() in ("none", "null") else float(parts[3])
    size = float(parts[4]) if len(parts) == 5 else 0.2
    return AnchorConfig(id=a_id, x=x, y=y, z=z, size=size)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="avasimrt",
        description=(
            "Toolchain to simulate avalanche-driven movement of a node and compute complex "
            "UWB channel evaluations from stationary anchors."
        ),
    )

    p.add_argument("--config", type=str, help="Path to a YAML config file. If set, all other CLI arguments are ignored.")

    p.add_argument("--run-id", type=str, help="Optional run identifier (ignored with --config).")
    p.add_argument("--output", type=str, default="output", help="Base output directory (ignored with --config).")
    p.add_argument("--delete-existing", action="store_true", help="Delete existing non-empty output directory.")

    p.add_argument("--debug", action="store_true", help="Enable debug logging and slow visual modes where applicable.")

    p.add_argument("--node", type=str, help="Node as 'x,y,z[,size]'. Use z=none to auto-place on terrain.")
    p.add_argument(
        "--anchor",
        dest="anchors",
        action="append",
        default=[],
        help="Anchor as 'id,x,y,z[,size]'. Use z=none to resolve via motion.",
    )
    p.add_argument("--scene-xml", type=str, help="Mitsuba/SionnaRT scene XML path (required for channelstate).")

    p.add_argument("--sim-time", type=float, default=60.0)
    p.add_argument("--sampling-rate", type=float, default=1.0)
    p.add_argument("--time-step", type=float, default=1.0 / 240.0)

    p.add_argument("--freq-center", type=float, default=3.8e9)
    p.add_argument("--sc-num", type=int, default=101)
    p.add_argument("--sc-spacing", type=float, default=5e6)
    p.add_argument("--reflect", dest="reflection_depth", type=int, default=3)
    p.add_argument("--seed", type=int, default=41)

    p.add_argument("--render", action="store_true", help="Enable scene rendering during channelstate step.")
    p.add_argument("--render-every-n", type=int, default=0, help="Render every n-th step (0 disables periodic rendering).")

    p.add_argument("--no-csv", action="store_true", help="Disable CSV export.")
    p.add_argument("--interactive-plots", action="store_true", help="Enable interactive visualization after run.")
    p.add_argument("--save-all-plots", action="store_true", help="Save all visualizations to output folder.")

    return p


def parse_args(argv: list[str] | None = None) -> CliArgs:
    ns = build_parser().parse_args(argv)
    return CliArgs(
        config=ns.config,
        run_id=ns.run_id,
        output=ns.output,
        delete_existing=ns.delete_existing,
        debug=ns.debug,
        node=ns.node,
        anchors=list(ns.anchors),
        scene_xml=ns.scene_xml,
        sim_time=ns.sim_time,
        sampling_rate=ns.sampling_rate,
        time_step=ns.time_step,
        freq_center=ns.freq_center,
        sc_num=ns.sc_num,
        sc_spacing=ns.sc_spacing,
        reflection_depth=ns.reflection_depth,
        seed=ns.seed,
        render=ns.render,
        render_every_n=ns.render_every_n,
        interactive_plots=ns.interactive_plots,
        save_all_plots=ns.save_all_plots,
        no_csv=ns.no_csv,
    )


def resolve_config(args: CliArgs) -> SimConfig:
    """Pure config resolution logic for unit testing."""
    if args.config:
        return SimConfig.from_yaml(args.config)

    if args.node is None:
        raise ValueError("--node is required if --config is not set")
    if not args.anchors:
        raise ValueError("at least one --anchor is required if --config is not set")

    node = _parse_node(args.node)
    anchors = [_parse_anchor(a) for a in args.anchors]

    motion = MotionConfig(
        time=MotionTime(sim_time=args.sim_time, sampling_rate=args.sampling_rate, time_step=args.time_step),
        physics=MotionPhysics(gravity_z=-9.81),
        debug=MotionDebug(mode="GUI" if args.debug else "DIRECT"),
    )

    channelstate: ChannelStateConfig | None
    if args.scene_xml is None:
        channelstate = None
    else:
        channelstate = ChannelStateConfig(
            scene=ChannelSceneConfig(xml_path=Path(args.scene_xml), out_dir=Path(args.output) / "channelstate" / "frames"),
            channel=ChannelConfig(
                freq_center=args.freq_center,
                sc_num=args.sc_num,
                sc_spacing=args.sc_spacing,
                reflection_depth=args.reflection_depth,
                seed=args.seed,
            ),
            render=RenderConfig(
                enabled=args.render,
                every_n_steps=args.render_every_n,
            ),
            debug=args.debug,
        )

    return SimConfig(
        run_id=args.run_id if args.run_id is not None else SimConfig().run_id,
        output=Path(args.output),
        delete_existing=args.delete_existing,
        debug=args.debug,
        node=node,
        anchors=anchors,
        motion=motion,
        channelstate=channelstate,
        reporting=ReportingConfig(enabled=True, csv=not args.no_csv),
        visualization=VisualizationConfig(
            interactive_plots=args.interactive_plots,
            save_all_plots=args.save_all_plots,
        ),
    )


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
