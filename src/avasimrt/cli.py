from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .app import run
from .config import AnchorConfig, NodeConfig, SimConfig, VisualizationConfig, ReportingConfig

from .channelstate.config import ChannelStateConfig, ChannelConfig, RenderConfig
from .motion.config import MotionConfig, MotionDebug, MotionPhysics, MotionTime


@dataclass(frozen=True, slots=True)
class CliArgs:
    config: str | None
    run_id: str | None
    output: str
    delete_existing: bool

    debug: bool

    nodes: list[str]
    anchors: list[str]
    scene_xml: str | None
    scene_obj: str | None
    scene_blender: str | None
    blender_cmd: str | None

    heightmap_resolution: float | None

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

    p.add_argument(
        "--node",
        dest="nodes",
        action="append",
        default=[],
        help="Node as 'x,y,z[,size]'. Use z=none to resolve via motion.",
    )

    p.add_argument(
        "--anchor",
        dest="anchors",
        action="append",
        default=[],
        help="Anchor as 'id,x,y,z[,size]'. Use z=none to resolve via motion.",
    )
    p.add_argument("--scene-xml", type=str, help="Mitsuba/SionnaRT scene XML path (required for channelstate).")
    p.add_argument("--scene-obj", type=str, help="PyBullet scene OBJ path (required for motion).")
    p.add_argument("--scene-blender", type=str, help="Blender scene file (.blend) to export OBJ/XML from.")
    p.add_argument(
        "--blender-cmd", 
        type=str, 
        default=None,
        help="Blender executable name or path. Can also be set via AVASIMRT_BLENDER_CMD environment variable. Default: 'blender'"
    )

    p.add_argument(
        "--heightmap-resolution",
        type=float,
        default=None,
        help="Heightmap resolution in meters (default: 0.5).",
    )

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
        nodes=list(ns.nodes),
        anchors=list(ns.anchors),
        scene_xml=ns.scene_xml,
        scene_obj=ns.scene_obj,
        scene_blender=ns.scene_blender,
        blender_cmd=ns.blender_cmd,
        heightmap_resolution=ns.heightmap_resolution,
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

    nodes = [_parse_node(n) for n in args.nodes]
    anchors = [_parse_anchor(a) for a in args.anchors]

    motion = MotionConfig(
        time=MotionTime(sim_time=args.sim_time, sampling_rate=args.sampling_rate, time_step=args.time_step),
        physics=MotionPhysics(gravity_z=-9.81),
        debug=MotionDebug(mode="GUI" if args.debug else "DIRECT"),
    )

    # Validate scene file arguments
    has_blender = args.scene_blender is not None
    has_xml = args.scene_xml is not None
    has_obj = args.scene_obj is not None

    if has_blender:
        if has_xml or has_obj:
            raise ValueError("Cannot specify both --scene-blender and --scene-xml/--scene-obj")
        assert args.scene_blender is not None  # Type guard
        scene_blender = Path(args.scene_blender)
        scene_xml = None
        scene_obj = None
    elif has_xml and has_obj:
        assert args.scene_xml is not None and args.scene_obj is not None  # Type guard
        scene_blender = None
        scene_xml = Path(args.scene_xml)
        scene_obj = Path(args.scene_obj)
    elif has_xml or has_obj:
        raise ValueError("Both --scene-xml and --scene-obj must be provided together")
    else:
        # No scene files specified - disable channelstate
        scene_blender = None
        scene_xml = None
        scene_obj = None

    # Configure channelstate based on whether we have scene files
    if scene_blender is not None or (scene_xml is not None and scene_obj is not None):
        channelstate = ChannelStateConfig(
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
    else:
        channelstate = None

    return SimConfig(
        run_id=args.run_id,
        scene_xml=scene_xml,
        scene_obj=scene_obj,
        scene_blender=scene_blender,
        heightmap_resolution=args.heightmap_resolution,
        output=Path(args.output),
        delete_existing=args.delete_existing,
        debug=args.debug,
        nodes=nodes,
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
        result = run(config, blender_cmd=args.blender_cmd)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not result.successful:
        print(f"ERROR: {result.message}", file=sys.stderr)
        return 1

    print("Run completed successfully.")
    if (result.message is not None):
        print(result.message)
    print(f"Run ID: {result.run_id}")
    if result.output_dir is not None:
        print(f"Output directory: {result.output_dir}")
    return 0
