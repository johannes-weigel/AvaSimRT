from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Any
import yaml

from .motion.config import MotionConfig
from .channelstate.config import ChannelStateConfig
from .visualization.config import VisualizationConfig
from .reporting.config import ReportingConfig
from .helpers import (
    coerce_bool, coerce_float, 
    get_dict, get_list)

@dataclass(frozen=True, slots=True)
class PositionConfig:
    id: str
    x: float
    y: float
    z: float | None = None
    z_offset: float | None = None
    size: float = 0.2

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("position.id must not be empty")
        if self.size <= 0:
            raise ValueError(f"position.size must be > 0, got {self.size!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PositionConfig":
        d = dict(data)

        return cls(
            id=str(d["id"]),
            x=coerce_float(d["x"]),
            y=coerce_float(d["y"]),
            z=coerce_float(d["z"]) if d.get("z") is not None else None,
            z_offset=coerce_float(d["z_offset"]) if d.get("z_offset") is not None else None,
            size=coerce_float(d.get("size", 0.2)),
        )


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Top-level configuration for a single simulation run."""
    scene_xml: Path | None = None
    scene_obj: Path | None = None
    scene_blender: Path | None = None
    scene_meshes: Path | None = None

    run_id: str | None = None

    output: Path | None = None
    delete_existing: bool = False

    heightmap_npy: Path | None = None
    heightmap_resolution: float | None = None

    # Trajectory caching and visualization
    trajectory_cache_dir: Path | None = None  # Load from this dir instead of simulating
    trajectory_cache_filter: str | None = None  # Only load trajectories matching this filter
    trajectory_save: bool = False  # Save trajectories after simulation
    trajectory_plots_png: bool = False  # Save PNG visualizations
    trajectory_plots_html: bool = False  # Save interactive HTML visualizations

    # Channelstate caching
    channelstate_cache_dir: Path | None = None  # Load from this dir instead of computing
    channelstate_save: bool = False  # Save channelstate after computation

    # Shared
    nodes: list[PositionConfig] = field(default_factory=list)
    anchors: list[PositionConfig] = field(default_factory=list)

    # Steps
    motion: MotionConfig = MotionConfig()
    channelstate: ChannelStateConfig = ChannelStateConfig()
    reporting: ReportingConfig = ReportingConfig()
    visualization: VisualizationConfig = VisualizationConfig()

    debug: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file does not exist: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("Config file must contain a YAML mapping at top level.")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimConfig":
        d = dict(data or {})

        heightmap_dict = get_dict(d, "heightmap", "heightmap") or {}
        heightmap_resolution = coerce_float(heightmap_dict["resolution"]) if "resolution" in heightmap_dict else None
        heightmap_npy = Path(heightmap_dict["npy"]) if "npy" in heightmap_dict else None

        trajectory_dict = get_dict(d, "trajectory", "trajectory") or {}
        channelstate_dict = get_dict(d, "channelstate", "channelstate") or {}

        return cls(
            run_id=d.get("run_id"),
            scene_xml = Path(d["xml"]) if d.get("xml") is not None else None,
            scene_obj = Path(d["obj"]) if d.get("obj") is not None else None,
            scene_blender = Path(d["blender"]) if d.get("blender") is not None else None,
            scene_meshes = Path(d["meshes"]) if d.get("meshes") is not None else None,

            output = Path(d.get("output", "output")),
            delete_existing=coerce_bool(d.get("delete_existing", False)),
            debug=coerce_bool(d.get("debug", False)),

            heightmap_npy=heightmap_npy,
            heightmap_resolution=heightmap_resolution,

            trajectory_cache_dir=Path(trajectory_dict["cache_dir"]) if trajectory_dict.get("cache_dir") else None,
            trajectory_cache_filter=str(trajectory_dict["cache_filter"]) if trajectory_dict.get("cache_filter") else None,
            trajectory_save=coerce_bool(trajectory_dict.get("save", False)),
            trajectory_plots_png=coerce_bool(trajectory_dict.get("plots_png", False)),
            trajectory_plots_html=coerce_bool(trajectory_dict.get("plots_html", False)),

            channelstate_cache_dir=Path(channelstate_dict["cache_dir"]) if channelstate_dict.get("cache_dir") else None,
            channelstate_save=coerce_bool(channelstate_dict.get("save", False)),

            nodes=[
                PositionConfig.from_dict(a)
                for a in get_list(d, "nodes", "nodes")
            ],
            anchors=[
                PositionConfig.from_dict(a)
                for a in get_list(d, "anchors", "anchors")
            ],

            motion=MotionConfig.from_dict(get_dict(d, "motion", "motion") or {}),
            reporting=ReportingConfig.from_dict(get_dict(d, "reporting", "reporting") or {}),
            visualization=VisualizationConfig.from_dict(get_dict(d, "visualization", "visualization") or {}),

            channelstate=(ChannelStateConfig.from_dict(get_dict(d, "channelstate", "channelstate") or {})
            )
        )


