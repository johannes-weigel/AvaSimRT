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
    generate_run_id, coerce_bool, coerce_float, 
    get_dict, get_list)


@dataclass(frozen=True, slots=True)
class NodeConfig:
    x: float = 0.0
    y: float = 0.0
    z: float | None = None
    size: float = 0.2

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"node.size must be > 0, got {self.size!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "NodeConfig":
        d = dict(data or {})

        return cls(
            x=coerce_float(d.get("x", cls.x)),
            y=coerce_float(d.get("y", cls.y)),
            z=coerce_float(d["z"]) if d.get("z") is not None else None,
            size=coerce_float(d.get("size", cls.size)),
        )

@dataclass(frozen=True, slots=True)
class AnchorConfig:
    id: str
    x: float
    y: float
    z: float | None = None
    size: float = 0.2

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("anchor.id must not be empty")
        if self.size <= 0:
            raise ValueError(f"anchor.size must be > 0, got {self.size!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnchorConfig":
        d = dict(data)

        return cls(
            id=str(d["id"]),
            x=coerce_float(d["x"]),
            y=coerce_float(d["y"]),
            z=coerce_float(d["z"]) if d.get("z") is not None else None,
            size=coerce_float(d.get("size", cls.size)),
        )


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Top-level configuration for a single simulation run."""
    run_id: str = field(default_factory=generate_run_id)

    output: Path = Path("output")
    delete_existing: bool = False

    # Shared
    node: NodeConfig = NodeConfig()
    anchors: list[AnchorConfig] = field(default_factory=list)

    # Steps
    motion: MotionConfig = MotionConfig()
    channelstate: ChannelStateConfig | None = None
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

        return cls(
            run_id=d.get("run_id", generate_run_id()),

            output = Path(d.get("output", "output")),
            delete_existing=d.get("delete_existing", "False"),
            debug=coerce_bool(d["debug"]) if "debug" in d else cls.debug,

            node=NodeConfig.from_dict(get_dict(d, "node", "node") or {}),
            anchors=[
                AnchorConfig.from_dict(a)
                for a in get_list(d, "anchors", "anchors")
            ],

            motion=MotionConfig.from_dict(get_dict(d, "motion", "motion") or {}),
            reporting=ReportingConfig.from_dict(get_dict(d, "reporting", "reporting") or {}),
            visualization=VisualizationConfig.from_dict(get_dict(d, "visualization", "visualization") or {}),

            channelstate=(
                ChannelStateConfig.from_dict(cs)
                if (cs := get_dict(d, "channelstate", "channelstate")) is not None
                else None
            ),
        )


