from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import yaml

from .motion.config import MotionConfig
from .channelstate.config import ChannelStateConfig, SceneConfig as ChannelSceneConfig

def _generate_run_id() -> str:
    import uuid
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class NodeConfig:
    x: float = 0.0
    y: float = 0.0
    z: float | None = None
    size: float = 0.2

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"node.size must be > 0, got {self.size!r}")


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


@dataclass(frozen=True, slots=True)
class ReportingConfig:
    enabled: bool = True
    csv: bool = True


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    interactive_plots: bool = False
    save_all_plots: bool = False


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Top-level configuration for a single simulation run."""
    run_id: str = field(default_factory=_generate_run_id)

    output: str = "output"
    delete_existing: bool = False

    # Shared inputs
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

        # nested conversions
        if "node" in data and isinstance(data["node"], dict):
            data["node"] = NodeConfig(**data["node"])

        if "anchors" in data and isinstance(data["anchors"], list):
            data["anchors"] = [AnchorConfig(**a) for a in data["anchors"]]

        if "motion" in data and isinstance(data["motion"], dict):
            data["motion"] = MotionConfig(**data["motion"])

        if "reporting" in data and isinstance(data["reporting"], dict):
            data["reporting"] = ReportingConfig(**data["reporting"])

        if "visualization" in data and isinstance(data["visualization"], dict):
            data["visualization"] = VisualizationConfig(**data["visualization"])

        if "channelstate" in data and isinstance(data["channelstate"], dict):
            cs = data["channelstate"]

            if "scene" in cs and isinstance(cs["scene"], dict):
                sc = cs["scene"]
                cs["scene"] = ChannelSceneConfig(
                    xml_path=Path(sc["xml_path"]),
                    out_dir=Path(sc["out_dir"]),
                )

            data["channelstate"] = ChannelStateConfig(**cs)

        return cls(**data)
