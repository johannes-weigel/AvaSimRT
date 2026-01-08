from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

from .motion.config import (
    MotionConfig, 
    MotionTime, 
    MotionDebug, 
    MotionPhysics
)
from .channelstate.config import (
    ChannelStateConfig,
    SceneConfig as ChannelSceneConfig,
    ChannelConfig,
    RenderConfig,
)

def _generate_run_id() -> str:
    import uuid
    return uuid.uuid4().hex

def _coerce_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in {"none", "null", "~", ""}:
        return None
    return v


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "y", "1", "on"}:
            return True
        if s in {"false", "no", "n", "0", "off"}:
            return False
    raise ValueError(f"Expected bool, got {v!r}")


def _coerce_int(v: Any) -> int:
    v = _coerce_none(v)
    if v is None:
        raise ValueError("Expected int, got None")
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f"Expected int, got non-integer float {v!r}")
        return int(v)
    if isinstance(v, str):
        s = v.strip().replace("_", "")
        # allow "101.0" style
        if "." in s:
            f = float(s)
            if not f.is_integer():
                raise ValueError(f"Expected int, got {v!r}")
            return int(f)
        return int(s)
    raise ValueError(f"Expected int, got {v!r}")


def _coerce_float(v: Any) -> float:
    v = _coerce_none(v)
    if v is None:
        raise ValueError("Expected float, got None")
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("_", "")
        # handles "3.8e9" reliably
        return float(s)
    raise ValueError(f"Expected float, got {v!r}")


def _coerce_optional_float(v: Any) -> float | None:
    v = _coerce_none(v)
    if v is None:
        return None
    return _coerce_float(v)


def _coerce_optional_int(v: Any) -> int | None:
    v = _coerce_none(v)
    if v is None:
        return None
    return _coerce_int(v)

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

        if "node" in data and isinstance(data["node"], dict):
            n = dict(data["node"])
            n["x"] = _coerce_float(n.get("x", 0.0))
            n["y"] = _coerce_float(n.get("y", 0.0))
            n["z"] = _coerce_optional_float(n.get("z"))
            n["size"] = _coerce_float(n.get("size", 0.2))
            data["node"] = NodeConfig(**n)

        if "anchors" in data and isinstance(data["anchors"], list):
            anchors: list[AnchorConfig] = []
            for a in data["anchors"]:
                if not isinstance(a, dict):
                    raise ValueError("anchors must be a list of mappings")
                aa = dict(a)
                aa["x"] = _coerce_float(aa["x"])
                aa["y"] = _coerce_float(aa["y"])
                aa["z"] = _coerce_optional_float(aa.get("z"))
                aa["size"] = _coerce_float(aa.get("size", 0.2))
                anchors.append(AnchorConfig(**aa))
            data["anchors"] = anchors

        if "motion" in data and isinstance(data["motion"], dict):
            m = dict(data["motion"])

            if "time" in m and isinstance(m["time"], dict):
                t = dict(m["time"])
                t["sim_time"] = _coerce_float(t.get("sim_time", 60.0))
                t["sampling_rate"] = _coerce_float(t.get("sampling_rate", 1.0))
                t["time_step"] = _coerce_float(t.get("time_step", 1.0 / 240.0))
                m["time"] = MotionTime(**t)

            if "physics" in m and isinstance(m["physics"], dict):
                ph = dict(m["physics"])
                ph["gravity_z"] = _coerce_float(ph.get("gravity_z", -9.81))
                m["physics"] = MotionPhysics(**ph)

            if "debug" in m and isinstance(m["debug"], dict):
                d = dict(m["debug"])
                m["debug"] = MotionDebug(**d)

            data["motion"] = MotionConfig(**m)

        if "reporting" in data and isinstance(data["reporting"], dict):
            rep = dict(data["reporting"])
            if "enabled" in rep:
                rep["enabled"] = _coerce_bool(rep["enabled"])
            if "csv" in rep:
                rep["csv"] = _coerce_bool(rep["csv"])
            data["reporting"] = ReportingConfig(**rep)

        if "visualization" in data and isinstance(data["visualization"], dict):
            v = dict(data["visualization"])
            if "interactive_plots" in v:
                v["interactive_plots"] = _coerce_bool(v["interactive_plots"])
            if "save_all_plots" in v:
                v["save_all_plots"] = _coerce_bool(v["save_all_plots"])
            data["visualization"] = VisualizationConfig(**v)

        if "channelstate" in data and isinstance(data["channelstate"], dict):
            cs = dict(data["channelstate"])

            if "scene" in cs and isinstance(cs["scene"], dict):
                sc = dict(cs["scene"])
                cs["scene"] = ChannelSceneConfig(
                    xml_path=Path(sc["xml_path"]),
                    out_dir=Path(sc["out_dir"]),
                )

            if "channel" in cs and isinstance(cs["channel"], dict):
                ch = dict(cs["channel"])
                ch["freq_center"] = _coerce_float(ch.get("freq_center", 3.8e9))
                ch["sc_num"] = _coerce_int(ch.get("sc_num", 101))
                ch["sc_spacing"] = _coerce_float(ch.get("sc_spacing", 5e6))
                ch["reflection_depth"] = _coerce_int(ch.get("reflection_depth", 3))
                ch["seed"] = _coerce_int(ch.get("seed", 41))
                cs["channel"] = ChannelConfig(**ch)

            if "render" in cs and isinstance(cs["render"], dict):
                r = dict(cs["render"])
                r["enabled"] = _coerce_bool(r.get("enabled", False))
                r["every_n_steps"] = _coerce_int(r.get("every_n_steps", 0))
                cs["render"] = RenderConfig(**r)

            if "debug" in cs:
                cs["debug"] = _coerce_bool(cs["debug"])

            data["channelstate"] = ChannelStateConfig(**cs)

        if "delete_existing" in data:
            data["delete_existing"] = _coerce_bool(data["delete_existing"])
        if "debug" in data:
            data["debug"] = _coerce_bool(data["debug"])

        return cls(**data)
