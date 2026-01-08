from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import yaml


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
class SimConfig:
    """Top-level configuration for a single simulation run."""
    run_id: str = field(default_factory=_generate_run_id)

    output: str = "output"
    delete_existing: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file does not exist: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("Config file must contain a YAML mapping at top level.")

        return cls(**data)
