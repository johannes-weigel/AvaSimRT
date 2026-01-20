from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    interactive_plots: bool = False
    save_all_plots: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "VisualizationConfig":
        d = dict(data or {})

        return cls(
            interactive_plots=bool(d.get("interactive_plots", False)),
            save_all_plots=bool(d.get("save_all_plots", False)),
        )