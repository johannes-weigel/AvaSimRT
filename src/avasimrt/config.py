from __future__ import annotations

import uuid
from dataclasses import dataclass, field


def _generate_run_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Top-level configuration for a single simulation run."""
    run_id: str = field(default_factory=_generate_run_id)
