from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SimResult:
    """Result of a simulation run."""
    successful: bool
    run_id: str
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str | None = None
    output_dir: Path | None = None
