from __future__ import annotations

from .config import SimConfig
from .result import SimResult


def run(config: SimConfig) -> SimResult:
    """Run a simulation."""

    _ = config  # placeholder

    return SimResult(successful=True, run_id=config.run_id, message="empty run completed")
