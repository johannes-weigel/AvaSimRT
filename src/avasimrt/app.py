from __future__ import annotations

from .config import SimConfig
from .io import prepare_output_dir
from .result import SimResult


def run(config: SimConfig) -> SimResult:
    """Run a simulation."""
    prep = prepare_output_dir(config)
    if not prep.ok or prep.path is None:
        return SimResult(
            successful=False,
            message=prep.message,
            run_id=config.run_id,
            output_dir=None,
        )

    # placeholder
    return SimResult(
        successful=True,
        run_id=config.run_id,
        message="empty run completed",
        output_dir=prep.path,
    )
