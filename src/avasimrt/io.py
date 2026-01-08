from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import SimConfig


@dataclass(frozen=True, slots=True)
class PrepareDirResult:
    ok: bool
    path: Path | None
    message: str


def resolve_output_base(config: SimConfig) -> Path:
    base = (config.output or "").strip()
    if not base:
        base = "output"
    return Path(base)


def prepare_output_dir(config: SimConfig) -> PrepareDirResult:
    """
    Prepare output directory: <base>/<run_id>

    Rules:
    - base defaults to "output" if config.output is empty/blank
    - base is relative to current working directory (unless absolute)
    - base must exist (create if needed)
    - run_dir must either be non-existent, or empty, or delete_existing=True
    """
    base = resolve_output_base(config)
    run_dir = base / config.run_id

    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return PrepareDirResult(False, None, f"Failed to create output base directory '{base}': {e}")

    if run_dir.exists():
        if run_dir.is_file():
            return PrepareDirResult(False, None, f"Output path '{run_dir}' exists and is a file, expected a directory.")

        try:
            has_contents = any(run_dir.iterdir())
        except OSError as e:
            return PrepareDirResult(False, None, f"Failed to inspect output directory '{run_dir}': {e}")

        if has_contents:
            if not config.delete_existing:
                return PrepareDirResult(
                    False,
                    None,
                    f"Output directory '{run_dir}' is not empty. Use delete_existing=True to remove it.",
                )

            try:
                shutil.rmtree(run_dir)
            except OSError as e:
                return PrepareDirResult(False, None, f"Failed to delete existing output directory '{run_dir}': {e}")

            try:
                run_dir.mkdir(parents=True, exist_ok=False)
            except OSError as e:
                return PrepareDirResult(False, None, f"Failed to recreate output directory '{run_dir}': {e}")
    else:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except OSError as e:
            return PrepareDirResult(False, None, f"Failed to create output directory '{run_dir}': {e}")

    return PrepareDirResult(True, run_dir, "Output directory prepared.")
