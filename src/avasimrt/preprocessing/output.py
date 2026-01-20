from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

def _generate_run_id() -> str:
    import uuid
    return uuid.uuid4().hex

def prepare_output(out_base: Path, run_id: str | None, delete_existing: bool) -> tuple[Path, str]:
    try:
        out_base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create output base directory '{out_base}': {e}") from e

    if run_id is None:
        run_id = _generate_run_id()
    out_dir = out_base / run_id

    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(f"Output path '{out_dir}' exists but is not a directory")

        try:
            has_contents = any(out_dir.iterdir())
        except OSError as e:
            raise OSError(f"Failed to inspect output directory '{out_dir}': {e}") from e

        if has_contents:
            if not delete_existing:
                raise FileExistsError(f"Output directory '{out_dir}' is not empty. "
                                      f"Use delete_existing=True to remove it.")

            logger.info(f"Deleting existing output directory: {out_dir}")
            try:
                shutil.rmtree(out_dir)
            except OSError as e:
                raise OSError(
                    f"Failed to delete existing output directory '{out_dir}': {e}"
                ) from e

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create output directory '{out_dir}': {e}") from e
    
    return out_dir, run_id

