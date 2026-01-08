from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from avasimrt.result import SimResult, Sample
from .config import CsvExportConfig

_SANITIZE_RE = re.compile(r"[^0-9A-Za-z]+")


def _sanitize_token(value: str) -> str:
    return _SANITIZE_RE.sub("_", value).strip("_")


def _freq_to_token(freq: float) -> str:
    if float(freq).is_integer():
        return str(int(freq))
    
    s = f"{float(freq):.12g}".replace(".", "p")
    return _sanitize_token(s)


def export_simresult_to_csv(
    sim_result: SimResult,
    csv_path: Path,
    *,
    options: CsvExportConfig = CsvExportConfig(),
) -> None:
    """
    Writes one row per Sample (timestamp) into a CSV file.

    Columns:
      - timestamp, node snapshot fields
      - discovered per-anchor/per-antenna/per-frequency: *_real, *_imag
    """
    samples = sim_result.samples
    anchor_cols = _discover_anchor_columns(samples)

    base_fields = [
        "timestamp",
        "node_pos_x", "node_pos_y", "node_pos_z",
        "node_orientation_w", "node_orientation_x", "node_orientation_y", "node_orientation_z",
        "node_linear_velocity_x", "node_linear_velocity_y", "node_linear_velocity_z",
    ]
    fieldnames = base_fields + anchor_cols

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding=options.encoding, newline=options.newline) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=options.delimiter,
        )
        if options.write_header:
            writer.writeheader()

        for sample in samples:
            row = _sample_to_row(sample, anchor_cols)
            writer.writerow(row)


def _discover_anchor_columns(samples: list[Sample]) -> list[str]:
    cols: set[str] = set()

    for s in samples:
        if not s.readings:
            continue

        for anchor in s.readings:
            aid = _sanitize_token(anchor.anchor_id)

            for antenna in anchor.values:
                a_label = _sanitize_token(antenna.label)

                for c in antenna.frequencies:
                    ftoken = _freq_to_token(c.freq)
                    base = f"anchor_{aid}_{a_label}_{ftoken}"
                    cols.add(f"{base}_real")
                    cols.add(f"{base}_imag")

    return sorted(cols)


def _sample_to_row(sample: Sample, anchor_columns: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}

    row["timestamp"] = sample.timestamp

    px, py, pz = sample.node.position
    row["node_pos_x"] = px
    row["node_pos_y"] = py
    row["node_pos_z"] = pz

    ow, ox, oy, oz = sample.node.orientation
    row["node_orientation_w"] = ow
    row["node_orientation_x"] = ox
    row["node_orientation_y"] = oy
    row["node_orientation_z"] = oz

    lvx, lvy, lvz = sample.node.linear_velocity
    row["node_linear_velocity_x"] = lvx
    row["node_linear_velocity_y"] = lvy
    row["node_linear_velocity_z"] = lvz

    lookup: dict[str, Any] = {}
    if sample.readings:
        for anchor in sample.readings:
            aid = _sanitize_token(anchor.anchor_id)

            for antenna in anchor.values:
                a_label = _sanitize_token(antenna.label)

                for c in antenna.frequencies:
                    ftoken = _freq_to_token(c.freq)
                    base = f"anchor_{aid}_{a_label}_{ftoken}"
                    lookup[f"{base}_real"] = getattr(c, "real", "")
                    lookup[f"{base}_imag"] = getattr(c, "imag", "")

    for col in anchor_columns:
        row[col] = lookup.get(col, "")

    return row
