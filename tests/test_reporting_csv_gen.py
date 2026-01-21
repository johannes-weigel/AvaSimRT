from __future__ import annotations

import re
from pathlib import Path

from avasimrt.reporting import csv
from avasimrt.result import SimResult, Sample, NodeSnapshot, AnchorReading, AntennaReading, ComplexReading



# -----------------------------
# Sanitizer / freq token tests
# -----------------------------

def test_sanitize_token_replaces_non_alnum_and_trims_underscores() -> None:
    assert csv._sanitize_token("ABC-123") == "ABC_123"
    assert csv._sanitize_token("  a b  ") == "a_b"
    assert csv._sanitize_token("__x__") == "x"
    assert csv._sanitize_token("a/b\\c") == "a_b_c"


def test_freq_to_token_integer_is_plain_int() -> None:
    assert csv._freq_to_token(2400.0) == "2400"
    assert csv._freq_to_token(0.0) == "0"


def test_freq_to_token_decimal_uses_p_instead_of_dot() -> None:
    assert csv._freq_to_token(2.5) == "2p5"
    assert csv._freq_to_token(3.125) == "3p125"


def test_freq_to_token_is_filesystem_safe() -> None:
    # scientific notation / plus signs etc should be sanitized away
    tok = csv._freq_to_token(3.8e9)
    assert re.fullmatch(r"[0-9A-Za-z_]+", tok) is not None
    assert "." not in tok
    assert "-" not in tok  # minus should not survive sanitizer
    assert "+" not in tok


# -----------------------------
# Full export test
# -----------------------------

def test_export_simresult_to_csv_writes_header_plus_3_rows(tmp_path: Path) -> None:
    # Arrange: 3 samples -> header + 3 rows = 4 lines in file
    node = NodeSnapshot(
        position=(1.0, 2.0, 3.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.1, 0.2, 0.3),
        size=0.2,
    )

    readings = [
        AnchorReading(
            anchor_id="A-01",
            values=[
                AntennaReading(
                    label="ant-0",
                    frequencies=[
                        ComplexReading(freq=2.5, real=1.0, imag=0.0),
                    ],
                    mean_db=42
                )
            ],
            distance=42
        )
    ]

    samples = [
        Sample(timestamp=0.0, node=node, readings=readings),
        Sample(timestamp=1.0, node=node, readings=readings),
        Sample(timestamp=2.0, node=node, readings=readings)
    ]

    out = tmp_path / "results.csv"

    # Act
    csv.export_simresult_to_csv(samples, out)

    # Assert: correct number of lines
    text = out.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    assert len(lines) == 4  #
