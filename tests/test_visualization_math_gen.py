from __future__ import annotations

import numpy as np

from avasimrt.result import ComplexReading
from avasimrt.visualization.math import amps_to_db, magnitude, phase_deg


def test_magnitude() -> None:
    v = ComplexReading(freq=1.0, real=3.0, imag=4.0)
    assert magnitude(v) == 5.0


def test_phase_deg_quadrants() -> None:
    # 45 degrees
    v1 = ComplexReading(freq=1.0, real=1.0, imag=1.0)
    assert abs(phase_deg(v1) - 45.0) < 1e-9

    # 180 degrees
    v2 = ComplexReading(freq=1.0, real=-1.0, imag=0.0)
    assert abs(phase_deg(v2) - 180.0) < 1e-9

    # -90 degrees
    v3 = ComplexReading(freq=1.0, real=0.0, imag=-1.0)
    assert abs(phase_deg(v3) + 90.0) < 1e-9


def test_amps_to_db_is_finite_for_zero() -> None:
    amps = np.array([0.0, 1.0, 10.0], dtype=float)
    db = amps_to_db(amps)

    assert np.isfinite(db[0])  # should not be -inf
    assert abs(db[1] - 0.0) < 1e-12
    assert abs(db[2] - 20.0) < 1e-12
