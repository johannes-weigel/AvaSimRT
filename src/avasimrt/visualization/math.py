from __future__ import annotations

import numpy as np

from avasimrt.result import ComplexReading


def magnitude(v: ComplexReading) -> float:
    return float(np.hypot(v.real, v.imag))


def phase_deg(v: ComplexReading) -> float:
    return float(np.degrees(np.arctan2(v.imag, v.real)))


def amps_to_db(amps: np.ndarray) -> np.ndarray:
    # avoid -inf for zeros
    amps_safe = np.maximum(amps, 1e-12)
    return 20.0 * np.log10(amps_safe)
