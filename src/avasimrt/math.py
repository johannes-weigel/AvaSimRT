import numpy as np
import math

from .result import ComplexReading

EPS = 1e-12


def magnitude(v: ComplexReading) -> float:
    return float(np.hypot(v.real, v.imag))

def phase_deg(v: ComplexReading) -> float:
    return float(np.degrees(np.arctan2(v.imag, v.real)))

def amps_to_db(amps: np.ndarray) -> np.ndarray:
    # avoid -inf for zeros
    amps_safe = np.maximum(amps, EPS)
    return 20.0 * np.log10(amps_safe)

def power_to_db(p: np.ndarray) -> np.ndarray:
    safe = np.maximum(p, EPS)
    return 10.0 * np.log10(safe)

def db20_to_amp(db20: float) -> float:
    return 10.0 ** (db20 / 20.0)

def mean_db_from_values(values: list[ComplexReading]) -> float:
    if not values:
        return float("nan")

    amps = np.array([magnitude(c) for c in values], dtype=float)
    amps_db = amps_to_db(amps)
    return float(amps_db.mean())

def distance(node: tuple[float, float, float],
             anchor: tuple[float, float, float]) -> float:
    return math.sqrt(
        (node[0] - anchor[0]) ** 2 +
        (node[1] - anchor[1]) ** 2 +
        (node[2] - anchor[2]) ** 2
    )