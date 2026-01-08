import numpy as np
import math
from typing import List
from .config import AnchorConfig
from .result import ComplexReading, NodeSnapshot

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

def mean_db_from_values(values: List[ComplexReading]) -> float:
    if not values:
        return float("nan")

    amps = np.array([magnitude(c) for c in values], dtype=float)
    amps_db = amps_to_db(amps)
    return float(amps_db.mean())

def distance(node: NodeSnapshot,
             anchor: AnchorConfig) -> float:
    return math.sqrt(
        (node.position[0] - anchor.x) ** 2 +
        (node.position[1] - anchor.y) ** 2 +
        (node.position[2] - anchor.z if anchor.z is not None else 0) ** 2
    )