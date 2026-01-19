from dataclasses import dataclass
from pathlib import Path

@dataclass
class ResolvedPosition:
    """A position with resolved z coordinate."""
    id: str
    x: float
    y: float
    z: float
    size: float