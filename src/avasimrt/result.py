from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from avasimrt.motion.result import NodeSnapshot

@dataclass(frozen=True, slots=True)
class ComplexReading:
    """Result of one complex channel evolution H(freq) = real + i*imag."""
    freq: float
    real: float
    imag: float


@dataclass(frozen=True, slots=True)
class AntennaReading:
    label: str
    mean_db: float
    frequencies: list[ComplexReading]


@dataclass(frozen=True, slots=True)
class AnchorReading:
    """Value reported by a single anchor at a given time."""
    anchor_id: str
    distance: float
    values: list[AntennaReading]


@dataclass(frozen=True, slots=True)
class Sample:
    """One time sample of simulation outputs."""
    timestamp: float
    node: NodeSnapshot
    readings: list[AnchorReading] | None = None
    image: Path | None = None


@dataclass(frozen=True, slots=True)
class SimResult:
    """Top-level run result integrating metadata + produced samples."""
    successful: bool
    run_id: str | None
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str | None = None
    output_dir: Path | None = None
    samples: list[Sample] = field(default_factory=list)

    def with_sample(self, sample: Sample) -> "SimResult":
        return SimResult(
            successful=self.successful,
            message=self.message,
            run_id=self.run_id,
            output_dir=self.output_dir,
            created_at_utc=self.created_at_utc,
            samples=[*self.samples, sample],
        )
