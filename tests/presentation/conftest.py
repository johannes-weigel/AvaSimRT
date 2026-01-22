from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pytest


@dataclass(frozen=True)
class PresentationConfig:
    """Shared config for all presentation tests."""
    resolution: tuple[int, int] = (1920, 1080)


@pytest.fixture(scope="session")
def presentation_output(pytestconfig) -> Path:
    """Output directory for presentation artifacts."""
    output = pytestconfig.rootpath / "presentation_output"
    output.mkdir(exist_ok=True)
    return output


@pytest.fixture(scope="session")
def presentation_config() -> PresentationConfig:
    """Shared configuration for presentation tests."""
    return PresentationConfig()
