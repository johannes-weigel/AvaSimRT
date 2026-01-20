from __future__ import annotations

from pathlib import Path
import pytest

import matplotlib
from dotenv import load_dotenv

root_dir = Path(__file__).parent.parent
env_file = root_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)

matplotlib.use("Agg", force=True)

@pytest.fixture(scope="session")
def examples(pytestconfig) -> Path:
    return pytestconfig.rootpath / "examples"
