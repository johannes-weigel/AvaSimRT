from pathlib import Path
from dotenv import load_dotenv

from .app import run
from .config import SimConfig
from .result import SimResult

_env_file = Path.cwd() / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

__all__ = ["run", "SimConfig", "SimResult"]
