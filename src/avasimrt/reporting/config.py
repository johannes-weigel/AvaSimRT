from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from avasimrt.helpers import coerce_bool


@dataclass(frozen=True, slots=True)
class CsvExportConfig:
    write_header: bool = True
    delimiter: str = ","
    encoding: str = "utf-8"
    newline: str = ""

@dataclass(frozen=True, slots=True)
class ReportingConfig:
    enabled: bool = True
    csv: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ReportingConfig":
        d = dict(data or {})

        return cls(
            enabled=coerce_bool(d.get("enabled", cls.enabled)),
            csv=coerce_bool(d.get("csv", cls.csv)),
        )
