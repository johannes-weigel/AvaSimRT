from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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
            enabled=bool(d.get("enabled", cls.enabled)),
            csv=bool(d.get("csv", cls.csv)),
        )
