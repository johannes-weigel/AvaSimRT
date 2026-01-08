from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class CsvExportConfig:
    write_header: bool = True
    delimiter: str = ","
    encoding: str = "utf-8"
    newline: str = ""
