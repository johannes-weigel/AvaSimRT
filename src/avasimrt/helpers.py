from typing import Any

def generate_run_id() -> str:
    import uuid
    return uuid.uuid4().hex

def _coerce_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in {"none", "null", "~", ""}:
        return None
    return v


def coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "y", "1", "on"}:
            return True
        if s in {"false", "no", "n", "0", "off"}:
            return False
    raise ValueError(f"Expected bool, got {v!r}")


def coerce_int(v: Any) -> int:
    v = _coerce_none(v)
    if v is None:
        raise ValueError("Expected int, got None")
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f"Expected int, got non-integer float {v!r}")
        return int(v)
    if isinstance(v, str):
        s = v.strip().replace("_", "")
        if "." in s:
            f = float(s)
            if not f.is_integer():
                raise ValueError(f"Expected int, got {v!r}")
            return int(f)
        return int(s)
    raise ValueError(f"Expected int, got {v!r}")


def coerce_float(v: Any) -> float:
    v = _coerce_none(v)
    if v is None:
        raise ValueError("Expected float, got None")
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("_", "")
        return float(s)
    raise ValueError(f"Expected float, got {v!r}")


def coerce_optional_float(v: Any) -> float | None:
    v = _coerce_none(v)
    if v is None:
        return None
    return coerce_float(v)

def get(d, key, default):
    return d[key] if key in d else default

def get_dict(d, key, name):
    v = d.get(key)
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError(f"{name} must be a mapping")
    return v

def get_list(d, key, name):
    v = d.get(key)
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError(f"{name} must be a list")
    return v