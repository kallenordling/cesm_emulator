# --- config I/O ---
import json, os, pathlib, argparse

def load_config(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if p.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml  # optional dep
        except ImportError as e:
            raise RuntimeError("PyYAML not installed; use a .json config or `pip install pyyaml`") from e
        with p.open("r") as f:
            return yaml.safe_load(f)
    else:
        with p.open("r") as f:
            return json.load(f)

def _parse_value(s: str):
    # turn "true"/"false"/numbers into proper types
    sl = s.lower()
    if sl in ("true","false"): return sl == "true"
    try:
        if "." in s: return float(s)
        return int(s)
    except ValueError:
        return s

def apply_overrides(cfg: dict, pairs: list[str]) -> None:
    # supports: --set train.batch_size=4 unet.num_res_blocks=3
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override (no '='): {pair}")
        key, val = pair.split("=", 1)
        val = _parse_value(val)
        d = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[parts[-1]] = val
