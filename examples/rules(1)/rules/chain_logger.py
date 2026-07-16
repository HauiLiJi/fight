import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path


_BASE_DIR = Path(__file__).resolve().parent / "chain_logs"
_file = None
_file_key = None
_context = {
    "episode_id": "episode",
    "side": None,
    "step": None,
    "sim_time": None,
}
_errors = []


def reset(context=None):
    global _file, _file_key, _context, _errors
    close()
    _errors = []
    _file_key = None
    _context = {
        "episode_id": _safe_name(_episode_id(context)),
        "side": None,
        "step": None,
        "sim_time": None,
    }


def close():
    global _file
    if _file is None:
        return
    try:
        _file.flush()
        _file.close()
    except Exception as error:
        _remember_error(error)
    finally:
        _file = None


def set_context(observation=None, side=None, step=None, sim_time=None):
    if observation is not None:
        side = getattr(observation, "side", side)
        step = getattr(observation, "step_index", step)
        sim_time = getattr(observation, "sim_time", sim_time)
    if side is not None:
        _context["side"] = str(side)
    if step is not None:
        try:
            _context["step"] = int(step)
        except Exception:
            _context["step"] = step
    if sim_time is not None:
        try:
            _context["sim_time"] = float(sim_time)
        except Exception:
            _context["sim_time"] = sim_time


def log_event(event_type, payload=None):
    payload = payload or {}
    try:
        item = {
            "type": str(event_type),
            "episode_id": _context.get("episode_id"),
            "side": _context.get("side"),
            "step": _context.get("step"),
            "sim_time": _context.get("sim_time"),
        }
        if _errors:
            item["logger_errors"] = list(_errors[-3:])
        item.update(_jsonable(payload))
        file_obj = _ensure_file()
        if file_obj is None:
            return
        file_obj.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
        file_obj.flush()
    except Exception as error:
        _remember_error(error)


def plan_summary(plan):
    if plan is None:
        return None
    return {
        "plan_id": getattr(plan, "plan_id", None),
        "mode": _enum_value(getattr(plan, "mode", None)),
        "tactic": _enum_value(getattr(plan, "tactic", None)),
        "roles": {
            key: _enum_value(value)
            for key, value in sorted((getattr(plan, "roles", {}) or {}).items())
        },
        "target_assignments": dict(sorted((getattr(plan, "target_assignments", {}) or {}).items())),
        "primary_target": getattr(plan, "primary_target", None),
        "valid_for_steps": getattr(plan, "valid_for_steps", None),
        "source": _enum_value(getattr(plan, "source", None)),
        "rationale": list(getattr(plan, "rationale", ()) or ()),
    }


def score_summary(score):
    if score is None:
        return None
    return {
        "valid": bool(getattr(score, "valid", False)),
        "final_score": _finite_or_none(getattr(score, "final_score", None)),
        "expected_utility": _finite_or_none(getattr(score, "expected_utility", None)),
        "worst_case_utility": _finite_or_none(getattr(score, "worst_case_utility", None)),
        "switch_cost": _finite_or_none(getattr(score, "switch_cost", None)),
        "summary": getattr(score, "summary", None),
    }


def _ensure_file():
    global _file, _file_key
    side = _safe_name(_context.get("side") or "unknown")
    key = (_context.get("episode_id") or "episode", side)
    if _file is not None and _file_key == key:
        return _file
    close()
    try:
        _BASE_DIR.mkdir(parents=True, exist_ok=True)
        path = _BASE_DIR / f"{key[0]}_{key[1]}.jsonl"
        _file = path.open("a", encoding="utf-8")
        _file_key = key
        return _file
    except Exception as error:
        _remember_error(error)
        _file = None
        _file_key = None
        return None


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    enum_value = _enum_value(value)
    if enum_value is not value:
        return enum_value
    return str(value)


def _enum_value(value):
    return getattr(value, "value", value)


def _finite_or_none(value):
    try:
        value = float(value)
        if value in (float("inf"), float("-inf")):
            return None
        return value
    except Exception:
        return None


def _episode_id(context):
    for name in ("episode_id", "scenario_id", "run_id"):
        value = getattr(context, name, None)
        if value is not None:
            return str(value)
    return "episode"


def _safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:80]
    return value or "episode"


def _remember_error(error):
    _errors.append(f"{type(error).__name__}: {error}"[:500])
    del _errors[:-5]
