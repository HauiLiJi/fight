"""Shared, dependency-free utilities for the LJH offline evolution workflow."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}") from error
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def index_by_episode(match_index_path: Path | None) -> dict[str, dict[str, Any]]:
    if match_index_path is None or not match_index_path.is_file():
        return {}
    return {item["episode_id"]: item for item in read_jsonl(match_index_path)}


def _fire_actions(submitted_actions: dict[str, Any]) -> dict[str, int]:
    counts = {"blue": 0, "red": 0}
    for side, batches in submitted_actions.items():
        for item in batches:
            actions = (item.get("batch") or {}).get("actions") or []
            counts[side] = counts.get(side, 0) + sum(
                action.get("type") in {"fire", "co_fire"} for action in actions
            )
    return counts


def extract_episode(replay_path: Path, match_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract stable aggregate features without assuming simulator-specific events."""
    records = read_jsonl(replay_path)
    start = next((record["payload"] for record in records if record["record_type"] == "episode_start"), None)
    summary = next((record["payload"] for record in reversed(records) if record["record_type"] == "summary"), None)
    if start is None or summary is None:
        raise ValueError(f"Replay lacks episode_start or summary: {replay_path}")

    episode = start["episode"]
    observations = episode["observations"]
    initial_units = {
        side: len(observations.get(side, {}).get("own_units") or []) for side in ("blue", "red")
    }
    first_detection = {
        side: (float(observations.get(side, {}).get("sim_time", 0.0)) if observations.get(side, {}).get("tracks") else None)
        for side in ("blue", "red")
    }
    first_fire = {"blue": None, "red": None}
    fired = {"blue": 0, "red": 0}
    rejected = {"blue": 0, "red": 0}
    event_types: dict[str, int] = {}
    final_units = dict(initial_units)
    last_sim_time = float(observations.get("blue", {}).get("sim_time", 0.0))

    for record in records:
        if record["record_type"] != "step":
            continue
        payload = record["payload"]
        result = payload.get("result") or {}
        sim_time = float(result.get("sim_time", last_sim_time))
        last_sim_time = sim_time
        step_fires = _fire_actions(payload.get("submitted_actions") or {})
        for side in ("blue", "red"):
            fired[side] += step_fires.get(side, 0)
            if step_fires.get(side, 0) and first_fire[side] is None:
                first_fire[side] = sim_time
            side_observation = (result.get("observations") or {}).get(side) or {}
            if side_observation.get("tracks") and first_detection[side] is None:
                first_detection[side] = sim_time
            if side_observation:
                final_units[side] = len(side_observation.get("own_units") or [])
            reports = (result.get("action_reports") or {}).get(side) or []
            rejected[side] += sum(report.get("status") != "accepted" for report in reports)
        for event in result.get("events") or []:
            event_type = str(event.get("event_type", "unknown"))
            event_types[event_type] = event_types.get(event_type, 0) + 1

    metadata = match_metadata or {}
    candidate_side = metadata.get("candidate_side")
    winner = summary.get("winner")
    candidate_outcome = None
    if candidate_side:
        candidate_outcome = "draw" if winner == "draw" else ("win" if winner == candidate_side else "loss")

    item = {
        "episode_id": summary["episode_id"],
        "replay_path": str(replay_path.resolve()),
        "seed": summary.get("seed"),
        "winner": winner,
        "reason": summary.get("reason"),
        "executed_steps": summary.get("executed_steps"),
        "duration_s": last_sim_time,
        "scenario_hash": summary.get("scenario_hash"),
        "agent_hashes": summary.get("agent_hashes", {}),
        "initial_units": initial_units,
        "final_units": final_units,
        "first_detection_s": first_detection,
        "first_fire_s": first_fire,
        "fire_actions": fired,
        "rejected_actions": rejected,
        "event_types": event_types,
        "candidate_name": metadata.get("candidate_name"),
        "opponent_name": metadata.get("opponent_name"),
        "candidate_side": candidate_side,
        "candidate_outcome": candidate_outcome,
        "manifest_paths": metadata.get("manifest_paths", {}),
    }
    item["tags"] = tags_for_episode(item)
    item["case_text"] = case_text(item)
    return item


def tags_for_episode(item: dict[str, Any]) -> list[str]:
    tags = [str(item.get("reason") or "unknown_end")]
    candidate_side = item.get("candidate_side")
    if candidate_side:
        tags.append(f"candidate_{item.get('candidate_outcome')}")
        opponent_side = "red" if candidate_side == "blue" else "blue"
        if item["final_units"].get(candidate_side, 0) < item["initial_units"].get(candidate_side, 0):
            tags.append("candidate_losses")
        if item["fire_actions"].get(candidate_side, 0) == 0:
            tags.append("candidate_no_fire")
        if item["first_fire_s"].get(candidate_side) is not None:
            their_fire = item["first_fire_s"].get(opponent_side)
            if their_fire is not None and item["first_fire_s"][candidate_side] > their_fire:
                tags.append("candidate_late_first_fire")
    if any(item["rejected_actions"].values()):
        tags.append("action_rejections")
    return sorted(set(tags))


def case_text(item: dict[str, Any]) -> str:
    candidate_side = item.get("candidate_side") or "unknown"
    opponent_side = "red" if candidate_side == "blue" else "blue"
    return (
        f"candidate={item.get('candidate_name') or 'unknown'} opponent={item.get('opponent_name') or 'unknown'} "
        f"candidate_side={candidate_side} outcome={item.get('candidate_outcome') or item.get('winner')} "
        f"reason={item.get('reason')} seed={item.get('seed')} steps={item.get('executed_steps')} "
        f"initial_units={item['initial_units']} final_units={item['final_units']} "
        f"candidate_first_detection={item['first_detection_s'].get(candidate_side)} "
        f"candidate_first_fire={item['first_fire_s'].get(candidate_side)} "
        f"opponent_first_fire={item['first_fire_s'].get(opponent_side)} "
        f"candidate_fires={item['fire_actions'].get(candidate_side, 0)} "
        f"candidate_rejections={item['rejected_actions'].get(candidate_side, 0)} "
        f"tags={' '.join(item['tags'])}"
    )


def hashed_embedding(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    for token in TOKEN_PATTERN.findall(text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        slot = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[slot] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = wins / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)
