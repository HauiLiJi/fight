"""Create an immutable LJH parameter candidate from a reviewed JSON proposal."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from lab import PROJECT_ROOT, read_json, write_json


PARAMETERS = {
    "medium_fire_range_m": ("MEDIUM_FIRE_RANGE_M", 30_000, 60_000),
    "medium_min_range_m": ("MEDIUM_MIN_RANGE_M", 10_000, 30_000),
    "short_fire_range_m": ("SHORT_FIRE_RANGE_M", 5_000, 20_000),
    "medium_launch_cooldown_s": ("MEDIUM_LAUNCH_COOLDOWN_S", 8, 60),
    "short_launch_cooldown_s": ("SHORT_LAUNCH_COOLDOWN_S", 4, 30),
    "target_memory_s": ("TARGET_MEMORY_S", 0, 120),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "agents" / "ljh_candidates")
    args = parser.parse_args()
    if not re.fullmatch(r"ljh_v[0-9]+_candidate_[0-9]{3}", args.name):
        parser.error("--name must look like ljh_v2_candidate_001")
    proposal = read_json(args.proposal)
    changes = proposal.get("parameter_changes") or {}
    if not changes:
        parser.error("proposal must include parameter_changes")
    unknown = set(changes) - set(PARAMETERS)
    if unknown:
        parser.error(f"unsupported parameters: {sorted(unknown)}")

    source = (PROJECT_ROOT / "examples" / "rules" / "a2a_rule_ljh.py").read_text(encoding="utf-8")
    for parameter, value in changes.items():
        constant, minimum, maximum = PARAMETERS[parameter]
        if not isinstance(value, (int, float)) or not minimum <= value <= maximum:
            parser.error(f"{parameter} must be within [{minimum}, {maximum}]")
        replacement = f"{constant} = {value!r}"
        source, replacements = re.subn(rf"^{constant} = .+$", replacement, source, flags=re.MULTILINE)
        if replacements != 1:
            raise RuntimeError(f"Unable to update {constant} in LJH baseline")
    medium_minimum = changes.get("medium_min_range_m", 18_000)
    medium_maximum = changes.get("medium_fire_range_m", 48_000)
    if medium_minimum >= medium_maximum:
        parser.error("medium_min_range_m must remain below medium_fire_range_m")

    target = args.output_root / args.name
    if target.exists():
        raise FileExistsError(f"candidate already exists: {target}")
    target.mkdir(parents=True)
    (target / "a2a_rule_ljh.py").write_text(source, encoding="utf-8")
    write_json(
        target / "agent.json",
        {"api_version": "1.0", "topology": "team", "entrypoint": "a2a_rule_ljh.py", "step_timeout_s": 5.0},
    )
    write_json(target / "proposal.json", proposal)
    print(target / "agent.json")


if __name__ == "__main__":
    main()
