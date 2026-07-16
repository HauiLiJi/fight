"""Copy a validated candidate to the immutable champion registry."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lab import PROJECT_ROOT, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--champion-name", required=True)
    parser.add_argument("--minimum-win-rate", type=float, default=0.55)
    args = parser.parse_args()
    evaluation = read_json(args.evaluation)
    overall = evaluation["overall"]
    if overall["episodes"] < 20:
        parser.error("Need at least 20 side-balanced episodes before promotion")
    if overall["win_rate"] < args.minimum_win_rate:
        parser.error("Candidate does not meet the requested win-rate threshold")
    destination = PROJECT_ROOT / "agents" / "ljh_champions" / args.champion_name
    if destination.exists():
        raise FileExistsError(f"Champion already exists: {destination}")
    shutil.copytree(args.candidate_dir, destination)
    registry_path = PROJECT_ROOT / "data" / "registry" / "champions.json"
    registry = read_json(registry_path) if registry_path.is_file() else {"champions": []}
    registry["champions"].append(
        {
            "name": args.champion_name,
            "source": str(args.candidate_dir),
            "manifest": str(destination / "agent.json"),
            "evaluation": str(args.evaluation),
            "overall": overall,
        }
    )
    write_json(registry_path, registry)
    print(destination / "agent.json")


if __name__ == "__main__":
    main()
