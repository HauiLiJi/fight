"""Run paired, seed-controlled matches and write a reproducible match index."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from lab import PROJECT_ROOT, read_json, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_json(args.config)
    candidate = config["candidate"]
    opponents = config["opponents"]
    output_dir = PROJECT_ROOT / "data" / "raw_replays" / args.run_id
    index_path = PROJECT_ROOT / "data" / "match_indexes" / f"{args.run_id}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_records = []

    for opponent in opponents:
        for seed in config["seeds"]:
            for candidate_side in ("blue", "red"):
                manifests = {
                    candidate_side: candidate["manifest"],
                    "red" if candidate_side == "blue" else "blue": opponent["manifest"],
                }
                command = [
                    "uv", "run", "air-combat", "run",
                    "--blue-agent", manifests["blue"],
                    "--red-agent", manifests["red"],
                    "--scenario", config["scenario"],
                    "--ip", str(config.get("afsim_ip", "127.0.0.1")),
                    "--port", str(config.get("afsim_port", 19920)),
                    "--seed", str(seed),
                    "--steps", str(config.get("steps", 2000)),
                    "--output", str(output_dir),
                ]
                if config.get("global_view", False):
                    command.append("--global-view")
                print(" ".join(command))
                if args.dry_run:
                    continue
                before = {path.name for path in output_dir.glob("*.summary.json")}
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                summaries = [path for path in output_dir.glob("*.summary.json") if path.name not in before]
                if len(summaries) != 1:
                    raise RuntimeError(f"Expected one new summary for seed {seed}; found {len(summaries)}")
                summary = read_json(summaries[0])
                index_records.append(
                    {
                        "episode_id": summary["episode_id"],
                        "run_id": args.run_id,
                        "seed": seed,
                        "candidate_name": candidate["name"],
                        "opponent_name": opponent["name"],
                        "candidate_side": candidate_side,
                        "manifest_paths": manifests,
                        "scenario": config["scenario"],
                    }
                )
                write_jsonl(index_path, index_records)

    if args.dry_run:
        return
    print(f"Wrote {len(index_records)} paired match records to {index_path}")


if __name__ == "__main__":
    main()
