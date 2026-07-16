"""Summarise a candidate's side-balanced tournament results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from lab import PROJECT_ROOT, read_jsonl, wilson_interval, write_json


def score(rows):
    wins = sum(row["candidate_outcome"] == "win" for row in rows)
    draws = sum(row["candidate_outcome"] == "draw" for row in rows)
    losses = sum(row["candidate_outcome"] == "loss" for row in rows)
    total = len(rows)
    low, high = wilson_interval(wins, total)
    return {
        "episodes": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total if total else 0.0,
        "non_loss_rate": (wins + draws) / total if total else 0.0,
        "win_rate_95ci": [low, high],
        "mean_candidate_rejected_actions": (
            sum(row["rejected_actions"].get(row.get("candidate_side"), 0) for row in rows) / total
            if total
            else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=PROJECT_ROOT / "data" / "metrics" / "episodes.jsonl")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in read_jsonl(args.metrics) if row.get("candidate_name") == args.candidate]
    if not rows:
        parser.error(f"No metrics found for {args.candidate}")
    by_opponent = defaultdict(list)
    by_side = defaultdict(list)
    for row in rows:
        by_opponent[row.get("opponent_name", "unknown")].append(row)
        by_side[row.get("candidate_side", "unknown")].append(row)
    report = {
        "candidate": args.candidate,
        "overall": score(rows),
        "by_opponent": {name: score(group) for name, group in sorted(by_opponent.items())},
        "by_side": {name: score(group) for name, group in sorted(by_side.items())},
        "promotion_rule": "Promote only after reviewing held-out, side-balanced matches; never promote from this summary alone.",
    }
    write_json(args.output, report)
    print(f"Wrote tournament report to {args.output}")


if __name__ == "__main__":
    main()
