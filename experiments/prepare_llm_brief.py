"""Prepare evidence-grounded context for an LLM without granting code-write access."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_vector_index import query
from lab import PROJECT_ROOT, write_json


ALLOWED_PARAMETERS = {
    "medium_fire_range_m": [30_000, 60_000],
    "medium_min_range_m": [10_000, 30_000],
    "short_fire_range_m": [5_000, 20_000],
    "medium_launch_cooldown_s": [8, 60],
    "short_launch_cooldown_s": [4, 30],
    "target_memory_s": [0, 120],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=PROJECT_ROOT / "data" / "vector_index" / "cases.json")
    parser.add_argument("--question", required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = query(args.index, args.question, args.limit)
    write_json(
        args.output,
        {
            "role": "You are an offline air-combat experiment analyst. Do not propose Python code.",
            "question": args.question,
            "evidence": evidence,
            "allowed_parameter_ranges": ALLOWED_PARAMETERS,
            "required_response_schema": {
                "hypothesis": "string",
                "evidence_episode_ids": ["episode id"],
                "parameter_changes": {"medium_fire_range_m": 50000},
                "expected_benefit": "string",
                "risk": "string",
            },
        },
    )
    print(f"Wrote LLM brief to {args.output}")


if __name__ == "__main__":
    main()
