"""Convert raw replay JSONL files to durable episode metrics and case text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lab import PROJECT_ROOT, extract_episode, index_by_episode, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", required=True, type=Path)
    parser.add_argument("--match-index", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "metrics" / "episodes.jsonl")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping incomplete replays")
    args = parser.parse_args()
    index = index_by_episode(args.match_index)
    rows = []
    skipped = 0
    for replay_path in sorted(args.replays.glob("*.jsonl")):
        episode_id = replay_path.stem
        try:
            rows.append(extract_episode(replay_path, index.get(episode_id)))
        except ValueError as error:
            if args.strict:
                raise
            skipped += 1
            print(f"Skipping {replay_path}: {error}", file=sys.stderr)
    write_jsonl(args.output, rows)
    print(f"Extracted {len(rows)} episodes to {args.output}; skipped {skipped} incomplete replays")


if __name__ == "__main__":
    main()
