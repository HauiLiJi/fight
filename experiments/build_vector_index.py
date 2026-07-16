"""Build and query a small local vector store for semantic case retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from lab import PROJECT_ROOT, cosine, hashed_embedding, read_json, read_jsonl, write_json


def build(metrics_path: Path, output_path: Path, dimensions: int):
    cases = read_jsonl(metrics_path)
    index = {
        "format": "air-combat-local-vector-index-v1",
        "dimensions": dimensions,
        "cases": [
            {
                "episode_id": item["episode_id"],
                "text": item["case_text"],
                "metadata": {key: value for key, value in item.items() if key not in {"case_text", "replay_path"}},
                "vector": hashed_embedding(item["case_text"], dimensions),
            }
            for item in cases
        ],
    }
    write_json(output_path, index)
    print(f"Indexed {len(index['cases'])} cases in {output_path}")


def query(index_path: Path, text: str, limit: int):
    index = read_json(index_path)
    query_vector = hashed_embedding(text, index["dimensions"])
    ranked = sorted(
        (
            {"score": round(cosine(query_vector, item["vector"]), 4), "episode_id": item["episode_id"], "text": item["text"], "metadata": item["metadata"]}
            for item in index["cases"]
        ),
        key=lambda item: item["score"],
        reverse=True,
    )[:limit]
    for item in ranked:
        print(f"{item['score']:>6} {item['episode_id']} {item['text']}")
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "vector_index" / "cases.json")
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--query")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.metrics:
        build(args.metrics, args.output, args.dimensions)
    if args.query:
        query(args.output, args.query, args.limit)
    elif not args.metrics:
        parser.error("Provide --metrics to build or --query to search an existing index")


if __name__ == "__main__":
    main()
