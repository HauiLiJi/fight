# LJH Offline Evolution Lab

This toolchain keeps `examples/rules/a2a_rule_ljh.py` immutable as `ljh_v1`.
All raw simulator outputs, candidate snapshots, and experiment reports are new
files. The tools use only the Python standard library.

## 1. Collect paired matches

Start AFSIM first, then run:

```powershell
uv run python experiments/collect_matches.py --config experiments/configs/ljh_v1_vs_baseline.json
```

Every seed is played twice with the candidate on blue and red. Raw replays are
written to `data/raw_replays/<run-id>/`; the corresponding match index records
the candidate side and opponent version.

## 2. Extract durable metrics and build the local vector index

```powershell
uv run python experiments/extract_features.py `
  --replays data/raw_replays/<run-id> `
  --match-index data/match_indexes/<run-id>.jsonl

uv run python experiments/build_vector_index.py `
  --metrics data/metrics/episodes.jsonl
```

The vector index is a deterministic hashed text embedding intended for local
prototyping. Raw replay JSONL and metrics JSONL remain the source of truth. It
can later be replaced with Chroma, FAISS, or a hosted vector database.

## 3. Retrieve evidence for an LLM analyst

```powershell
uv run python experiments/prepare_llm_brief.py `
  --question "Why does ljh_v1 lose 2v2 engagements after the first exchange?" `
  --output data/reports/ljh_v1_failure_brief.json
```

Give the produced JSON to an LLM. Its response must conform to the included
schema: it can cite evidence and suggest only whitelisted numeric parameters.
It must not edit Python source.

## 4. Create and test a candidate

Save the reviewed LLM response as `data/proposals/ljh_v2_candidate_001.json`,
then create an immutable source snapshot:

```powershell
uv run python experiments/make_candidate.py `
  --proposal data/proposals/ljh_v2_candidate_001.json `
  --name ljh_v2_candidate_001
```

Point a new match config at
`agents/ljh_candidates/ljh_v2_candidate_001/agent.json`, collect paired
matches, and evaluate them:

```powershell
uv run python experiments/evaluate_tournament.py `
  --candidate ljh_v2_candidate_001 `
  --output data/reports/ljh_v2_candidate_001.json
```

Promotion copies a candidate into `agents/ljh_champions/` only when it has at
least 20 side-balanced matches and clears the requested win-rate threshold.
The report should also be reviewed against a held-out seed set and opponent
pool before promotion.
