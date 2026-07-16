import hashlib
import json
from pathlib import Path


def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReplayWriter:
    def __init__(self, output_dir, episode_id):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.replay_path = self.output_dir / f"{episode_id}.jsonl"

    def write(self, record_type, payload):
        record = {"record_type": record_type, "payload": payload}
        with self.replay_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")

    def write_summary(self, summary):
        summary_path = self.output_dir / f"{summary['episode_id']}.summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary_path
