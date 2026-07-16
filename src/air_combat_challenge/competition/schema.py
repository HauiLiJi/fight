import json
from pathlib import Path

from .models import (
    ActionBatchV1,
    ActionReportV1,
    EpisodeStartV1,
    ObservationV1,
    StepResultV1,
)


SCHEMA_MODELS = {
    "observation-v1.json": ObservationV1,
    "action-batch-v1.json": ActionBatchV1,
    "action-report-v1.json": ActionReportV1,
    "episode-start-v1.json": EpisodeStartV1,
    "step-result-v1.json": StepResultV1,
}


def generate_schemas(output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, model in SCHEMA_MODELS.items():
        path = output_dir / filename
        path.write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths
