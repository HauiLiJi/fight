from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Type

from .agent_source import (
    inspect_agent_source_tree,
    load_agent_module,
    remove_agent_modules,
)
from .models import ActionBatchV1, AgentManifestV1, EpisodeContextV1, ObservationV1


class BaseAgent(ABC):
    def reset(self, context: EpisodeContextV1) -> None:
        self.context = context

    @abstractmethod
    def act(self, observation: ObservationV1) -> ActionBatchV1:
        raise NotImplementedError

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class LoadedAgent:
    manifest: AgentManifestV1
    manifest_path: Path
    source_root: Path
    source_path: Path
    source_hash: str
    agent_class: Type[BaseAgent]


def load_agent_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = AgentManifestV1.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    source_tree = inspect_agent_source_tree(manifest_path, manifest.entrypoint)
    module = load_agent_module(source_tree)

    candidates = {}
    for value in module.__dict__.values():
        if (
            isinstance(value, type)
            and value is not BaseAgent
            and issubclass(value, BaseAgent)
            and value.__module__ == module.__name__
        ):
            candidates.setdefault(id(value), value)
    if len(candidates) != 1:
        remove_agent_modules(source_tree.package_name)
        names = ", ".join(
            sorted(candidate.__name__ for candidate in candidates.values())
        )
        raise TypeError(
            "agent source must define exactly one BaseAgent subclass; "
            f"found {len(candidates)}" + (f": {names}" if names else "")
        )
    agent_class = next(iter(candidates.values()))
    return LoadedAgent(
        manifest=manifest,
        manifest_path=manifest_path,
        source_root=source_tree.root,
        source_path=source_tree.entrypoint,
        source_hash=source_tree.source_hash,
        agent_class=agent_class,
    )


def validate_agent_manifest(manifest_path):
    loaded = load_agent_manifest(manifest_path)
    instance = loaded.agent_class()
    if not isinstance(instance, BaseAgent):
        raise TypeError("agent constructor must return a BaseAgent instance")
    context = EpisodeContextV1(
        episode_id="validation",
        side="blue",
        controlled_platform_ids=["blue_fighter_01"],
        seed=0,
    )
    observation = ObservationV1(
        episode_id="validation",
        step_index=0,
        sim_time=0.0,
        side="blue",
        controlled_platform_ids=["blue_fighter_01"],
        own_units=[],
        tracks=[],
    )
    try:
        instance.reset(context)
        result = instance.act(observation)
        if not isinstance(result, ActionBatchV1):
            ActionBatchV1.model_validate(result)
    finally:
        instance.close()
    return {
        "valid": True,
        "api_version": loaded.manifest.api_version,
        "topology": loaded.manifest.topology,
        "entrypoint": loaded.manifest.entrypoint,
        "source_hash": loaded.source_hash,
    }
