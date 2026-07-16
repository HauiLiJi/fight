from dataclasses import dataclass
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import keyword
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
import uuid


@dataclass(frozen=True)
class AgentSourceTree:
    root: Path
    entrypoint: Path
    python_files: tuple[Path, ...]
    source_hash: str
    package_name: str
    module_name: str


def inspect_agent_source_tree(manifest_path, entrypoint):
    manifest_path = Path(manifest_path).resolve()
    source_root = manifest_path.parent
    entrypoint_path = _resolve_entrypoint(source_root, entrypoint)
    module_parts = _entrypoint_module_parts(entrypoint_path.relative_to(source_root))
    python_files = _collect_python_files(source_root)
    source_hash = _hash_and_compile(source_root, python_files)
    package_name = f"_air_combat_submission_{source_hash[:16]}_{uuid.uuid4().hex[:12]}"
    module_name = ".".join((package_name, *module_parts))
    return AgentSourceTree(
        root=source_root,
        entrypoint=entrypoint_path,
        python_files=python_files,
        source_hash=source_hash,
        package_name=package_name,
        module_name=module_name,
    )


def load_agent_module(source_tree):
    importlib.invalidate_caches()
    try:
        if source_tree.module_name == source_tree.package_name:
            return _load_root_package(source_tree)
        _register_namespace_package(source_tree)
        return importlib.import_module(source_tree.module_name)
    except BaseException:
        remove_agent_modules(source_tree.package_name)
        raise


def remove_agent_modules(package_name):
    prefix = f"{package_name}."
    for module_name in tuple(sys.modules):
        if module_name == package_name or module_name.startswith(prefix):
            sys.modules.pop(module_name, None)


def _resolve_entrypoint(source_root, entrypoint):
    raw_path = Path(entrypoint)
    if raw_path.is_absolute():
        raise ValueError("agent entrypoint must be a relative path")
    if raw_path.suffix.lower() != ".py":
        raise ValueError("agent entrypoint must be a Python file path ending in .py")

    entrypoint_path = (source_root / raw_path).resolve()
    try:
        relative_path = entrypoint_path.relative_to(source_root)
    except ValueError as error:
        raise ValueError("agent entrypoint must stay inside submission root") from error
    if "__pycache__" in relative_path.parts:
        raise ValueError("agent entrypoint cannot be inside __pycache__")
    if not entrypoint_path.is_file():
        raise FileNotFoundError(f"agent source does not exist: {entrypoint_path}")
    return entrypoint_path


def _entrypoint_module_parts(relative_path):
    if relative_path.name == "__init__.py":
        parts = relative_path.parent.parts
    else:
        parts = (*relative_path.parent.parts, relative_path.stem)
    if any(not part.isidentifier() or keyword.iskeyword(part) for part in parts):
        raise ValueError(
            "agent entrypoint must map to a valid Python module path: "
            f"{relative_path.as_posix()}"
        )
    return parts


def _collect_python_files(source_root):
    python_files = []
    for current_root, directory_names, file_names in os.walk(
        source_root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current_root)
        retained_directories = []
        for directory_name in sorted(directory_names):
            directory_path = current_path / directory_name
            relative_path = directory_path.relative_to(source_root)
            if "__pycache__" in relative_path.parts:
                continue
            resolved_path = directory_path.resolve()
            try:
                resolved_path.relative_to(source_root)
            except ValueError as error:
                raise ValueError(
                    "agent Python source directory must stay inside submission root: "
                    f"{relative_path.as_posix()}"
                ) from error
            if _is_linked_directory(directory_path):
                raise ValueError(
                    "linked agent source directories are not supported: "
                    f"{relative_path.as_posix()}"
                )
            retained_directories.append(directory_name)
        directory_names[:] = retained_directories

        for file_name in sorted(file_names):
            if Path(file_name).suffix.lower() != ".py":
                continue
            candidate = current_path / file_name
            relative_path = candidate.relative_to(source_root)
            resolved_path = candidate.resolve()
            try:
                resolved_path.relative_to(source_root)
            except ValueError as error:
                raise ValueError(
                    "agent Python source must stay inside submission root: "
                    f"{relative_path.as_posix()}"
                ) from error
            if candidate.is_file():
                python_files.append(candidate)
    return tuple(
        sorted(
            python_files,
            key=lambda path: path.relative_to(source_root).as_posix(),
        )
    )


def _is_linked_directory(path):
    if path.is_symlink():
        return True
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def _hash_and_compile(source_root, python_files):
    digest = hashlib.sha256(b"air-combat-agent-source-v1\0")
    for source_path in python_files:
        relative_path = source_path.relative_to(source_root).as_posix()
        relative_bytes = relative_path.encode("utf-8")
        source_bytes = source_path.read_bytes()
        compile(source_bytes, str(source_path), "exec")
        digest.update(len(relative_bytes).to_bytes(4, "big"))
        digest.update(relative_bytes)
        digest.update(len(source_bytes).to_bytes(8, "big"))
        digest.update(source_bytes)
    return digest.hexdigest()


def _register_namespace_package(source_tree):
    package = ModuleType(source_tree.package_name)
    package.__package__ = source_tree.package_name
    package.__path__ = [str(source_tree.root)]
    package_spec = ModuleSpec(source_tree.package_name, loader=None, is_package=True)
    package_spec.submodule_search_locations = [str(source_tree.root)]
    package.__spec__ = package_spec
    sys.modules[source_tree.package_name] = package


def _load_root_package(source_tree):
    spec = importlib.util.spec_from_file_location(
        source_tree.package_name,
        source_tree.entrypoint,
        submodule_search_locations=[str(source_tree.root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent module: {source_tree.entrypoint}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[source_tree.package_name] = module
    spec.loader.exec_module(module)
    return module
