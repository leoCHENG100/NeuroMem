from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

from neuromem_core import BudgetedSupportPathSelector, LifecyclePolicy, entry_text, successor_trace_attribution

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def model_directory_manifest(path: str | Path) -> Dict:
    """Hash every local model artifact so a frozen encoder is identifiable."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Local semantic model directory is missing: {root}")
    files = []
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append({
            "path": str(file_path.relative_to(root)),
            "bytes": file_path.stat().st_size,
            "sha256": sha256_file(file_path),
        })
    if not files:
        raise ValueError(f"Local semantic model directory contains no files: {root}")
    return {
        "model_id": "BAAI/bge-m3",
        "local_path": str(root),
        "files": files,
        "file_count": len(files),
        "directory_manifest_sha256": sha256_json(files),
    }


def write_json(path: str | Path, payload) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def append_jsonl(path: str | Path, payload) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def memory_size_stats(memory: Sequence[Mapping]) -> Dict:
    rendered = [entry_text(item) for item in memory]
    return {
        "entries": len(rendered),
        "chars": sum(len(text) for text in rendered),
        "utf8_bytes": sum(len(text.encode("utf-8")) for text in rendered),
        "max_entry_chars": max((len(text) for text in rendered), default=0),
    }


def active_memory(persistent_memory: Sequence[Mapping], policy: LifecyclePolicy) -> list[Dict]:
    """The only retrievable set: M_t union threshold-admissible B_t."""

    return [dict(item) for item in persistent_memory] + policy.pending_as_memory()


def select_path_and_credit(
    *,
    query: str,
    memory: Sequence[Mapping],
    selector: BudgetedSupportPathSelector,
    credit_mode: str,
) -> tuple[list[int], list[Mapping], Dict[str, float], Dict[str, float]]:
    selected_indexes = selector.select(query, memory)
    selected = [memory[index] for index in selected_indexes]
    by_id = {str(item["mem_id"]): item for item in memory}
    if credit_mode == "prefix":
        path_credit = dict(selector.last_path_details.get("attribution", {}))
    elif credit_mode == "similarity":
        path_credit = selector.similarity_credit(query, selector.last_selected_ids, by_id)
    else:
        raise ValueError("credit_mode must be prefix or similarity")
    successor_credit = successor_trace_attribution(memory, path_credit)
    return selected_indexes, selected, path_credit, successor_credit


class FixedDemonstrationSelector:
    """Select fixed benchmark demonstrations in their supplied order."""

    strategy = "fixed_benchmark_demonstrations"

    def __init__(self) -> None:
        self.last_selected_indexes: list[int] = []

    def select(self, query: str, memory: Sequence[Mapping], num_shots: int) -> list[int]:
        del query
        count = min(max(0, int(num_shots)), len(memory))
        self.last_selected_indexes = list(range(count))
        return list(self.last_selected_indexes)

    def sync_memory_size(self, memory_size: int) -> None:
        del memory_size

    def state_dict(self) -> Dict:
        return {"schema": "fixed-demonstration-selection/1"}
