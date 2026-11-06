"""NeuroMem retrieval, delayed credit, and memory lifecycle primitives.

The framework combines frozen BGE-M3 embeddings, deterministic budgeted
support-path search, and a four-action softmax policy optimized online with
Adam. Its baseline is the cumulative mean of finalized regularized returns.
Action-transition penalties account for memory growth and redundancy.

Candidate utility starts from task feedback and accumulates attributed future
reuse evidence while the candidate remains in the buffer. Promotion uses
confidence and utility gates. MERGE requires immutable anchor-key overlap and
non-duplicate content. Redundancy is mean positive nearest-neighbour cosine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
import hashlib
import heapq
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Mapping, MutableSequence, Protocol, Sequence, Tuple

import numpy as np


DEFAULT_SEMANTIC_MODEL_PATH = "models/bge-m3"
DEFAULT_MERGE_MAX_CHARS = 8192
MERGE_GATE_NAME = "immutable_anchor_key_overlap_and_non_noop"

ACTIONS: Tuple[str, ...] = ("PROMOTE", "MERGE", "RETAIN", "DISCARD")
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTIONS)}

TRACE_TEXT_FIELDS: Tuple[str, ...] = (
    "knowledge",
    "applicability",
    "reasoning_pattern",
    "composition_keys",
    "evidence",
    "cautions",
    "confidence",
)

MERGE_TEXT_FIELDS: Tuple[str, ...] = (
    "knowledge",
    "applicability",
    "reasoning_pattern",
    "evidence",
    "cautions",
)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _softmax(logits: np.ndarray, allowed: Sequence[bool]) -> np.ndarray:
    if len(logits) != len(allowed):
        raise ValueError("logits and allowed-action mask must have equal length")
    masked = np.full(len(logits), -np.inf, dtype=float)
    for index, is_allowed in enumerate(allowed):
        if is_allowed:
            masked[index] = float(logits[index])
    finite = np.isfinite(masked)
    if not finite.any():
        raise ValueError("At least one lifecycle action must be admissible")
    maximum = float(np.max(masked[finite]))
    weights = np.zeros_like(masked)
    weights[finite] = np.exp(masked[finite] - maximum)
    return weights / float(weights.sum())


def _json_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def entry_text(entry: Mapping) -> str:
    """Canonical rendering for TTK traces and initial raw evidence.

    Transferable traces use the seven fields in TRACE_TEXT_FIELDS. When none
    are present, question/code evidence can supply a fallback representation.
    Identity, provenance, utility, and reference-answer fields are excluded.
    """

    parts: List[str] = []
    for key in TRACE_TEXT_FIELDS:
        value = entry.get(key)
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key}: {_json_text(value)}")
    if not parts:
        for key in ("question", "knowledge", "code"):
            value = entry.get(key)
            if value not in (None, "", [], {}):
                parts.append(f"{key}: {_json_text(value)}")
    return "\n".join(parts)


def confidence_value(value) -> float:
    if isinstance(value, (int, float)):
        return _clip01(float(value))
    normalized = str(value or "").strip().casefold()
    mapping = {
        "very low": 0.10,
        "low": 0.25,
        "medium": 0.50,
        "moderate": 0.50,
        "high": 0.90,
        "very high": 0.98,
    }
    return mapping.get(normalized, 0.50)


def confidence_label(value) -> str:
    """Normalize qualitative confidence for transferable traces.

    Numeric inputs are accepted when normalizing records and unit fixtures,
    but are projected to the nearest declared low/medium/high anchor.  The
    stored transferable trace itself always keeps the qualitative label;
    lifecycle code derives a separate numeric promotion score only where a
    numeric gate is needed.
    """

    normalized = str(value or "").strip().casefold()
    if normalized in {"low", "medium", "high"}:
        return normalized
    numeric = confidence_value(value)
    anchors = (("low", 0.25), ("medium", 0.50), ("high", 0.90))
    return min(anchors, key=lambda item: (abs(numeric - item[1]), item[1]))[0]


def normalize_transferable_trace(payload: Mapping) -> Dict:
    """Validate and normalize a transferable-knowledge record."""

    required = (
        "knowledge",
        "applicability",
        "reasoning_pattern",
        "composition_keys",
        "evidence",
        "cautions",
        "confidence",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"TTK payload is missing required fields: {missing}")
    keys = payload.get("composition_keys")
    if not isinstance(keys, list):
        raise ValueError("TTK composition_keys must be a JSON list")
    result = {
        "knowledge": str(payload.get("knowledge", "")).strip(),
        "applicability": str(payload.get("applicability", "")).strip(),
        "reasoning_pattern": str(payload.get("reasoning_pattern", "")).strip(),
        "composition_keys": list(dict.fromkeys(str(item).strip() for item in keys if str(item).strip())),
        "evidence": str(payload.get("evidence", "")).strip(),
        "cautions": str(payload.get("cautions", "")).strip(),
        "confidence": confidence_label(payload.get("confidence")),
        "source": "neuromem_ttk",
    }
    if not result["knowledge"] or not result["reasoning_pattern"]:
        raise ValueError("TTK knowledge and reasoning_pattern must be non-empty")
    return result


def remove_instance_values(trace: Mapping, source_text: str) -> Dict:
    """Replace numeric instance values copied from a source task.

    Mask matching numeric tokens in generated trace fields while preserving
    alphabetic schema identifiers. This targeted filter is not comprehensive
    de-identification; trace text still requires clinical-data safeguards.
    """

    result = dict(trace)
    # Match complete numbers, excluding identifier suffixes and decimal parts.
    left_boundary = r"(?<![\w.])"
    right_boundary = r"(?!\w|\.\d)"
    source_numbers = set(re.findall(
        left_boundary + r"\d+(?:\.\d+)?" + right_boundary, str(source_text or "")
    ))
    if not source_numbers:
        return result
    pattern = re.compile(
        left_boundary + r"(?:"
        + "|".join(sorted((re.escape(item) for item in source_numbers), key=len, reverse=True))
        + r")" + right_boundary
    )
    for field_name in ("knowledge", "applicability", "reasoning_pattern", "evidence", "cautions"):
        result[field_name] = pattern.sub("<VALUE>", str(result.get(field_name, "")))
    cleaned_keys = []
    for item in result.get("composition_keys", []):
        cleaned = pattern.sub("<VALUE>", str(item))
        if cleaned not in cleaned_keys:
            cleaned_keys.append(cleaned)
    result["composition_keys"] = cleaned_keys
    return result


class RelationScorer(Protocol):
    @property
    def embedding_dimension(self) -> int: ...

    def encode(self, text: str) -> np.ndarray: ...

    def similarity(self, left: str, right: str) -> float: ...

    def metadata(self) -> Dict: ...


class HashRelationScorer:
    """Deterministic local test scorer with normalized hashed token vectors."""

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = max(8, int(dimension))

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=float)
        for token in re.findall(r"[A-Za-z0-9_]+", str(text or "").casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0.0 else vector

    def similarity(self, left: str, right: str) -> float:
        return float(np.dot(self.encode(left), self.encode(right)))

    def metadata(self) -> Dict:
        return {"kind": "hashed-token-cosine-test-only", "dimension": self._dimension, "transform": "raw_cosine"}


class LocalBGERelationScorer:
    """Frozen local BAAI/bge-m3 encoder using raw normalized cosine."""

    def __init__(
        self,
        model_path: str = DEFAULT_SEMANTIC_MODEL_PATH,
        device: str = "auto",
        max_seq_length: int = 8192,
        batch_size: int = 32,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Local BAAI/bge-m3 directory is missing: {path}")
        try:
            import sentence_transformers
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("BGE scoring requires sentence-transformers and torch") from error
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        self.model_path = str(path)
        self.device = str(resolved_device)
        self.max_seq_length = max(1, int(max_seq_length))
        self.batch_size = max(1, int(batch_size))
        self.sentence_transformers_version = str(sentence_transformers.__version__)
        self.torch_version = str(torch.__version__)
        self.model = SentenceTransformer(
            self.model_path,
            device=self.device,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model.max_seq_length = self.max_seq_length
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            dimension = len(self.model.encode(["dimension probe"], normalize_embeddings=True)[0])
        self._dimension = int(dimension)
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        text = str(text or "")
        cached = self._cache.get(text)
        if cached is not None:
            return cached.copy()
        vector = self.model.encode(
            [text],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        vector = np.asarray(vector, dtype=float)
        self._cache[text] = vector
        return vector.copy()

    def similarity(self, left: str, right: str) -> float:
        return float(np.dot(self.encode(left), self.encode(right)))

    def metadata(self) -> Dict:
        return {
            "kind": "frozen_bge_m3_raw_cosine",
            "model_id": "BAAI/bge-m3",
            "model_path": self.model_path,
            "device": self.device,
            "embedding_dimension": self.embedding_dimension,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
            "relation_transform": "raw_cosine",
            "sentence_transformers_version": self.sentence_transformers_version,
            "torch_version": self.torch_version,
        }


def build_relation_scorer(
    mode: str = "bge_m3",
    model_path: str = DEFAULT_SEMANTIC_MODEL_PATH,
    device: str = "auto",
    max_seq_length: int = 8192,
    batch_size: int = 32,
) -> RelationScorer:
    if mode == "bge_m3":
        return LocalBGERelationScorer(model_path, device, max_seq_length, batch_size)
    if mode == "hash_test":
        return HashRelationScorer()
    raise ValueError(f"Unsupported relation mode: {mode}")


class BudgetedSupportPathSelector:
    """Deterministic best-first search under an exact extension budget.

    A root-to-singleton generation is one extension.  Extending an existing
    path with one unused candidate is one extension.  Every generated path is
    scored immediately, so each generated path consumes one budget unit.
    """

    def __init__(
        self,
        relation_scorer: RelationScorer,
        expansion_budget: int = 3072,
        candidate_pool_size: int = 32,
        max_path_length: int | None = None,
        attribution_epsilon: float = 1e-8,
    ) -> None:
        self.relation_scorer = relation_scorer
        self.expansion_budget = max(1, int(expansion_budget))
        self.candidate_pool_size = max(1, int(candidate_pool_size))
        self.max_path_length = (
            self.candidate_pool_size if max_path_length is None else max(1, int(max_path_length))
        )
        self.attribution_epsilon = max(float(attribution_epsilon), 1e-12)
        self.last_selected_ids: List[str] = []
        self.last_candidate_ids: List[str] = []
        self.last_path_details: Dict = {}
        self.last_search_summary: Dict = {}

    def _similarity(self, left: str, right: str) -> float:
        return float(self.relation_scorer.similarity(left, right))

    @staticmethod
    def _entry_by_id(memory: Sequence[Mapping]) -> Dict[str, Mapping]:
        result: Dict[str, Mapping] = {}
        for entry in memory:
            mem_id = str(entry.get("mem_id", "") or "")
            if not mem_id:
                raise ValueError("Every active trace requires a stable mem_id")
            if mem_id in result:
                raise ValueError(f"Duplicate active mem_id: {mem_id}")
            result[mem_id] = entry
        return result

    def candidate_ids(self, query: str, memory: Sequence[Mapping]) -> List[str]:
        scored = [
            (self._similarity(query, entry_text(entry)), str(entry["mem_id"]))
            for entry in memory
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [mem_id for _, mem_id in scored[: self.candidate_pool_size]]

    def path_value(self, query: str, path_ids: Sequence[str], by_id: Mapping[str, Mapping]) -> float:
        if not path_ids:
            return 0.0
        terms = [self._similarity(query, entry_text(by_id[path_ids[0]]))]
        terms.extend(
            self._similarity(entry_text(by_id[left]), entry_text(by_id[right]))
            for left, right in zip(path_ids, path_ids[1:])
        )
        return float(sum(terms) / len(path_ids))

    @staticmethod
    def _is_better(path: Tuple[str, ...], score: float, best_path: Tuple[str, ...], best_score: float) -> bool:
        if score > best_score:
            return True
        if score < best_score:
            return False
        if not best_path:
            return True
        return (len(path), path) < (len(best_path), best_path)

    def select(self, query: str, memory: Sequence[Mapping], max_path_length: int | None = None) -> List[int]:
        if not memory:
            self.last_selected_ids = []
            self.last_candidate_ids = []
            self.last_path_details = self.prefix_attribution(query, [], {})
            self.last_search_summary = {
                "expansion_budget": self.expansion_budget,
                "expansions": 0,
                "candidate_count": 0,
                "explored_paths": 0,
                "termination": "empty_active_memory",
                "relation": self.relation_scorer.metadata(),
            }
            self.last_path_details["search"] = dict(self.last_search_summary)
            return []

        by_id = self._entry_by_id(memory)
        candidates = self.candidate_ids(query, memory)
        self.last_candidate_ids = list(candidates)
        maximum_length = min(
            len(candidates),
            self.max_path_length if max_path_length is None else max(1, int(max_path_length)),
        )
        frontier: List[Tuple[float, int, Tuple[str, ...]]] = []
        explored: Dict[Tuple[str, ...], float] = {}
        expansions = 0
        best_path: Tuple[str, ...] = ()
        best_score = -float("inf")

        for mem_id in candidates:
            if expansions >= self.expansion_budget:
                break
            path = (mem_id,)
            score = self.path_value(query, path, by_id)
            explored[path] = score
            heapq.heappush(frontier, (-score, len(path), path))
            expansions += 1
            if self._is_better(path, score, best_path, best_score):
                best_path, best_score = path, score

        while frontier and expansions < self.expansion_budget:
            _, _, parent = heapq.heappop(frontier)
            if len(parent) >= maximum_length:
                continue
            used = set(parent)
            for mem_id in candidates:
                if mem_id in used:
                    continue
                if expansions >= self.expansion_budget:
                    break
                child = parent + (mem_id,)
                score = self.path_value(query, child, by_id)
                explored[child] = score
                heapq.heappush(frontier, (-score, len(child), child))
                expansions += 1
                if self._is_better(child, score, best_path, best_score):
                    best_path, best_score = child, score

        selected_ids = list(best_path)
        id_to_index = {str(entry["mem_id"]): index for index, entry in enumerate(memory)}
        self.last_selected_ids = selected_ids
        self.last_path_details = self.prefix_attribution(query, selected_ids, by_id)
        self.last_search_summary = {
            "algorithm": "deterministic_best_first",
            "score": "length_normalized_head_plus_edges",
            "expansion_budget": self.expansion_budget,
            "expansions": expansions,
            "candidate_count": len(candidates),
            "active_memory_count": len(memory),
            "max_path_length": maximum_length,
            "selected_path_length": len(selected_ids),
            "selected_path_value": best_score if selected_ids else None,
            "explored_paths": len(explored),
            "termination": "budget_exhausted" if expansions >= self.expansion_budget else "no_valid_extension",
            "relation": self.relation_scorer.metadata(),
        }
        self.last_path_details["search"] = dict(self.last_search_summary)
        return [id_to_index[mem_id] for mem_id in selected_ids]

    def prefix_attribution(
        self,
        query: str,
        path_ids: Sequence[str],
        by_id: Mapping[str, Mapping],
    ) -> Dict:
        prefix_values = [0.0]
        for end in range(1, len(path_ids) + 1):
            prefix_values.append(self.path_value(query, path_ids[:end], by_id))
        marginal = [max(0.0, prefix_values[i + 1] - prefix_values[i]) for i in range(len(path_ids))]
        denominator = float(sum(marginal) + self.attribution_epsilon)
        attribution = {str(mem_id): float(delta / denominator) for mem_id, delta in zip(path_ids, marginal)}
        return {
            "path_ids": list(path_ids),
            "prefix_values": prefix_values,
            "marginal_gains": marginal,
            "attribution": attribution,
            "attributed_mass": float(sum(attribution.values())),
            "unassigned_mass": float(1.0 - sum(attribution.values())),
        }

    def similarity_credit(self, query: str, path_ids: Sequence[str], by_id: Mapping[str, Mapping]) -> Dict[str, float]:
        positive = [max(0.0, self._similarity(query, entry_text(by_id[mem_id]))) for mem_id in path_ids]
        total = float(sum(positive))
        if total <= 0.0:
            return {str(mem_id): 0.0 for mem_id in path_ids}
        return {str(mem_id): float(value / total) for mem_id, value in zip(path_ids, positive)}


def successor_trace_attribution(
    memory: Sequence[Mapping],
    path_attribution: Mapping[str, float],
) -> Dict[str, float]:
    """Map path attribution directly to the selected trace identity.

    The identity is the exact active record, including the successor created
    by a MERGE. Attribution is neither split by text ancestry nor propagated
    to predecessor records.
    """

    valid_ids = {str(entry["mem_id"]) for entry in memory}
    return {
        str(mem_id): float(alpha)
        for mem_id, alpha in path_attribution.items()
        if str(mem_id) in valid_ids and float(alpha) > 0.0
    }


def _merge_unique_text(left: str, right: str) -> str:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not right or right == left or right in left:
        return left
    if not left:
        return right
    if left in right:
        return right
    return left + "\n\n" + right


def _composition_key_signature(entry: Mapping) -> Tuple[str, ...]:
    """Return a deterministic case-insensitive composition-key signature."""

    values = entry.get("merge_anchor_keys", entry.get("composition_keys", []))
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(
        normalized
        for item in values
        if (normalized := str(item).strip().casefold())
    ))


def _merge_adds_transferable_content(anchor: Mapping, candidate: Mapping) -> bool:
    """Whether unbounded integration adds non-duplicate trace content.

    The candidate must add a non-duplicate composition key or field. Full
    eligibility also requires that some novelty survives length bounding.
    """

    for field_name in MERGE_TEXT_FIELDS:
        left = str(anchor.get(field_name, "") or "").strip()
        right = str(candidate.get(field_name, "") or "").strip()
        if _merge_unique_text(left, right) != left:
            return True
    anchor_keys = {
        str(item).strip().casefold()
        for item in anchor.get("composition_keys", [])
        if str(item).strip()
    }
    candidate_keys = {
        str(item).strip().casefold()
        for item in candidate.get("composition_keys", [])
        if str(item).strip()
    }
    return bool(candidate_keys - anchor_keys)


def _merge_retains_novel_content(anchor: Mapping, merged: Mapping) -> bool:
    """Require content gain after bounding, excluding truncation-only changes."""

    anchor_keys = {str(key).strip().casefold() for key in anchor.get("composition_keys", [])}
    if any(str(key).strip().casefold() not in anchor_keys for key in merged.get("composition_keys", [])):
        return True
    for field_name in MERGE_TEXT_FIELDS:
        original = str(anchor.get(field_name, "") or "").strip()
        # The extractive operator joins source pieces with blank lines and
        # marks clipped prefixes with an ellipsis. Neither is new knowledge.
        for piece in str(merged.get(field_name, "") or "").split("\n\n"):
            retained = piece.strip().removesuffix("…").rstrip()
            if retained and retained not in original:
                return True
    return False


def _clip_text_at_boundary(text: str, limit: int) -> str:
    """Deterministically preserve an extractive prefix within a hard limit."""

    text = str(text or "").strip()
    limit = max(0, int(limit))
    if len(text) <= limit:
        return text
    if limit == 0:
        return ""
    if limit == 1:
        return "…"
    body_limit = limit - 1
    lower_bound = max(1, body_limit // 2)
    cuts = [
        text.rfind(marker, lower_bound, body_limit + 1)
        for marker in ("\n", ". ", "; ", ", ", ": ")
    ]
    cut = max(cuts, default=-1)
    if cut < lower_bound:
        cut = body_limit
    return text[:cut].rstrip() + "…"


def _bounded_pair_text(left: str, right: str, limit: int) -> str:
    """Compact fields into extractive pieces within a hard character limit.

    Duplicate or contained content is stored once. For distinct fields, keep
    both sides when space permits; tight limits may retain only the right
    side or a truncation marker.
    """

    left = str(left or "").strip()
    right = str(right or "").strip()
    limit = max(0, int(limit))
    if not right or right == left or right in left:
        return _clip_text_at_boundary(left, limit)
    if not left or left in right:
        return _clip_text_at_boundary(right, limit)
    combined = left + "\n\n" + right
    if len(combined) <= limit:
        return combined
    if limit <= 2:
        return _clip_text_at_boundary(right, limit)
    available = limit - 2
    left_limit = available // 2
    right_limit = available - left_limit
    left_piece = _clip_text_at_boundary(left, left_limit)
    right_piece = _clip_text_at_boundary(right, right_limit)
    unused = available - len(left_piece) - len(right_piece)
    if unused > 0 and len(right_piece) < len(right):
        right_piece = _clip_text_at_boundary(right, len(right_piece) + unused)
        unused = available - len(left_piece) - len(right_piece)
    if unused > 0 and len(left_piece) < len(left):
        left_piece = _clip_text_at_boundary(left, len(left_piece) + unused)
    return "\n\n".join(part for part in (left_piece, right_piece) if part)


def merge_trace_records(
    anchor: Mapping,
    candidate: Mapping,
    *,
    max_chars: int = DEFAULT_MERGE_MAX_CHARS,
) -> Tuple[Dict, Dict]:
    """Bounded structured integration of overlapping transferable traces.

    Integration first performs field-aware exact de-duplication. Only when
    the complete serialized entry exceeds ``max_chars`` are its longest fields
    compacted into extractive pieces. Trailing composition keys may be removed.
    The policy checks the bounded result for retained novelty before admission.
    """

    max_chars = max(256, int(max_chars))
    merged = dict(anchor)
    text_fields = MERGE_TEXT_FIELDS
    source_pairs = {
        field_name: (str(anchor.get(field_name, "") or ""), str(candidate.get(field_name, "") or ""))
        for field_name in text_fields
    }
    for field_name, (left, right) in source_pairs.items():
        merged[field_name] = _merge_unique_text(left, right)
    merged["composition_keys"] = list(dict.fromkeys([
        *(str(item) for item in anchor.get("composition_keys", []) if str(item)),
        *(str(item) for item in candidate.get("composition_keys", []) if str(item)),
    ]))
    confidence_source = max(
        (anchor.get("confidence"), candidate.get("confidence")),
        key=confidence_value,
    )
    merged["confidence"] = confidence_label(confidence_source)
    merged["source"] = "neuromem_merged"
    # Keep the feasibility signature tied to the originally promoted anchor.
    # Newly merged keys must not make unrelated future candidates admissible.
    merged["merge_anchor_keys"] = list(_composition_key_signature(anchor))

    before_chars = len(entry_text(anchor))
    normalized_anchor = {**anchor, "confidence": confidence_label(anchor.get("confidence"))}
    normalized_before_chars = len(entry_text(normalized_anchor))
    candidate_chars = len(entry_text(candidate))
    unbounded_chars = len(entry_text(merged))
    minimum_lengths = {"knowledge": 8, "reasoning_pattern": 8}
    while len(entry_text(merged)) > max_chars:
        reducible = [
            field_name
            for field_name in text_fields
            if len(str(merged.get(field_name, ""))) > minimum_lengths.get(field_name, 0)
        ]
        if reducible:
            field_name = max(reducible, key=lambda name: (len(str(merged.get(name, ""))), name))
            current_length = len(str(merged.get(field_name, "")))
            minimum = minimum_lengths.get(field_name, 0)
            overflow = len(entry_text(merged)) - max_chars
            new_limit = max(minimum, current_length - max(1, overflow))
            left, right = source_pairs[field_name]
            merged[field_name] = _bounded_pair_text(left, right, new_limit)
            continue
        if merged["composition_keys"]:
            merged["composition_keys"].pop()
            continue
        raise RuntimeError("MERGE entry cannot satisfy the configured serialized character bound")

    after_chars = len(entry_text(merged))
    merge_audit = {
        "merge_operator": "bounded_structured_extractive",
        "merge_max_chars": max_chars,
        "merge_before_chars": before_chars,
        "merge_normalized_before_chars": normalized_before_chars,
        "merge_candidate_chars": candidate_chars,
        "merge_unbounded_chars": unbounded_chars,
        "merge_after_chars": after_chars,
        "merge_added_chars": max(0, after_chars - normalized_before_chars),
        "merge_removed_chars": max(0, before_chars - after_chars),
        "merge_compacted_chars": max(0, unbounded_chars - after_chars),
        "merge_was_compacted": bool(unbounded_chars > after_chars),
    }
    merged["merge_operator"] = merge_audit["merge_operator"]
    merged["merge_max_chars"] = max_chars
    provenance = dict(anchor.get("provenance", {}) or {})
    candidate_provenance = dict(candidate.get("provenance", {}) or {})
    for key, value in candidate_provenance.items():
        if isinstance(value, list):
            current = provenance.get(key, [])
            if not isinstance(current, list):
                current = [current]
            provenance[key] = list(dict.fromkeys([*current, *value]))
        elif key not in provenance:
            provenance[key] = value
    if provenance:
        merged["provenance"] = provenance
    return merged, merge_audit


@dataclass
class DecisionTrace:
    decision_id: str
    candidate_id: str
    action: str
    state: np.ndarray
    probabilities: np.ndarray
    time_step: int
    successor_id: str | None
    delta_memory: float
    delta_redundancy: float
    future_return: float = 0.0
    credit_hits: int = 0


@dataclass
class LifecyclePolicy:
    relation_scorer: RelationScorer = field(repr=False, compare=False)
    seed: int = 0
    learning_rate: float = 0.03
    gamma: float = 0.99
    credit_horizon: int = 32
    review_delay: int = 12
    lambda_memory: float = 0.05
    beta_redundancy: float = 0.10
    pending_retrieval_threshold: float = 0.60
    promotion_score_threshold: float = 0.50
    promotion_min_utility: float = 0.55
    max_pending_age: int = 96
    max_memory_size: int | None = None
    merge_max_chars: int = DEFAULT_MERGE_MAX_CHARS
    policy_mode: str = "learned"
    adam_beta1: float = 0.90
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    rng: np.random.Generator = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)
    bias: np.ndarray = field(init=False, repr=False)
    adam_m_w: np.ndarray = field(init=False, repr=False)
    adam_v_w: np.ndarray = field(init=False, repr=False)
    adam_m_b: np.ndarray = field(init=False, repr=False)
    adam_v_b: np.ndarray = field(init=False, repr=False)
    adam_step: int = field(default=0, init=False)
    baseline_sum: float = field(default=0.0, init=False)
    baseline_count: int = field(default=0, init=False)
    pending: Dict[str, Dict] = field(default_factory=dict, init=False)
    open_decisions: Dict[str, DecisionTrace] = field(default_factory=dict, init=False)
    history: List[Dict] = field(default_factory=list, init=False)
    policy_updates: int = field(default=0, init=False)
    forced_age_discards: int = field(default=0, init=False)
    last_pending_utility_updates: Dict[str, Dict] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.policy_mode not in {"learned", "static", "fixed_high_threshold"}:
            raise ValueError("policy_mode must be learned, static, or fixed_high_threshold")
        self.rng = np.random.default_rng(int(self.seed))
        feature_dim = int(self.relation_scorer.embedding_dimension) + 2
        self.weights = np.zeros((len(ACTIONS), feature_dim), dtype=float)
        self.bias = np.zeros(len(ACTIONS), dtype=float)
        self.adam_m_w = np.zeros_like(self.weights)
        self.adam_v_w = np.zeros_like(self.weights)
        self.adam_m_b = np.zeros_like(self.bias)
        self.adam_v_b = np.zeros_like(self.bias)
        self.credit_horizon = max(1, int(self.credit_horizon))
        self.review_delay = max(0, int(self.review_delay))
        self.max_pending_age = max(1, int(self.max_pending_age))
        self.merge_max_chars = max(256, int(self.merge_max_chars))
        if self.max_memory_size is not None:
            self.max_memory_size = max(1, int(self.max_memory_size))

    @property
    def baseline(self) -> float:
        return self.baseline_sum / self.baseline_count if self.baseline_count else 0.0

    def buffer_records(self, retrievable_only: bool = False) -> List[Dict]:
        rows: List[Dict] = []
        for token, entry in self.pending.items():
            if retrievable_only and float(entry["candidate_utility"]) < self.pending_retrieval_threshold:
                continue
            payload = dict(entry["payload"])
            payload.update({
                "mem_id": token,
                "source": "neuromem_buffer",
                "candidate_utility": float(entry["candidate_utility"]),
                "birth_utility": float(entry.get("birth_utility", entry["candidate_utility"])),
                "utility_reward_sum": float(entry.get("utility_reward_sum", 0.0)),
                "utility_weight": float(entry.get("utility_weight", 0.0)),
                "promotion_score": float(entry["promotion_score"]),
                "birth_step": int(entry["birth_step"]),
                "review_after": int(entry["review_after"]),
            })
            rows.append(payload)
        return rows

    def pending_as_memory(self) -> List[Dict]:
        return self.buffer_records(retrievable_only=True)

    def add_candidate(
        self,
        token: str,
        payload: Mapping,
        candidate_utility: float,
        promotion_score: float,
        time_step: int,
    ) -> None:
        if not token or token in self.pending:
            raise ValueError(f"Candidate token must be unique and non-empty: {token!r}")
        normalized = normalize_transferable_trace(payload)
        if isinstance(payload.get("provenance"), Mapping):
            normalized["provenance"] = dict(payload["provenance"])
        if payload.get("prompt_module"):
            normalized["prompt_module"] = str(payload["prompt_module"])
        if payload.get("trace_kind"):
            normalized["trace_kind"] = str(payload["trace_kind"])
        self.pending[token] = {
            "payload": normalized,
            "candidate_utility": _clip01(candidate_utility),
            "birth_utility": _clip01(candidate_utility),
            "utility_reward_sum": 0.0,
            "utility_weight": 0.0,
            "promotion_score": _clip01(promotion_score),
            "birth_step": int(time_step),
            "review_after": int(time_step) + self.review_delay,
        }

    def expire_pending(self, time_step: int) -> List[str]:
        expired: List[str] = []
        for token, entry in sorted(list(self.pending.items())):
            if int(time_step) - int(entry["birth_step"]) >= self.max_pending_age:
                self.pending.pop(token, None)
                # An externally expired RETAIN successor must stop receiving
                # path credit.  Its already-observed return is still finalized.
                for trace in self.open_decisions.values():
                    if trace.candidate_id == token and trace.successor_id == token:
                        trace.successor_id = None
                expired.append(token)
                self.forced_age_discards += 1
        return expired

    def relation_to_memory(self, payload: Mapping, memory: Sequence[Mapping]) -> float:
        if not memory:
            return 0.0
        text = entry_text(payload)
        return max(float(self.relation_scorer.similarity(text, entry_text(item))) for item in memory)

    def state(self, entry: Mapping, memory: Sequence[Mapping], time_step: int) -> np.ndarray:
        embedding = self.relation_scorer.encode(entry_text(entry["payload"]))
        age = float(max(0, int(time_step) - int(entry["birth_step"])))
        relation = self.relation_to_memory(entry["payload"], memory)
        state = np.concatenate([np.asarray(embedding, dtype=float), np.array([age, relation], dtype=float)])
        if state.shape != (self.weights.shape[1],):
            raise ValueError(f"Unexpected state shape: {state.shape}")
        return state

    def redundancy(self, memory: Sequence[Mapping]) -> float:
        """Mean positive nearest-neighbour raw cosine, normalized to [0, 1]."""

        if len(memory) < 2:
            return 0.0
        nearest_values: List[float] = []
        for index, entry in enumerate(memory):
            nearest = max(
                float(self.relation_scorer.similarity(entry_text(entry), entry_text(other)))
                for other_index, other in enumerate(memory)
                if other_index != index
            )
            nearest_values.append(max(0.0, nearest))
        return _clip01(sum(nearest_values) / len(nearest_values))

    def merge_compatibility(self, payload: Mapping, memory: Sequence[Mapping]) -> Dict:
        """Find overlapping persistent targets that gain transferable content.

        A compatible target shares at least one composition key with the
        immutable signature of the originally promoted anchor and would gain
        non-duplicate structured content after length bounding. BGE relation is
        a state feature and target-ranking signal, not a hard similarity threshold.
        """

        candidate_keys = _composition_key_signature(payload)
        candidate_key_set = set(candidate_keys)
        overlapping_targets: List[Dict] = []
        compatible_targets: List[Dict] = []
        for index, item in enumerate(memory):
            anchor_keys = _composition_key_signature(item)
            shared_keys = sorted(candidate_key_set.intersection(anchor_keys))
            if not shared_keys:
                continue
            target = {
                "index": int(index),
                "mem_id": str(item["mem_id"]),
                "anchor_keys": list(anchor_keys),
                "shared_keys": shared_keys,
            }
            overlapping_targets.append(target)
            if _merge_adds_transferable_content(item, payload):
                merged, _ = merge_trace_records(item, payload, max_chars=self.merge_max_chars)
                if _merge_retains_novel_content(item, merged):
                    compatible_targets.append(target)

        if not memory:
            reason = "empty_persistent_memory"
        elif not candidate_keys:
            reason = "candidate_has_no_composition_keys"
        elif not overlapping_targets:
            reason = "no_shared_immutable_anchor_keys"
        elif not compatible_targets:
            reason = "no_novel_transferable_content"
        else:
            reason = "compatible_non_noop_target_available"
        return {
            "gate": MERGE_GATE_NAME,
            "reason": reason,
            "candidate_keys": list(candidate_keys),
            "overlapping_targets": overlapping_targets,
            "compatible_targets": compatible_targets,
        }

    def allowed_actions(
        self,
        entry: Mapping,
        memory: Sequence[Mapping],
        merge_compatibility: Mapping | None = None,
    ) -> List[bool]:
        capacity_ok = self.max_memory_size is None or len(memory) < self.max_memory_size
        below_age_cap = int(entry.get("current_age", 0)) < self.max_pending_age
        promote_ok = (
            capacity_ok
            and float(entry["promotion_score"]) >= self.promotion_score_threshold
            and float(entry["candidate_utility"]) >= self.promotion_min_utility
        )
        compatibility = merge_compatibility or self.merge_compatibility(entry["payload"], memory)
        merge_ok = bool(compatibility["compatible_targets"])
        return [promote_ok, merge_ok, below_age_cap, True]

    def _static_action(self, entry: Mapping, memory: Sequence[Mapping], allowed: Sequence[bool]) -> Tuple[str, np.ndarray]:
        """Fixed four-action ablation with no learned action selection.

        The age boundary takes priority, followed by admissible PROMOTE,
        compatible MERGE with sufficient utility, and otherwise RETAIN.
        """

        age = int(entry.get("current_age", 0))
        if age >= self.max_pending_age:
            action = "DISCARD"
        elif allowed[ACTION_TO_INDEX["PROMOTE"]]:
            action = "PROMOTE"
        elif allowed[ACTION_TO_INDEX["MERGE"]] and float(entry["candidate_utility"]) >= self.promotion_min_utility:
            action = "MERGE"
        else:
            action = "RETAIN"
        probabilities = np.zeros(len(ACTIONS), dtype=float)
        probabilities[ACTION_TO_INDEX[action]] = 1.0
        return action, probabilities

    def _fixed_high_threshold_action(
        self,
        entry: Mapping,
        allowed: Sequence[bool],
    ) -> Tuple[str, np.ndarray]:
        """Non-RL high-threshold admission ablation.

        PROMOTE requires the pending-retrieval utility threshold and uses
        promotion_min_utility as a stricter confidence-score threshold.
        Otherwise, choose an admissible MERGE at the retrieval utility
        threshold, DISCARD at the age boundary, or RETAIN.
        """

        age = int(entry.get("current_age", 0))
        strict_promote = (
            allowed[ACTION_TO_INDEX["PROMOTE"]]
            and float(entry["candidate_utility"]) >= self.pending_retrieval_threshold
            and float(entry["promotion_score"]) >= self.promotion_min_utility
        )
        if strict_promote:
            action = "PROMOTE"
        elif (
            allowed[ACTION_TO_INDEX["MERGE"]]
            and float(entry["candidate_utility"]) >= self.pending_retrieval_threshold
        ):
            action = "MERGE"
        elif age >= self.max_pending_age:
            action = "DISCARD"
        else:
            action = "RETAIN"
        probabilities = np.zeros(len(ACTIONS), dtype=float)
        probabilities[ACTION_TO_INDEX[action]] = 1.0
        return action, probabilities

    def sample_action(
        self,
        state: np.ndarray,
        entry: Mapping,
        memory: Sequence[Mapping],
        allowed: Sequence[bool] | None = None,
    ) -> Tuple[str, np.ndarray]:
        allowed = list(allowed) if allowed is not None else self.allowed_actions(entry, memory)
        if self.policy_mode == "static":
            return self._static_action(entry, memory, allowed)
        if self.policy_mode == "fixed_high_threshold":
            return self._fixed_high_threshold_action(entry, allowed)
        probabilities = _softmax(self.weights @ state + self.bias, allowed)
        index = int(self.rng.choice(len(ACTIONS), p=probabilities))
        return ACTIONS[index], probabilities

    def merge_target(
        self,
        payload: Mapping,
        memory: Sequence[Mapping],
        merge_compatibility: Mapping | None = None,
    ) -> int:
        compatibility = merge_compatibility or self.merge_compatibility(payload, memory)
        compatible_indexes = {
            int(item["index"])
            for item in compatibility["compatible_targets"]
        }
        if not compatible_indexes:
            raise ValueError("MERGE requires a compatible non-noop persistent target")
        candidate_text = entry_text(payload)
        ranked = [
            (float(self.relation_scorer.similarity(candidate_text, entry_text(item))), str(item["mem_id"]), index)
            for index, item in enumerate(memory)
            if index in compatible_indexes
        ]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][2]

    def review_and_apply(self, memory: MutableSequence[Dict], time_step: int) -> List[Dict]:
        events: List[Dict] = []
        for token in sorted(list(self.pending)):
            entry = self.pending.get(token)
            if entry is None or int(time_step) < int(entry["review_after"]):
                continue
            entry["current_age"] = max(0, int(time_step) - int(entry["birth_step"]))
            state = self.state(entry, memory, time_step)
            merge_compatibility = self.merge_compatibility(entry["payload"], memory)
            allowed = self.allowed_actions(entry, memory, merge_compatibility)
            action, probabilities = self.sample_action(state, entry, memory, allowed)
            before_size = len(memory)
            before_redundancy = self.redundancy(memory)
            successor_id: str | None = None
            merge_audit: Dict = {}

            if action == "PROMOTE":
                promoted = dict(entry["payload"])
                promoted.update({
                    "mem_id": token,
                    "source": "neuromem_promoted",
                    "merge_anchor_keys": list(_composition_key_signature(promoted)),
                })
                memory.append(promoted)
                self.pending.pop(token, None)
                successor_id = token
            elif action == "MERGE":
                target_index = self.merge_target(entry["payload"], memory, merge_compatibility)
                anchor = memory[target_index]
                target_details = next(
                    item
                    for item in merge_compatibility["compatible_targets"]
                    if int(item["index"]) == target_index
                )
                merged, merge_audit = merge_trace_records(
                    anchor,
                    entry["payload"],
                    max_chars=self.merge_max_chars,
                )
                merged["mem_id"] = str(anchor["mem_id"])
                memory[target_index] = merged
                successor_id = str(anchor["mem_id"])
                merge_audit.update({
                    "merge_target_id": successor_id,
                    "merge_target_anchor_keys": list(target_details["anchor_keys"]),
                    "merge_shared_anchor_keys": list(target_details["shared_keys"]),
                })
                self.pending.pop(token, None)
            elif action == "RETAIN":
                successor_id = token
                entry["review_after"] = int(time_step) + self.review_delay
            elif action == "DISCARD":
                self.pending.pop(token, None)
            else:  # pragma: no cover
                raise RuntimeError(f"Unsupported action: {action}")

            after_size = len(memory)
            after_redundancy = self.redundancy(memory)
            decision_id = f"{token}@{int(time_step)}"
            trace = DecisionTrace(
                decision_id=decision_id,
                candidate_id=token,
                action=action,
                state=state,
                probabilities=probabilities,
                time_step=int(time_step),
                successor_id=successor_id,
                delta_memory=float(after_size - before_size),
                delta_redundancy=float(after_redundancy - before_redundancy),
            )
            self.open_decisions[decision_id] = trace
            positive_probabilities = probabilities[probabilities > 0.0]
            action_entropy = float(
                -np.sum(positive_probabilities * np.log(positive_probabilities))
            )
            events.append({
                "decision_id": decision_id,
                "candidate_id": token,
                "action": action,
                "successor_id": successor_id,
                "admissible_actions": {name: bool(allowed[index]) for index, name in enumerate(ACTIONS)},
                "probabilities": {name: float(probabilities[index]) for index, name in enumerate(ACTIONS)},
                "candidate_utility": float(entry["candidate_utility"]),
                "birth_utility": float(entry.get("birth_utility", entry["candidate_utility"])),
                "utility_evidence_weight": float(entry.get("utility_weight", 0.0)),
                "promotion_score": float(entry["promotion_score"]),
                "age": int(entry["current_age"]),
                "delta_memory": trace.delta_memory,
                "delta_redundancy": trace.delta_redundancy,
                "action_entropy": action_entropy,
                "max_action_probability": float(np.max(probabilities)),
                "merge_gate": merge_compatibility["gate"],
                "merge_gate_reason": merge_compatibility["reason"],
                "merge_candidate_keys": list(merge_compatibility["candidate_keys"]),
                "merge_overlapping_target_count": len(merge_compatibility["overlapping_targets"]),
                "merge_compatible_target_count": len(merge_compatibility["compatible_targets"]),
                "merge_compatible_target_ids": [
                    item["mem_id"] for item in merge_compatibility["compatible_targets"]
                ],
                **merge_audit,
            })
        return events

    def credit_selected_path(
        self,
        successor_attribution: Mapping[str, float],
        task_reward: float,
        time_step: int,
    ) -> Dict[str, float]:
        """Accumulate delayed returns and update retained candidates' utility.

        Each open action decision accumulates discounted task reward weighted
        by its exact successor's attribution, independently of other open
        decisions (paper Eqs. 14–15).
        """

        self.last_pending_utility_updates = {}
        # The static controller freezes both birth utility and policy
        # parameters. Other modes update buffered utility from path evidence.
        if self.policy_mode != "static":
            for successor_id, alpha in successor_attribution.items():
                entry = self.pending.get(str(successor_id))
                if entry is None or float(alpha) <= 0.0:
                    continue
                weight = float(alpha)
                before = float(entry["candidate_utility"])
                entry["utility_reward_sum"] = float(entry.get("utility_reward_sum", 0.0)) + weight * float(task_reward)
                entry["utility_weight"] = float(entry.get("utility_weight", 0.0)) + weight
                numerator = float(entry.get("birth_utility", before)) + float(entry["utility_reward_sum"])
                denominator = 1.0 + float(entry["utility_weight"])
                entry["candidate_utility"] = _clip01(numerator / denominator)
                self.last_pending_utility_updates[str(successor_id)] = {
                    "before": before,
                    "after": float(entry["candidate_utility"]),
                    "attribution": weight,
                    "task_reward": float(task_reward),
                    "evidence_weight": float(entry["utility_weight"]),
                }

        groups: Dict[str, List[DecisionTrace]] = {}
        for trace in self.open_decisions.values():
            elapsed = int(time_step) - trace.time_step
            if trace.successor_id is None or elapsed <= 0 or elapsed > self.credit_horizon:
                continue
            groups.setdefault(trace.successor_id, []).append(trace)

        credited: Dict[str, float] = {}
        for successor_id, alpha in successor_attribution.items():
            traces = groups.get(str(successor_id), [])
            if float(alpha) <= 0.0 or not traces:
                continue
            for trace in traces:
                elapsed = int(time_step) - trace.time_step
                contribution = (self.gamma ** elapsed) * float(alpha) * float(task_reward)
                trace.future_return += contribution
                trace.credit_hits += 1
                credited[trace.decision_id] = contribution
        return credited

    def _adam_ascent(self, trace: DecisionTrace, adjusted_return: float) -> Dict:
        baseline_before = self.baseline
        advantage = float(adjusted_return - baseline_before)
        if self.policy_mode in {"static", "fixed_high_threshold"}:
            return {
                "baseline_before": baseline_before,
                "advantage": None,
                "baseline_after": baseline_before,
                "baseline_count": self.baseline_count,
                "adam_step": self.adam_step,
            }
        if self.policy_mode == "learned":
            target = np.zeros(len(ACTIONS), dtype=float)
            target[ACTION_TO_INDEX[trace.action]] = 1.0
            score_gradient = target - trace.probabilities
            gradient_w = advantage * np.outer(score_gradient, trace.state)
            gradient_b = advantage * score_gradient
            self.adam_step += 1
            self.adam_m_w = self.adam_beta1 * self.adam_m_w + (1.0 - self.adam_beta1) * gradient_w
            self.adam_v_w = self.adam_beta2 * self.adam_v_w + (1.0 - self.adam_beta2) * np.square(gradient_w)
            self.adam_m_b = self.adam_beta1 * self.adam_m_b + (1.0 - self.adam_beta1) * gradient_b
            self.adam_v_b = self.adam_beta2 * self.adam_v_b + (1.0 - self.adam_beta2) * np.square(gradient_b)
            correction1 = 1.0 - self.adam_beta1 ** self.adam_step
            correction2 = 1.0 - self.adam_beta2 ** self.adam_step
            m_w = self.adam_m_w / correction1
            v_w = self.adam_v_w / correction2
            m_b = self.adam_m_b / correction1
            v_b = self.adam_v_b / correction2
            self.weights += self.learning_rate * m_w / (np.sqrt(v_w) + self.adam_epsilon)
            self.bias += self.learning_rate * m_b / (np.sqrt(v_b) + self.adam_epsilon)
            self.policy_updates += 1
        self.baseline_sum += float(adjusted_return)
        self.baseline_count += 1
        return {
            "baseline_before": baseline_before,
            "advantage": advantage,
            "baseline_after": self.baseline,
            "baseline_count": self.baseline_count,
            "adam_step": self.adam_step,
        }

    def learning_state_dict(self) -> Dict:
        """Policy, optimizer, baseline, and RNG state without episode memory.

        Buffer contents, persistent records, open decisions, and history are
        excluded so learning state can be carried across memory episodes.
        """

        return {
            "schema": "neuromem-learning-state/1",
            "weights": self.weights.copy(),
            "bias": self.bias.copy(),
            "adam_m_w": self.adam_m_w.copy(),
            "adam_v_w": self.adam_v_w.copy(),
            "adam_m_b": self.adam_m_b.copy(),
            "adam_v_b": self.adam_v_b.copy(),
            "adam_step": self.adam_step,
            "baseline_sum": self.baseline_sum,
            "baseline_count": self.baseline_count,
            "policy_updates": self.policy_updates,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_learning_state_dict(self, payload: Mapping) -> None:
        if payload.get("schema") != "neuromem-learning-state/1":
            raise ValueError("Unsupported NeuroMem learning-state schema")
        for name in ("weights", "bias", "adam_m_w", "adam_v_w", "adam_m_b", "adam_v_b"):
            value = np.asarray(payload[name], dtype=float)
            target = getattr(self, name)
            if value.shape != target.shape:
                raise ValueError(f"Learning-state {name} shape mismatch: {value.shape} != {target.shape}")
            target[:] = value
        self.adam_step = int(payload["adam_step"])
        self.baseline_sum = float(payload["baseline_sum"])
        self.baseline_count = int(payload["baseline_count"])
        self.policy_updates = int(payload.get("policy_updates", self.adam_step))
        self.rng.bit_generator.state = dict(payload["rng_state"])

    def finalize_decisions(self, time_step: int, force: bool = False) -> List[Dict]:
        finalized: List[Dict] = []
        for decision_id, trace in list(self.open_decisions.items()):
            elapsed = int(time_step) - trace.time_step
            if not force and elapsed < self.credit_horizon:
                continue
            adjusted_return = float(
                trace.future_return
                - self.lambda_memory * trace.delta_memory
                - self.beta_redundancy * trace.delta_redundancy
            )
            update = self._adam_ascent(trace, adjusted_return)
            record = {
                "decision_id": decision_id,
                "candidate_id": trace.candidate_id,
                "action": trace.action,
                "successor_id": trace.successor_id,
                "elapsed": elapsed,
                "future_return": trace.future_return,
                "delta_memory": trace.delta_memory,
                "delta_redundancy": trace.delta_redundancy,
                "memory_penalty": self.lambda_memory * trace.delta_memory,
                "redundancy_penalty": self.beta_redundancy * trace.delta_redundancy,
                "regularized_return": adjusted_return,
                "credit_hits": trace.credit_hits,
                **update,
            }
            finalized.append(record)
            self.history.append(record)
            self.open_decisions.pop(decision_id, None)
        return finalized

    def state_dict(self) -> Dict:
        return deepcopy({
            "schema": "neuromem-policy/1",
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "credit_horizon": self.credit_horizon,
            "review_delay": self.review_delay,
            "lambda_memory": self.lambda_memory,
            "beta_redundancy": self.beta_redundancy,
            "pending_retrieval_threshold": self.pending_retrieval_threshold,
            "promotion_score_threshold": self.promotion_score_threshold,
            "promotion_min_utility": self.promotion_min_utility,
            "max_pending_age": self.max_pending_age,
            "max_memory_size": self.max_memory_size,
            "merge_max_chars": self.merge_max_chars,
            "policy_mode": self.policy_mode,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "weights": self.weights.copy(),
            "bias": self.bias.copy(),
            "adam_m_w": self.adam_m_w.copy(),
            "adam_v_w": self.adam_v_w.copy(),
            "adam_m_b": self.adam_m_b.copy(),
            "adam_v_b": self.adam_v_b.copy(),
            "adam_step": self.adam_step,
            "baseline_sum": self.baseline_sum,
            "baseline_count": self.baseline_count,
            "pending": self.pending,
            "open_decisions": self.open_decisions,
            "history": self.history,
            "policy_updates": self.policy_updates,
            "forced_age_discards": self.forced_age_discards,
            "rng_state": self.rng.bit_generator.state,
            "relation": self.relation_scorer.metadata(),
        })

    def load_state_dict(self, payload: Mapping) -> None:
        """Restore configuration and independent policy state with the supplied scorer."""

        if payload.get("schema") != "neuromem-policy/1":
            raise ValueError("Unsupported NeuroMem policy checkpoint schema")
        config = {
            setting.name: deepcopy(payload[setting.name])
            for setting in fields(LifecyclePolicy)
            if setting.init and setting.name != "relation_scorer"
        }
        if config["policy_mode"] not in {"learned", "static", "fixed_high_threshold"}:
            raise ValueError("Unsupported checkpoint policy mode")
        arrays = {}
        for name in ("weights", "bias", "adam_m_w", "adam_v_w", "adam_m_b", "adam_v_b"):
            value = np.asarray(payload[name], dtype=float)
            target = getattr(self, name)
            if value.shape != target.shape:
                raise ValueError(f"Checkpoint {name} shape mismatch: {value.shape} != {target.shape}")
            arrays[name] = value.copy()
        rng = np.random.default_rng(int(config["seed"]))
        rng.bit_generator.state = deepcopy(payload["rng_state"])
        for name, value in config.items():
            setattr(self, name, value)
        for name, value in arrays.items():
            setattr(self, name, value)
        self.adam_step = int(payload["adam_step"])
        self.baseline_sum = float(payload["baseline_sum"])
        self.baseline_count = int(payload["baseline_count"])
        self.pending = deepcopy(dict(payload.get("pending", {})))
        self.open_decisions = deepcopy(dict(payload.get("open_decisions", {})))
        self.history = deepcopy(list(payload.get("history", [])))
        self.policy_updates = int(payload.get("policy_updates", 0))
        self.forced_age_discards = int(payload.get("forced_age_discards", 0))
        self.rng = rng

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(self.state_dict(), handle)


def assert_prediction_only(text: str) -> None:
    lowered = str(text).casefold()
    forbidden = ("ground-truth answer", "gold answer", "reference answer")
    if any(marker in lowered for marker in forbidden):
        raise ValueError("Prediction input contains a gold/reference-answer marker")
