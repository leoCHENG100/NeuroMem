"""NeuroMem transferable-knowledge and support-context prompts.

P_ttk constructs a transferable trace from a completed prediction-side
trajectory and scalar feedback. P_path explains and composes the fixed,
ordered support path for the task agent; it does not score paths or choose
memory lifecycle actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Dict, Mapping, Protocol, Sequence

try:
    from .neuromem_core import (
        assert_prediction_only,
        entry_text,
        normalize_transferable_trace,
        remove_instance_values,
    )
except ImportError:  # pragma: no cover - direct script execution
    from neuromem_core import (
        assert_prediction_only,
        entry_text,
        normalize_transferable_trace,
        remove_instance_values,
    )


P_TTK_SYSTEM = """You are P_ttk in the NeuroMem memory framework.
Convert a completed task trajectory into reusable, task-level knowledge.
Never copy a final answer, patient/person identifier, exact date, or other
instance-specific value. Preserve reusable schema names, operator semantics,
reasoning patterns, composition cues, supporting evidence, and cautions.
Return exactly one JSON object and no prose."""

P_TTK_USER = """Task family: {task_family}
Current query (for abstraction only):
{query}

Task-solving trajectory (prediction side; no gold/reference answer):
{trajectory}

Final feedback (task-native scalar reward in [0,1]; no gold/reference text):
{task_reward}

Outcome summary (must not contain a gold/reference answer):
{outcome_summary}

Selected support traces used by the task:
{selected_traces}

Return this schema:
{{
  "knowledge": "reusable knowledge without the current final answer",
  "applicability": "when this knowledge applies",
  "reasoning_pattern": "abstract multi-step reasoning pattern",
  "composition_keys": ["short keys for combining compatible traces"],
  "evidence": "prediction-side evidence and the supplied scalar final feedback only",
  "cautions": "failure modes and non-applicable cases",
  "confidence": "low / medium / high"
}}
confidence must be exactly low, medium, or high."""

P_PATH_SYSTEM = """You are P_path in the NeuroMem memory framework.
The frozen deterministic search algorithm has already selected and ordered the
numerical support path. Explain the role and reason of every selected trace,
record why other search-proposed candidates are excluded, and compose the fixed
selected path into useful downstream context. Do not re-rank, drop, add, or
retrieve traces. Do not compute a path score and do not make
PROMOTE/MERGE/RETAIN/DISCARD decisions. Return exactly one JSON object and no
prose."""

P_PATH_USER = """Current query:
{query}

The deterministic search-selected path is ordered and immutable:
{selected_path}

Other search-proposed candidate traces (not in the selected path):
{excluded_candidates}

Optional task context:
{task_context}

Return this schema:
{{
  "selected_support_path": [
    {{
      "memory_id": "one selected id, preserving the exact search order",
      "role": "the role this trace plays",
      "reason": "why it supports the task"
    }}
  ],
  "composition_rationale": "how the ordered traces complement each other",
  "task_adaptation": "how to adapt the reusable knowledge to this query",
  "excluded_memories": [
    {{
      "memory_id": "one non-selected proposed id",
      "reason": "why it is excluded"
    }}
  ],
  "cautions": "conflicts, uncertainty, and unsafe compositions",
  "composed_context": "the complete support context for the downstream task",
  "confidence": "low / medium / high"
}}
Do not include a final answer to the current query."""


P_TTK_SCHEMA_MAX_GENERATIONS = 3
_P_TTK_CONFIDENCE_LABELS = frozenset({"low", "medium", "high"})
_P_TTK_SCHEMA_REPAIR = """

SCHEMA CORRECTION: The previous response did not satisfy the required
transferable_trace/1 schema ({error}). Regenerate the complete JSON object.
The confidence value must be exactly one of: low, medium, high. Do not add
prose, markdown fences, or a final answer."""


def _prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_manifest() -> Dict:
    return {
        "p_ttk": {
            "system_sha256": _prompt_sha256(P_TTK_SYSTEM),
            "user_template_sha256": _prompt_sha256(P_TTK_USER),
            "schema": "transferable_trace/1",
            "schema_max_generations": P_TTK_SCHEMA_MAX_GENERATIONS,
            "invalid_confidence_fallback": "low_with_audit_metadata",
        },
        "p_path": {
            "system_sha256": _prompt_sha256(P_PATH_SYSTEM),
            "user_template_sha256": _prompt_sha256(P_PATH_USER),
            "schema": "support_path_composition/1",
        },
    }


class JSONPromptClient(Protocol):
    def generate_json(self, system: str, user: str, purpose: str) -> Mapping: ...


def _extract_json_object(text: str) -> Mapping:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response does not contain a JSON object")
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be one JSON object")
    return payload


@dataclass
class OpenAIJSONPromptClient:
    """Generate structured JSON with provider retries and prompt fingerprints."""

    config: Mapping
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1600
    timeout: float = 120.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("P_ttk/P_path require the openai package") from error
        api_type = str(self.config.get("api_type", "openai") or "openai").casefold()
        api_key = self.config.get("api_key")
        base_url = self.config.get("base_url") or self.config.get("api_base")
        if "azure" in api_type:
            self._client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=self.config.get("api_version"),
                timeout=self.timeout,
            )
        else:
            self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)
        self.model = str(self.model or self.config.get("model") or "")
        if not self.model:
            raise ValueError("Prompt client requires an explicit model/deployment name")
        self.calls: list[Dict] = []

    def generate_json(self, system: str, user: str, purpose: str) -> Mapping:
        """Retry provider failures; let callers handle invalid generated JSON."""

        last_error: Exception | None = None
        for attempt in range(1, max(1, int(self.max_attempts)) + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=float(self.temperature),
                    max_tokens=int(self.max_tokens),
                )
            except Exception as error:
                last_error = error
                continue
            # A generation with invalid syntax consumes a schema attempt, not
            # another provider retry. Transport failures remain distinguishable.
            self.calls.append({
                "purpose": str(purpose),
                "attempt": attempt,
                "system_sha256": _prompt_sha256(system),
                "user_sha256": _prompt_sha256(user),
            })
            return _extract_json_object(response.choices[0].message.content)
        raise RuntimeError(f"{purpose} failed after {self.max_attempts} attempts") from last_error

    def metadata(self) -> Dict:
        return {
            "client": "openai-compatible-chat-completions",
            "model": self.model,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
            "timeout": float(self.timeout),
            "max_attempts": int(self.max_attempts),
        }


def build_ttk_candidate(
    client: JSONPromptClient,
    *,
    task_family: str,
    query: str,
    trajectory: str,
    selected_traces: Sequence[Mapping],
    task_reward: float,
    outcome_summary: str,
    provenance: Mapping | None = None,
) -> Dict:
    """Invoke P_ttk with scalar final feedback but no gold/reference text."""

    assert_prediction_only(trajectory)
    assert_prediction_only(outcome_summary)
    reward = max(0.0, min(1.0, float(task_reward)))
    rendered_selected = [
        {"mem_id": str(trace.get("mem_id", "")), "trace": entry_text(trace)}
        for trace in selected_traces
    ]
    user = P_TTK_USER.format(
        task_family=str(task_family),
        query=str(query),
        trajectory=str(trajectory),
        task_reward=f"{reward:.8f}",
        outcome_summary=str(outcome_summary),
        selected_traces=json.dumps(rendered_selected, ensure_ascii=False, sort_keys=True),
    )
    schema_errors: list[str] = []
    invalid_confidence_labels: list[str] = []
    payload: Mapping = {}
    trace: Dict | None = None
    generation_attempts = 0
    for schema_attempt in range(1, P_TTK_SCHEMA_MAX_GENERATIONS + 1):
        generation_attempts = schema_attempt
        request_user = user
        if schema_errors:
            request_user += _P_TTK_SCHEMA_REPAIR.format(error=schema_errors[-1])
        payload = {}
        try:
            payload = client.generate_json(P_TTK_SYSTEM, request_user, purpose="P_ttk")
            confidence_label = str(payload.get("confidence", "")).strip().casefold()
            if confidence_label not in _P_TTK_CONFIDENCE_LABELS:
                invalid_confidence_labels.append(str(payload.get("confidence", ""))[:160])
                raise ValueError("confidence must be exactly low, medium, or high")
            trace = normalize_transferable_trace(payload)
            break
        except ValueError as error:
            schema_errors.append(str(error)[:300])

    used_low_fallback = False
    if trace is None:
        # After schema retries, assign low confidence to an otherwise-valid
        # trace. Other schema violations remain fatal.
        fallback_payload = dict(payload)
        fallback_payload["confidence"] = "low"
        try:
            trace = normalize_transferable_trace(fallback_payload)
        except ValueError as error:
            raise RuntimeError(
                "P_ttk failed transferable_trace/1 validation after "
                f"{generation_attempts} generations: {error}"
            ) from error
        used_low_fallback = True

    trace = remove_instance_values(trace, "\n".join((str(query), str(trajectory))))
    if provenance:
        trace["provenance"] = dict(provenance)
    trace["prompt_module"] = "P_ttk"
    trace["schema_validation"] = {
        "schema": "transferable_trace/1",
        "generation_attempts": generation_attempts,
        "repair_requests": max(0, generation_attempts - 1),
        "used_low_confidence_fallback": used_low_fallback,
        "invalid_confidence_labels": invalid_confidence_labels,
        "errors": schema_errors,
    }
    return trace


def compose_selected_path(
    client: JSONPromptClient,
    *,
    query: str,
    selected_traces: Sequence[Mapping],
    candidate_traces: Sequence[Mapping] | None = None,
    task_context: str = "",
) -> Dict:
    """Invoke P_path iff search selected a non-empty ordered path."""

    if not selected_traces:
        return {
            "invoked": False,
            "selected_support_path": [],
            "selected_path_ids": [],
            "composition_rationale": "",
            "task_adaptation": "",
            "cautions": "",
            "composed_context": "",
            "model_composed_context": "",
            "grounded_selected_trace_ids": [],
            "excluded_memories": [],
            "confidence_label": "low",
            "confidence": 0.0,
        }
    selected_ids = [str(trace.get("mem_id", "")) for trace in selected_traces]
    if any(not mem_id for mem_id in selected_ids) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("P_path requires unique, non-empty selected mem_id values")
    selected_path = [
        {"position": index + 1, "mem_id": selected_ids[index], "trace": entry_text(trace)}
        for index, trace in enumerate(selected_traces)
    ]
    selected_set = set(selected_ids)
    proposed_by_id = {
        str(trace.get("mem_id", "")): trace
        for trace in (candidate_traces or selected_traces)
        if str(trace.get("mem_id", ""))
    }
    for trace in selected_traces:
        proposed_by_id[str(trace["mem_id"])] = trace
    excluded = [
        {"mem_id": mem_id, "trace": entry_text(trace)}
        for mem_id, trace in proposed_by_id.items()
        if mem_id not in selected_set
    ]
    user = P_PATH_USER.format(
        query=str(query),
        selected_path=json.dumps(selected_path, ensure_ascii=False, sort_keys=True),
        excluded_candidates=json.dumps(excluded, ensure_ascii=False, sort_keys=True),
        task_context=str(task_context or "(none)"),
    )
    payload = client.generate_json(P_PATH_SYSTEM, user, purpose="P_path")
    required_fields = {
        "selected_support_path",
        "composition_rationale",
        "task_adaptation",
        "excluded_memories",
        "cautions",
        "composed_context",
        "confidence",
    }
    missing = sorted(required_fields - set(payload))
    if missing:
        raise ValueError(f"P_path payload is missing required fields: {missing}")
    selected_support_path = payload.get("selected_support_path", [])
    if not isinstance(selected_support_path, list):
        raise ValueError("P_path selected_support_path must be a JSON list")
    returned_ids = [str(item.get("memory_id", "")) for item in selected_support_path if isinstance(item, Mapping)]
    if returned_ids != selected_ids:
        raise ValueError("P_path changed, dropped, added, or re-ordered the selected path")
    normalized_support = []
    for item in selected_support_path:
        if not isinstance(item, Mapping):
            raise ValueError("Every selected_support_path item must be an object")
        role = str(item.get("role", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not role or not reason:
            raise ValueError("Every selected P_path trace requires a non-empty role and reason")
        normalized_support.append({
            "memory_id": str(item.get("memory_id", "")),
            "role": role,
            "reason": reason,
        })
    excluded_rows = payload.get("excluded_memories", [])
    if not isinstance(excluded_rows, list):
        raise ValueError("P_path excluded_memories must be a JSON list")
    allowed_excluded_ids = set(proposed_by_id) - selected_set
    normalized_excluded = []
    for item in excluded_rows:
        if not isinstance(item, Mapping):
            raise ValueError("Every excluded_memories item must be an object")
        mem_id = str(item.get("memory_id", ""))
        if mem_id not in allowed_excluded_ids:
            raise ValueError("P_path excluded an id outside the non-selected proposed candidate set")
        reason = str(item.get("reason", "")).strip()
        if not reason:
            raise ValueError("Every excluded P_path trace requires a non-empty reason")
        normalized_excluded.append({"memory_id": mem_id, "reason": reason})
    if len({item["memory_id"] for item in normalized_excluded}) != len(normalized_excluded):
        raise ValueError("P_path excluded_memories contains duplicate ids")
    reported_excluded_ids = {item["memory_id"] for item in normalized_excluded}
    for mem_id in sorted(allowed_excluded_ids - reported_excluded_ids):
        normalized_excluded.append({
            "memory_id": mem_id,
            "reason": "Not selected by the deterministic support-path search.",
        })
    model_composed_context = str(payload.get("composed_context", "")).strip()
    if not model_composed_context:
        raise ValueError("P_path composed_context must be non-empty")
    for field_name in ("composition_rationale", "task_adaptation", "cautions"):
        if not str(payload.get(field_name, "")).strip():
            raise ValueError(f"P_path {field_name} must be non-empty")
    confidence_label = str(payload.get("confidence", "medium")).strip().casefold()
    confidence_mapping = {"low": 0.25, "medium": 0.50, "high": 0.90}
    if confidence_label not in confidence_mapping:
        raise ValueError("P_path confidence must be low, medium, or high")
    grounded_sections = []
    for position, (trace, support) in enumerate(zip(selected_traces, normalized_support), start=1):
        grounded_sections.append(
            f"[Selected NeuroMem trace {position}: {selected_ids[position - 1]}; "
            f"role={support['role']}]\n{entry_text(trace)}"
        )
    grounded_selected_context = "\n\n".join(grounded_sections)
    composed_context = (
        grounded_selected_context
        + "\n\n[P_path task-specific synthesis]\n"
        + model_composed_context
    )
    return {
        "invoked": True,
        "selected_support_path": normalized_support,
        "selected_path_ids": selected_ids,
        "composition_rationale": str(payload.get("composition_rationale", "")).strip(),
        "task_adaptation": str(payload.get("task_adaptation", "")).strip(),
        "cautions": str(payload.get("cautions", "")).strip(),
        "composed_context": composed_context,
        "model_composed_context": model_composed_context,
        "grounded_selected_trace_ids": selected_ids,
        "excluded_memories": normalized_excluded,
        "confidence_label": confidence_label,
        "confidence": confidence_mapping[confidence_label],
    }
