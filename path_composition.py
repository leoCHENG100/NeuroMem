from typing import Mapping, Sequence

from neuromem_core import entry_text
from neuromem_prompts import compose_selected_path

_P_PATH_VALIDATION_MAX_GENERATIONS = 3

def _deterministic_path_fallback(
    *,
    selected_traces: Sequence[Mapping],
    candidate_traces: Sequence[Mapping] | None,
    validation_errors: Sequence[str],
) -> dict:
    """Preserve the immutable search path when P_path violates its contract."""

    selected_ids = [str(trace.get("mem_id", "")) for trace in selected_traces]
    if any(not mem_id for mem_id in selected_ids) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("P_path requires unique, non-empty selected mem_id values")
    selected_set = set(selected_ids)
    proposed_by_id = {
        str(trace.get("mem_id", "")): trace
        for trace in (candidate_traces or selected_traces)
        if str(trace.get("mem_id", ""))
    }
    for trace in selected_traces:
        proposed_by_id[str(trace["mem_id"])] = trace

    support = [
        {
            "memory_id": mem_id,
            "role": f"Immutable search-selected support trace {position}",
            "reason": (
                "Preserved at its deterministic search position after P_path output "
                "failed strict validation."
            ),
        }
        for position, mem_id in enumerate(selected_ids, start=1)
    ]
    grounded_sections = [
        (
            f"[Selected NeuroMem trace {position}: {selected_ids[position - 1]}; "
            f"role={support[position - 1]['role']}]\n{entry_text(trace)}"
        )
        for position, trace in enumerate(selected_traces, start=1)
    ]
    model_context = (
        "P_path synthesis was unavailable after strict validation. Use the immutable "
        "search-selected traces above in their original order, applying only their "
        "reusable schema, operator, and reasoning guidance to the current task."
    )
    return {
        "invoked": True,
        "selected_support_path": support,
        "selected_path_ids": selected_ids,
        "composition_rationale": (
            "Deterministic fallback preserves the complete frozen search-selected path "
            "without adding, dropping, or re-ordering traces."
        ),
        "task_adaptation": (
            "Adapt the grounded reusable trace contents to the current query without "
            "copying instance-specific answers."
        ),
        "cautions": (
            "No model-generated P_path synthesis passed validation; rely only on the "
            "grounded selected traces and verify task-specific details during execution."
        ),
        "composed_context": (
            "\n\n".join(grounded_sections)
            + "\n\n[P_path deterministic validation fallback]\n"
            + model_context
        ),
        "model_composed_context": model_context,
        "grounded_selected_trace_ids": selected_ids,
        "excluded_memories": [
            {
                "memory_id": mem_id,
                "reason": "Not selected by the deterministic support-path search.",
            }
            for mem_id in proposed_by_id
            if mem_id not in selected_set
        ],
        "confidence_label": "low",
        "confidence": 0.25,
        "schema_validation": {
            "schema": "support_path_composition/1",
            "generation_attempts": _P_PATH_VALIDATION_MAX_GENERATIONS,
            "generation_retries": _P_PATH_VALIDATION_MAX_GENERATIONS - 1,
            "used_deterministic_immutable_path_fallback": True,
            "errors": list(validation_errors),
        },
    }


def compose_with_recovery(
    client,
    *,
    query: str,
    selected_traces: Sequence[Mapping],
    candidate_traces: Sequence[Mapping] | None = None,
    task_context: str = "",
) -> dict:
    """Retry P_path with unchanged inputs, then preserve the frozen path.

    Validation errors are recorded in metadata, not appended to subsequent
    prompts. ``generation_retries`` excludes the initial generation attempt.
    """

    if not selected_traces:
        return compose_selected_path(
            client,
            query=query,
            selected_traces=selected_traces,
            candidate_traces=candidate_traces,
            task_context=task_context,
        )
    selected_ids = [str(trace.get("mem_id", "")) for trace in selected_traces]
    if any(not mem_id for mem_id in selected_ids) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("P_path requires unique, non-empty selected mem_id values")

    validation_errors = []
    for generation_attempt in range(1, _P_PATH_VALIDATION_MAX_GENERATIONS + 1):
        try:
            result = compose_selected_path(
                client,
                query=query,
                selected_traces=selected_traces,
                candidate_traces=candidate_traces,
                task_context=task_context,
            )
            result = dict(result)
            result["schema_validation"] = {
                "schema": "support_path_composition/1",
                "generation_attempts": generation_attempt,
                "generation_retries": generation_attempt - 1,
                "used_deterministic_immutable_path_fallback": False,
                "errors": validation_errors,
            }
            return result
        except ValueError as error:
            validation_errors.append(f"{type(error).__name__}: {str(error)[:500]}")

    print(
        "[neuromem] P_path failed strict validation after "
        f"{_P_PATH_VALIDATION_MAX_GENERATIONS} generations; preserving immutable path",
        flush=True,
    )
    return _deterministic_path_fallback(
        selected_traces=selected_traces,
        candidate_traces=candidate_traces,
        validation_errors=validation_errors,
    )
