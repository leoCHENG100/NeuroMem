"""NeuroMem transient-buffer review scheduling.

The policy scales candidate age by the maximum buffer age and defers a
candidate's nonterminal review while its earlier delayed-return window is
open. A terminal review remains available at the age boundary."""

from __future__ import annotations

from typing import Dict, Mapping, MutableSequence

import numpy as np

try:
    from .neuromem_core import LifecyclePolicy, entry_text
except ImportError:  # pragma: no cover - direct script execution
    from neuromem_core import LifecyclePolicy, entry_text


POLICY_NAME = "NeuroMemBuffer"


class BufferReviewPolicy(LifecyclePolicy):
    """Use normalized age and suppress overlapping nonterminal reviews.

    ``review_delay`` remains the earliest reconsideration time.  If a previous
    action-conditioned return for the same pending candidate has not finalized,
    the overdue review remains eligible and is executed as soon as that open
    decision is removed.  At ``max_pending_age`` the normal terminal review is
    still executed, with RETAIN masked by the base policy.
    """

    deferred_review_count: int = 0
    last_deferred_review_ids: list[str]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.deferred_review_count = 0
        self.last_deferred_review_ids = []

    def state(self, entry: Mapping, memory: list[Mapping], time_step: int) -> np.ndarray:
        embedding = self.relation_scorer.encode(entry_text(entry["payload"]))
        raw_age = float(max(0, int(time_step) - int(entry["birth_step"])))
        normalized_age = float(np.clip(raw_age / float(self.max_pending_age), 0.0, 1.0))
        relation = self.relation_to_memory(entry["payload"], memory)
        state = np.concatenate(
            [np.asarray(embedding, dtype=float), np.array([normalized_age, relation], dtype=float)]
        )
        if state.shape != (self.weights.shape[1],):
            raise ValueError(f"Unexpected state shape: {state.shape}")
        return state

    def review_and_apply(self, memory: MutableSequence[Dict], time_step: int) -> list[Dict]:
        open_candidate_ids = {str(trace.candidate_id) for trace in self.open_decisions.values()}
        deferred: Dict[str, int] = {}
        for token, entry in self.pending.items():
            age = max(0, int(time_step) - int(entry["birth_step"]))
            is_due = int(time_step) >= int(entry["review_after"])
            if (
                is_due
                and token in open_candidate_ids
                and age < int(self.max_pending_age)
            ):
                deferred[token] = int(entry["review_after"])
                entry["review_after"] = int(time_step) + 1

        self.last_deferred_review_ids = sorted(deferred)
        self.deferred_review_count += len(deferred)
        try:
            events = super().review_and_apply(memory, time_step)
        finally:
            for token, original_review_after in deferred.items():
                if token in self.pending:
                    self.pending[token]["review_after"] = original_review_after

        for event in events:
            event["policy_name"] = POLICY_NAME
            event["raw_age"] = int(event["age"])
            event["normalized_age_feature"] = float(
                np.clip(float(event["age"]) / float(self.max_pending_age), 0.0, 1.0)
            )
        return events

    def state_dict(self) -> Dict:
        payload = super().state_dict()
        payload.update({
            "policy_name": POLICY_NAME,
            "age_feature": "clip(raw_age / max_pending_age, 0, 1)",
            "single_open_pending_candidate": True,
            "deferred_review_count": int(self.deferred_review_count),
        })
        return payload

    def load_state_dict(self, payload: Mapping) -> None:
        policy_name = payload.get("policy_name")
        if policy_name not in (None, POLICY_NAME):
            raise ValueError(f"Unsupported policy name: {policy_name}")
        super().load_state_dict(payload)
        self.deferred_review_count = int(payload.get("deferred_review_count", 0))
        self.last_deferred_review_ids = []
