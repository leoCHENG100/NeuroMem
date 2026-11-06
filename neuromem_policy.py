"""NeuroMem consolidation policy with exact successor identity.

Each PROMOTE or MERGE decision is bound to the precise memory record it
produces. Only the current record of each logical memory slot is retrievable.
Inactive records retain lineage metadata, and credit does not propagate to
ancestors. A MERGE replaces one active record without increasing persistent
memory cardinality."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Mapping, MutableSequence

try:
    from .neuromem_buffer import (
        POLICY_NAME as BASE_POLICY_NAME,
        BufferReviewPolicy,
    )
except ImportError:  # pragma: no cover - direct script execution
    from neuromem_buffer import (
        POLICY_NAME as BASE_POLICY_NAME,
        BufferReviewPolicy,
    )


POLICY_NAME = "NeuroMem"
VERSION_ID_SEPARATOR = "::neuromem-v"
SUCCESSOR_IDENTITY_MODE = "exact_active_successor_no_lineage_credit"


def version_id(logical_mem_id: str, version: int) -> str:
    """Return the deterministic physical id for one logical-memory version."""

    logical = str(logical_mem_id)
    if not logical:
        raise ValueError("logical_mem_id must be non-empty")
    if int(version) < 0:
        raise ValueError("memory version must be non-negative")
    return f"{logical}{VERSION_ID_SEPARATOR}{int(version)}"


class NeuroMemPolicy(BufferReviewPolicy):
    """Bind lifecycle credit to exact, active successor versions."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.version_lineage: Dict[str, Dict] = {}
        self.logical_latest_version: Dict[str, int] = {}
        self.initialized_identity_count = 0

    def _record_version(
        self,
        *,
        logical_mem_id: str,
        memory_version: int,
        parent_version_id: str | None,
        created_by_decision: str | None,
        action: str,
    ) -> Dict:
        physical_id = version_id(logical_mem_id, memory_version)
        metadata = {
            "version_id": physical_id,
            "logical_mem_id": str(logical_mem_id),
            "memory_version": int(memory_version),
            "parent_version_id": (
                None if parent_version_id is None else str(parent_version_id)
            ),
            "created_by_decision": (
                None if created_by_decision is None else str(created_by_decision)
            ),
            "action": str(action),
        }
        existing = self.version_lineage.get(physical_id)
        if existing is not None and existing != metadata:
            raise ValueError(f"Conflicting lineage metadata for {physical_id}")
        self.version_lineage[physical_id] = metadata
        self.logical_latest_version[str(logical_mem_id)] = max(
            int(memory_version),
            int(self.logical_latest_version.get(str(logical_mem_id), -1)),
        )
        return metadata

    def _ensure_active_versions(self, memory: MutableSequence[Dict]) -> None:
        """Assign exact identities to untagged active records without copying them."""

        assigned_ids: Dict[str, str] = {}
        seen_physical_ids: set[str] = set()
        for record in memory:
            old_id = str(record.get("mem_id", ""))
            if not old_id:
                raise ValueError("Persistent memory records require a non-empty mem_id")

            logical = str(record.get("logical_mem_id", old_id))
            memory_version = int(record.get("memory_version", 0))
            physical_id = version_id(logical, memory_version)
            if physical_id in seen_physical_ids:
                raise ValueError(f"Duplicate active memory version: {physical_id}")
            seen_physical_ids.add(physical_id)

            parent = record.get("parent_version_id")
            created_by = record.get("created_by_decision")
            action = str(record.get("version_created_by_action", "INITIAL"))
            metadata = self._record_version(
                logical_mem_id=logical,
                memory_version=memory_version,
                parent_version_id=None if parent is None else str(parent),
                created_by_decision=None if created_by is None else str(created_by),
                action=action,
            )
            record.update({
                "mem_id": metadata["version_id"],
                "logical_mem_id": metadata["logical_mem_id"],
                "memory_version": metadata["memory_version"],
                "parent_version_id": metadata["parent_version_id"],
                "created_by_decision": metadata["created_by_decision"],
                "version_created_by_action": metadata["action"],
                "successor_identity_mode": SUCCESSOR_IDENTITY_MODE,
            })
            if old_id != physical_id:
                assigned_ids[old_id] = physical_id

        if assigned_ids:
            self.initialized_identity_count += len(assigned_ids)
            for trace in self.open_decisions.values():
                if trace.successor_id in assigned_ids:
                    trace.successor_id = assigned_ids[str(trace.successor_id)]

    def _next_version(
        self,
        *,
        parent_version_id: str,
        decision_id: str,
    ) -> Dict:
        parent = self.version_lineage.get(str(parent_version_id))
        if parent is None:
            raise ValueError(f"Unknown MERGE parent version: {parent_version_id}")
        logical = str(parent["logical_mem_id"])
        next_number = int(self.logical_latest_version.get(logical, -1)) + 1
        return self._record_version(
            logical_mem_id=logical,
            memory_version=next_number,
            parent_version_id=str(parent_version_id),
            created_by_decision=str(decision_id),
            action="MERGE",
        )

    def review_and_apply(self, memory: MutableSequence[Dict], time_step: int) -> list[Dict]:
        # Initialize exact identities before evaluating merge targets.
        # Identity metadata is excluded from semantic scoring and trace text.
        self._ensure_active_versions(memory)
        events = super().review_and_apply(memory, time_step)

        # Resolve each slot alias to the latest successor created in this
        # review batch, including multiple merges into the same slot.
        latest_for_batch_id = {
            str(record["mem_id"]): str(record["mem_id"])
            for record in memory
            if str(record.get("mem_id", ""))
        }

        for event in events:
            action = str(event["action"])
            decision_id = str(event["decision_id"])
            trace = self.open_decisions[decision_id]

            if action == "PROMOTE":
                logical = str(event["candidate_id"])
                metadata = self._record_version(
                    logical_mem_id=logical,
                    memory_version=0,
                    parent_version_id=None,
                    created_by_decision=decision_id,
                    action="PROMOTE",
                )
                base_id = str(event["successor_id"])
                latest_for_batch_id[base_id] = str(metadata["version_id"])
                trace.successor_id = str(metadata["version_id"])
                event.update({
                    "successor_id": metadata["version_id"],
                    "logical_successor_id": metadata["logical_mem_id"],
                    "successor_version": metadata["memory_version"],
                    "parent_version_id": None,
                    "successor_identity_mode": SUCCESSOR_IDENTITY_MODE,
                    "lineage_credit_propagation": False,
                })
            elif action == "MERGE":
                base_target_id = str(event["merge_target_id"])
                parent_version_id = latest_for_batch_id.get(
                    base_target_id, base_target_id
                )
                metadata = self._next_version(
                    parent_version_id=parent_version_id,
                    decision_id=decision_id,
                )
                latest_for_batch_id[base_target_id] = str(metadata["version_id"])
                trace.successor_id = str(metadata["version_id"])
                event.update({
                    "merge_target_id": parent_version_id,
                    "merge_input_version_id": parent_version_id,
                    "successor_id": metadata["version_id"],
                    "logical_successor_id": metadata["logical_mem_id"],
                    "successor_version": metadata["memory_version"],
                    "parent_version_id": metadata["parent_version_id"],
                    "successor_identity_mode": SUCCESSOR_IDENTITY_MODE,
                    "lineage_credit_propagation": False,
                })
            else:
                event.update({
                    "successor_identity_mode": SUCCESSOR_IDENTITY_MODE,
                    "lineage_credit_propagation": False,
                })

            event["policy_name"] = POLICY_NAME

        # Only final active versions are materialized in M. Inactive versions
        # remain in ``version_lineage`` for audit and cannot enter search,
        # prompts, redundancy, or memory-cardinality calculations.
        for record in memory:
            base_id = str(record["mem_id"])
            active_id = latest_for_batch_id.get(base_id, base_id)
            metadata = self.version_lineage.get(active_id)
            if metadata is None:
                raise ValueError(f"Missing active-version metadata for {active_id}")
            record.update({
                "mem_id": metadata["version_id"],
                "logical_mem_id": metadata["logical_mem_id"],
                "memory_version": metadata["memory_version"],
                "parent_version_id": metadata["parent_version_id"],
                "created_by_decision": metadata["created_by_decision"],
                "version_created_by_action": metadata["action"],
                "successor_identity_mode": SUCCESSOR_IDENTITY_MODE,
            })

        return events

    def state_dict(self) -> Dict:
        payload = super().state_dict()
        payload.update({
            "policy_name": POLICY_NAME,
            "successor_identity_mode": SUCCESSOR_IDENTITY_MODE,
            "single_open_pending_candidate": True,
            "merge_target_scope": "compatible_active_memory",
            "exact_merge_successor": True,
            "lineage_credit_propagation": False,
            "inactive_versions_active": False,
            "version_lineage": deepcopy(self.version_lineage),
            "logical_latest_version": dict(self.logical_latest_version),
            "initialized_identity_count": int(
                self.initialized_identity_count
            ),
        })
        return payload

    def load_state_dict(self, payload: Mapping) -> None:
        policy_name = payload.get("policy_name")
        if policy_name not in (None, POLICY_NAME):
            raise ValueError(f"Unsupported policy name: {policy_name}")
        parent_payload = dict(payload)
        parent_payload["policy_name"] = BASE_POLICY_NAME
        super().load_state_dict(parent_payload)
        self.version_lineage = {
            str(key): deepcopy(dict(value))
            for key, value in dict(payload.get("version_lineage", {})).items()
        }
        self.logical_latest_version = {
            str(key): int(value)
            for key, value in dict(
                payload.get("logical_latest_version", {})
            ).items()
        }
        self.initialized_identity_count = int(
            payload.get("initialized_identity_count", 0)
        )
