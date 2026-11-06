from __future__ import annotations

import unittest
import numpy as np

from neuromem_policy import (
    POLICY_NAME,
    NeuroMemPolicy,
    version_id,
)



class TinyScorer:
    @property
    def embedding_dimension(self):
        return 2

    def encode(self, text):
        return np.asarray([0.6, 0.8], dtype=float)

    def similarity(self, left, right):
        return 0.25

    def metadata(self):
        return {"kind": "tiny-test"}


def candidate(name: str, confidence: float = 0.4) -> dict:
    return {
        "knowledge": f"candidate knowledge {name}",
        "applicability": "when applicable",
        "reasoning_pattern": f"candidate reasoning {name}",
        "composition_keys": ["root_topic"],
        "evidence": f"candidate evidence {name}",
        "cautions": "none",
        "confidence": confidence,
    }


def anchor(mem_id: str) -> dict:
    return {
        "mem_id": mem_id,
        "knowledge": f"anchor knowledge {mem_id}",
        "applicability": "when applicable",
        "reasoning_pattern": f"anchor reasoning {mem_id}",
        "composition_keys": ["root_topic"],
        "merge_anchor_keys": ["root_topic"],
        "evidence": f"anchor evidence {mem_id}",
        "cautions": "none",
        "confidence": 0.8,
    }


def add_due_candidate(policy, token: str, *, utility: float = 0.8, score: float = 0.4) -> None:
    policy.add_candidate(token, candidate(token, score), utility, score, 0)


def test_promotion_requires_gates_and_creates_exact_version_zero():
    policy = NeuroMemPolicy(
        TinyScorer(), review_delay=0, policy_mode="static"
    )
    policy.add_candidate("c", candidate("c", 0.8), 0.8, 0.8, 0)
    memory = []

    events = policy.review_and_apply(memory, 0)

    assert [event["action"] for event in events] == ["PROMOTE"]
    assert events[0]["successor_id"] == version_id("c", 0)
    assert events[0]["admissible_actions"]["PROMOTE"] is True
    assert memory[0]["mem_id"] == version_id("c", 0)
    assert memory[0]["logical_mem_id"] == "c"
    assert memory[0]["memory_version"] == 0


def test_promotion_keeps_both_thresholds_at_all_review_steps():
    for step in (0, 200):
        for score, utility, allowed in ((0.50, 0.55, True), (0.49, 1.0, False), (1.0, 0.54, False)):
            policy = NeuroMemPolicy(TinyScorer(), review_delay=0, policy_mode="static")
            policy.add_candidate("c", candidate("c", score), utility, score, step)
            events = policy.review_and_apply([], step)
            assert events[0]["admissible_actions"]["PROMOTE"] is allowed
            assert (events[0]["action"] == "PROMOTE") is allowed


def test_same_target_merges_remain_admissible_and_create_a_version_chain():
    policy = NeuroMemPolicy(
        TinyScorer(), review_delay=0, credit_horizon=32, policy_mode="static"
    )
    memory = [anchor("A")]
    add_due_candidate(policy, "c1")
    add_due_candidate(policy, "c2")

    events = policy.review_and_apply(memory, 0)

    assert [event["action"] for event in events] == ["MERGE", "MERGE"]
    assert all(event["admissible_actions"]["MERGE"] for event in events)
    assert events[0]["parent_version_id"] == version_id("A", 0)
    assert events[0]["successor_id"] == version_id("A", 1)
    assert events[1]["parent_version_id"] == version_id("A", 1)
    assert events[1]["successor_id"] == version_id("A", 2)
    assert len(memory) == 1
    assert memory[0]["mem_id"] == version_id("A", 2)
    assert [event["delta_memory"] for event in events] == [0.0, 0.0]


def test_credit_matches_only_the_retrieved_exact_version():
    policy = NeuroMemPolicy(
        TinyScorer(), review_delay=0, credit_horizon=32, gamma=0.99,
        policy_mode="static"
    )
    memory = [anchor("A")]
    add_due_candidate(policy, "c1")
    add_due_candidate(policy, "c2")
    events = policy.review_and_apply(memory, 0)

    credited = policy.credit_selected_path(
        {memory[0]["mem_id"]: 1.0}, task_reward=1.0, time_step=1
    )

    first = policy.open_decisions[events[0]["decision_id"]]
    second = policy.open_decisions[events[1]["decision_id"]]
    assert first.successor_id == version_id("A", 1)
    assert first.future_return == 0.0
    assert second.successor_id == version_id("A", 2)
    assert second.future_return == 0.99
    assert list(credited) == [events[1]["decision_id"]]


def test_replaced_version_keeps_prior_credit_but_gets_no_descendant_credit():
    policy = NeuroMemPolicy(
        TinyScorer(), review_delay=0, credit_horizon=32, gamma=1.0,
        policy_mode="static"
    )
    memory = [anchor("A")]
    add_due_candidate(policy, "c1")
    first_event = policy.review_and_apply(memory, 0)[0]
    assert memory[0]["mem_id"] == version_id("A", 1)
    policy.credit_selected_path({version_id("A", 1): 0.5}, 1.0, 1)

    policy.add_candidate("c2", candidate("c2"), 0.8, 0.4, 2)
    second_event = policy.review_and_apply(memory, 2)[0]
    assert memory[0]["mem_id"] == version_id("A", 2)
    policy.credit_selected_path({version_id("A", 2): 1.0}, 1.0, 3)

    first = policy.open_decisions[first_event["decision_id"]]
    second = policy.open_decisions[second_event["decision_id"]]
    assert first.future_return == 0.5
    assert second.future_return == 1.0
    assert policy.version_lineage[version_id("A", 2)]["parent_version_id"] == version_id("A", 1)


def test_inactive_versions_are_metadata_only_and_not_counted_in_memory():
    policy = NeuroMemPolicy(
        TinyScorer(), review_delay=0, policy_mode="static"
    )
    memory = [anchor("A")]
    add_due_candidate(policy, "c1")
    add_due_candidate(policy, "c2")
    policy.review_and_apply(memory, 0)

    assert len(memory) == 1
    assert len(policy.version_lineage) == 3
    assert {record["mem_id"] for record in memory} == {version_id("A", 2)}
    assert all("knowledge" not in row for row in policy.version_lineage.values())


def test_candidate_level_single_open_defers_nonterminal_review():
    policy = NeuroMemPolicy(
        TinyScorer(), review_delay=12, credit_horizon=32, policy_mode="static"
    )
    policy.add_candidate("c", candidate("c"), 0.8, 0.4, 0)
    assert [event["action"] for event in policy.review_and_apply([], 12)] == ["RETAIN"]
    assert policy.review_and_apply([], 24) == []
    assert policy.last_deferred_review_ids == ["c"]


def test_checkpoint_retains_exact_version_no_lineage_credit():
    policy = NeuroMemPolicy(
        TinyScorer(), review_delay=0, policy_mode="static"
    )
    memory = [anchor("A")]
    add_due_candidate(policy, "c1")
    policy.review_and_apply(memory, 0)
    payload = policy.state_dict()

    assert payload["policy_name"] == POLICY_NAME
    assert payload["merge_target_scope"] == "compatible_active_memory"
    assert payload["lineage_credit_propagation"] is False
    assert payload["inactive_versions_active"] is False

    restored = NeuroMemPolicy(TinyScorer())
    restored.load_state_dict(payload)
    assert restored.version_lineage == policy.version_lineage
    assert restored.logical_latest_version == policy.logical_latest_version


def load_tests(loader, tests, pattern):
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            tests.addTest(unittest.FunctionTestCase(fn))
    return tests
