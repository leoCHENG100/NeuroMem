"""Verify policy snapshots and normal delayed-return settlement."""

from copy import deepcopy
from dataclasses import fields
import unittest

import numpy as np

from neuromem_core import LifecyclePolicy
from neuromem_policy import NeuroMemPolicy
from tests.test_memory_policy import TinyScorer, anchor, candidate


class PolicyStateTests(unittest.TestCase):
    def test_lineage_snapshot_is_independent_in_both_directions(self):
        policy = NeuroMemPolicy(TinyScorer(), review_delay=0, policy_mode="static")
        memory = [anchor("A")]
        policy.add_candidate("c", candidate("c"), 0.8, 0.25, 0)
        event = policy.review_and_apply(memory, 0)[0]
        parent, successor = event["parent_version_id"], event["successor_id"]
        snapshot = policy.state_dict()

        for identity, metadata in policy.version_lineage.items():
            self.assertIsNot(snapshot["version_lineage"][identity], metadata)
        snapshot["version_lineage"][parent]["action"] = "Snapshot annotation"
        snapshot["version_lineage"][successor]["action"] = "Snapshot annotation"
        self.assertEqual(policy.version_lineage[parent]["action"], "INITIAL")
        self.assertEqual(policy.version_lineage[successor]["action"], "MERGE")
        # Editing an exported snapshot cannot invalidate the next policy review.
        self.assertEqual(policy.review_and_apply(memory, 1), [])

        independent = policy.state_dict()
        policy.version_lineage[parent]["action"] = "Policy annotation"
        self.assertEqual(independent["version_lineage"][parent]["action"], "INITIAL")

    def test_nested_lineage_metadata_is_independent_on_export_and_restore(self):
        policy = NeuroMemPolicy(TinyScorer(), review_delay=0, policy_mode="static")
        memory = [anchor("A")]
        policy.review_and_apply(memory, 0)
        identity = memory[0]["mem_id"]
        policy.version_lineage[identity]["annotations"] = {"labels": ["initial"]}
        snapshot = policy.state_dict()
        restored = NeuroMemPolicy(TinyScorer())
        restored.load_state_dict(snapshot)

        original_labels = policy.version_lineage[identity]["annotations"]["labels"]
        snapshot_labels = snapshot["version_lineage"][identity]["annotations"]["labels"]
        restored_labels = restored.version_lineage[identity]["annotations"]["labels"]
        original_labels.append("policy")
        self.assertEqual(snapshot_labels, ["initial"])
        snapshot_labels.append("snapshot")
        self.assertEqual(restored_labels, ["initial"])
        restored_labels.append("restored")
        self.assertEqual(snapshot_labels, ["initial", "snapshot"])
        self.assertEqual(original_labels, ["initial", "policy"])

    def test_checkpoint_restores_all_policy_settings(self):
        original = NeuroMemPolicy(
            TinyScorer(), seed=7, learning_rate=0.02, gamma=0.8, credit_horizon=9,
            review_delay=0, lambda_memory=0.03, beta_redundancy=0.07,
            pending_retrieval_threshold=0.65, promotion_score_threshold=0.6,
            promotion_min_utility=0.7, max_pending_age=60, max_memory_size=5,
            merge_max_chars=1024, policy_mode="static", adam_beta1=0.8,
            adam_beta2=0.95, adam_epsilon=1e-7,
        )
        original.add_candidate("c", candidate("c"), 1.0, 0.25, 0)
        original.review_and_apply([anchor("A")], 0)
        snapshot = original.state_dict()
        restored = NeuroMemPolicy(TinyScorer())
        restored.load_state_dict(snapshot)
        for setting in fields(LifecyclePolicy):
            if setting.init and setting.name != "relation_scorer":
                with self.subTest(setting=setting.name):
                    self.assertEqual(getattr(restored, setting.name), getattr(original, setting.name))
        self.assertEqual(restored.version_lineage, original.version_lineage)
        self.assertEqual(restored.logical_latest_version, original.logical_latest_version)

    def test_checkpoint_restoration_continues_the_same_updates_and_actions(self):
        original = NeuroMemPolicy(TinyScorer(), seed=9, gamma=0.8, review_delay=0, credit_horizon=2)
        memory = [anchor("A")]
        original.add_candidate("first", candidate("first"), 0.9, 0.25, 0)
        original.review_and_apply(memory, 0)
        original.credit_selected_path({memory[0]["mem_id"]: 1.0}, 1.0, 1)
        original.finalize_decisions(2)
        original.add_candidate("pending", candidate("pending"), 0.9, 0.25, 2)
        original.review_and_apply(memory, 2)
        restored = NeuroMemPolicy(TinyScorer())
        restored.load_state_dict(deepcopy(original.state_dict()))
        restored_memory = deepcopy(memory)
        for step in range(3, 8):
            for policy, active in ((original, memory), (restored, restored_memory)):
                policy.credit_selected_path({active[0]["mem_id"]: 1.0}, 1.0, step)
            self.assertEqual(original.finalize_decisions(step), restored.finalize_decisions(step))
            for policy in (original, restored):
                policy.add_candidate(f"c{step}", candidate(f"c{step}"), 0.9, 0.25, step)
            self.assertEqual(original.review_and_apply(memory, step),
                             restored.review_and_apply(restored_memory, step))
            self.assertEqual(memory, restored_memory)
            for name in ("weights", "bias", "adam_m_w", "adam_v_w", "adam_m_b", "adam_v_b"):
                np.testing.assert_array_equal(getattr(original, name), getattr(restored, name))
        self.assertEqual(original.rng.bit_generator.state, restored.rng.bit_generator.state)

    def test_checkpoint_snapshot_and_restored_state_do_not_alias(self):
        original = NeuroMemPolicy(TinyScorer(), review_delay=0, policy_mode="static")
        original.add_candidate("c", candidate("c"), 0.8, 0.25, 0)
        original.review_and_apply([], 0)
        snapshot = original.state_dict()
        restored = NeuroMemPolicy(TinyScorer())
        restored.load_state_dict(snapshot)
        original.pending["c"]["payload"]["knowledge"] = "Changed original"
        original.open_decisions["c@0"].future_return = 9.0
        self.assertNotEqual(snapshot["pending"]["c"]["payload"]["knowledge"], "Changed original")
        self.assertEqual(snapshot["open_decisions"]["c@0"].future_return, 0.0)
        restored.pending["c"]["payload"]["knowledge"] = "Changed restored"
        restored.open_decisions["c@0"].future_return = 4.0
        self.assertNotEqual(snapshot["pending"]["c"]["payload"]["knowledge"], "Changed restored")
        self.assertEqual(snapshot["open_decisions"]["c@0"].future_return, 0.0)

    def test_invalid_checkpoint_shape_does_not_partially_replace_weights(self):
        policy = NeuroMemPolicy(TinyScorer())
        snapshot = policy.state_dict()
        snapshot["weights"][:] = 5.0
        snapshot["bias"] = np.zeros(5)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            policy.load_state_dict(snapshot)
        np.testing.assert_array_equal(policy.weights, np.zeros_like(policy.weights))

    def test_default_horizon_finalizes_during_the_stream_without_force(self):
        policy = NeuroMemPolicy(TinyScorer())
        memory = []
        policy.add_candidate("c", candidate("c", 0.8), 1.0, 0.8, 0)
        self.assertEqual(policy.review_and_apply(memory, policy.review_delay - 1), [])
        self.assertEqual(len(policy.review_and_apply(memory, policy.review_delay)), 1)
        deadline = policy.review_delay + policy.credit_horizon
        for step in range(policy.review_delay + 1, deadline):
            policy.credit_selected_path({row["mem_id"]: 1.0 for row in memory}, 1.0, step)
            self.assertEqual(policy.finalize_decisions(step), [])
            self.assertEqual(policy.adam_step, 0)
        settled = policy.finalize_decisions(deadline)
        self.assertEqual(len(settled), 1)
        self.assertEqual(settled[0]["elapsed"], policy.credit_horizon)
        self.assertEqual(policy.adam_step, 1)
        self.assertEqual(len(policy.open_decisions), 0)
