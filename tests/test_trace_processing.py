"""Exercise numeric abstraction and bounded memory integration."""

import unittest

from neuromem_core import _bounded_pair_text, entry_text, merge_trace_records, remove_instance_values
from neuromem_policy import NeuroMemPolicy, version_id
from tests.test_memory_policy import TinyScorer


class TraceProcessingTests(unittest.TestCase):
    def test_tight_pair_bounds_may_keep_only_the_right_side_or_a_marker(self):
        for limit, expected in ((0, ""), (1, "…"), (2, "x…")):
            with self.subTest(limit=limit):
                result = _bounded_pair_text("abc", "xyz", limit)
                self.assertEqual(result, expected)
                self.assertLessEqual(len(result), limit)

    def test_pair_bounding_retains_both_sides_when_space_permits(self):
        self.assertEqual(_bounded_pair_text("abc", "xyz", 8), "abc\n\nxyz")
        self.assertEqual(_bounded_pair_text("abc", "xyz", 6), "a…\n\nx…")
        self.assertEqual(_bounded_pair_text("abc", "abc", 8), "abc")
        self.assertEqual(_bounded_pair_text("abc", "bc", 8), "abc")

    def test_masking_matches_complete_numbers_in_every_trace_field(self):
        value = "12 123 312 12.5 0.12 2.5 2.50 12. patient12 schema_12 ICD9"
        expected = "<VALUE> 123 312 12.5 0.12 <VALUE> 2.50 <VALUE>. patient12 schema_12 ICD9"
        fields = ("knowledge", "applicability", "reasoning_pattern", "evidence", "cautions")
        trace = {field: value for field in fields}
        trace["composition_keys"] = [value]
        result = remove_instance_values(trace, "Patient 12, measurement 2.5")
        for field in fields:
            self.assertEqual(result[field], expected)
        self.assertEqual(result["composition_keys"], [expected])
        self.assertEqual(trace["knowledge"], value)

    def test_schema_identifier_suffixes_are_not_source_values(self):
        trace = {"knowledge": "2 and 12", "composition_keys": ["2", "12"]}
        result = remove_instance_values(trace, "Use table12 and schema_2")
        self.assertEqual(result, trace)

    def test_decimal_suffixes_are_not_independent_source_values(self):
        trace = {"knowledge": "12.5 12 5 112.5 12.50", "composition_keys": []}
        result = remove_instance_values(trace, "measurement 12.5")
        self.assertEqual(result["knowledge"], "<VALUE> 12 5 112.5 12.50")

    def bounded_anchor(self):
        anchor = {"mem_id": "anchor", "knowledge": "abcdefgh", "applicability": "",
                  "reasoning_pattern": "abcdefgh", "composition_keys": ["shared", ""],
                  "evidence": "", "cautions": "", "confidence": "low"}
        anchor["composition_keys"][1] = "x" * (8192 - len(entry_text(anchor)))
        self.assertEqual(len(entry_text(anchor)), 8192)
        return anchor

    def test_discarded_novel_key_does_not_make_merge_admissible(self):
        anchor = self.bounded_anchor()
        candidate = dict(anchor, composition_keys=["shared", "novel"])
        candidate.pop("mem_id")
        merged, _ = merge_trace_records(anchor, candidate)
        self.assertEqual(entry_text(merged), entry_text(anchor))
        policy = NeuroMemPolicy(TinyScorer(), review_delay=0, policy_mode="static")
        policy.add_candidate("candidate", candidate, 1.0, 0.25, 0)
        memory = [anchor]
        events = policy.review_and_apply(memory, 0)
        self.assertFalse(events[0]["admissible_actions"]["MERGE"])
        self.assertEqual(events[0]["action"], "RETAIN")
        self.assertEqual(memory[0]["mem_id"], version_id("anchor", 0))
        self.assertEqual(len(policy.version_lineage), 1)

    def test_compaction_alone_does_not_count_as_retained_novelty(self):
        anchor = self.bounded_anchor()
        anchor["knowledge"] = "abcdefgh " * 10
        # Only the suffix is new; the bound removes it and retains old text.
        candidate = dict(anchor, knowledge=anchor["knowledge"] + " novel suffix")
        policy = NeuroMemPolicy(TinyScorer())
        compatibility = policy.merge_compatibility(candidate, [anchor])
        self.assertEqual(compatibility["compatible_targets"], [])

    def test_retained_novel_content_allows_merge_within_the_bound(self):
        anchor = self.bounded_anchor()
        anchor["composition_keys"] = ["shared"]
        candidate = dict(anchor, knowledge="Reusable new operator guidance.", confidence="high")
        policy = NeuroMemPolicy(TinyScorer(), review_delay=0, policy_mode="static")
        policy.add_candidate("candidate", candidate, 1.0, 0.25, 0)
        memory = [anchor]
        events = policy.review_and_apply(memory, 0)
        self.assertEqual(events[0]["action"], "MERGE")
        self.assertIn(candidate["knowledge"], memory[0]["knowledge"])
        self.assertEqual(memory[0]["mem_id"], version_id("anchor", 1))
        self.assertLessEqual(len(entry_text(memory[0])), policy.merge_max_chars)
