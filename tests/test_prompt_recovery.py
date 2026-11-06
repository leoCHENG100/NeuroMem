"""Check structured-output recovery without making provider requests."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from neuromem_prompts import OpenAIJSONPromptClient, build_ttk_candidate
from path_composition import compose_with_recovery


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class PromptRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.provider = Mock()
        with patch("openai.OpenAI", return_value=self.provider):
            self.client = OpenAIJSONPromptClient({"model": "local-fixture", "api_key": "local-test-only"})
        self.create = self.provider.chat.completions.create
        self.selected = [{"mem_id": "first", "knowledge": "Aggregate rows."},
                         {"mem_id": "second", "knowledge": "Unpack the scalar."}]
        self.candidates = self.selected + [{"mem_id": "other", "knowledge": "Sort rows."}]

    def compose(self):
        return compose_with_recovery(self.client, query="Synthetic question", selected_traces=self.selected,
                                     candidate_traces=self.candidates)

    def test_invalid_json_and_invalid_schema_use_the_same_path_fallback(self):
        for content in ("not json", "{broken}", "[]", "{}", ""):
            with self.subTest(content=content):
                self.create.reset_mock()
                self.create.return_value = response(content)
                result = self.compose()
                self.assertEqual(self.create.call_count, 3)
                self.assertEqual(result["selected_path_ids"], ["first", "second"])
                self.assertEqual([x["memory_id"] for x in result["excluded_memories"]], ["other"])
                self.assertTrue(result["schema_validation"]["used_deterministic_immutable_path_fallback"])
                self.assertEqual(len(result["schema_validation"]["errors"]), 3)
                self.assertEqual(result["schema_validation"]["generation_attempts"], 3)
                self.assertEqual(result["schema_validation"]["generation_retries"], 2)
                self.assertNotIn("repair_requests", result["schema_validation"])
                requests = [call.kwargs["messages"] for call in self.create.call_args_list]
                self.assertTrue(all(request == requests[0] for request in requests))

    def test_generation_retry_counts_preserve_search_order_and_request_contents(self):
        payload = {
            "selected_support_path": [{"memory_id": x["mem_id"], "role": "support", "reason": "Relevant"}
                                      for x in self.selected],
            "excluded_memories": [{"memory_id": "other", "reason": "Outside selected path"}],
            "composition_rationale": "Use both traces.", "task_adaptation": "Execute the query.",
            "cautions": "Check the shape.", "composed_context": "Aggregate then unpack.", "confidence": "high",
        }
        for retries in range(3):
            with self.subTest(retries=retries):
                self.create.reset_mock()
                self.create.side_effect = [response("not json") for _ in range(retries)] + [response(json.dumps(payload))]
                result = self.compose()
                self.assertEqual(self.create.call_count, retries + 1)
                self.assertEqual(result["selected_path_ids"], ["first", "second"])
                self.assertFalse(result["schema_validation"]["used_deterministic_immutable_path_fallback"])
                self.assertEqual(result["schema_validation"]["generation_attempts"], retries + 1)
                self.assertEqual(result["schema_validation"]["generation_retries"], retries)
                self.assertNotIn("repair_requests", result["schema_validation"])
                requests = [call.kwargs["messages"] for call in self.create.call_args_list]
                self.assertTrue(all(request == requests[0] for request in requests))

    def test_provider_failure_propagates_without_a_validation_fallback(self):
        failure = ConnectionError("Synthetic provider unavailable")
        self.create.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            self.compose()
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(self.create.call_count, 3)

    def test_provider_retry_can_recover(self):
        self.create.side_effect = [ConnectionError("Synthetic outage"), response('{"ok": true}')]
        self.assertEqual(self.client.generate_json("system", "user", "fixture"), {"ok": True})
        self.assertEqual(self.create.call_count, 2)

    def test_empty_path_does_not_request_a_generation(self):
        result = compose_with_recovery(self.client, query="Synthetic question", selected_traces=[])
        self.assertFalse(result["invoked"])
        self.create.assert_not_called()

    def test_candidate_constructor_retries_invalid_json(self):
        payload = {"knowledge": "Aggregate rows.", "applicability": "Aggregation", "reasoning_pattern": "Run SQL.",
                   "composition_keys": ["aggregation"], "evidence": "Execution completed.",
                   "cautions": "Check shape.", "confidence": "high"}
        self.create.side_effect = [response("not json"), response(json.dumps(payload))]
        result = build_ttk_candidate(self.client, task_family="synthetic", query="Sum values", trajectory="Run SQL",
                                     selected_traces=[], task_reward=1.0, outcome_summary="Completed")
        self.assertEqual(result["knowledge"], payload["knowledge"])
        self.assertEqual(result["schema_validation"]["generation_attempts"], 2)
        self.assertEqual(self.create.call_count, 2)

    def test_candidate_constructor_does_not_invent_a_trace_from_invalid_json(self):
        self.create.return_value = response("not json")
        with self.assertRaisesRegex(RuntimeError, "transferable_trace/1"):
            build_ttk_candidate(self.client, task_family="synthetic", query="Sum values", trajectory="Run SQL",
                                selected_traces=[], task_reward=1.0, outcome_summary="Completed")
        self.assertEqual(self.create.call_count, 3)

    def test_candidate_invalid_confidence_uses_the_declared_low_fallback(self):
        payload = {"knowledge": "Aggregate rows.", "applicability": "Aggregation", "reasoning_pattern": "Run SQL.",
                   "composition_keys": ["aggregation"], "evidence": "Execution completed.",
                   "cautions": "Check shape.", "confidence": "unsupported"}
        self.create.return_value = response(json.dumps(payload))
        result = build_ttk_candidate(self.client, task_family="synthetic", query="Sum values", trajectory="Run SQL",
                                     selected_traces=[], task_reward=1.0, outcome_summary="Completed")
        self.assertEqual(result["confidence"], "low")
        self.assertTrue(result["schema_validation"]["used_low_confidence_fallback"])
        self.assertEqual(self.create.call_count, 3)
