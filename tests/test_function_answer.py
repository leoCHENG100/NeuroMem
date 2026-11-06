import json
import unittest
from unittest.mock import patch

from benchmark import execute_ehr_code
from ehr_adapter import MedAgent


class FunctionAnswerTests(unittest.TestCase):
    def setUp(self):
        self.agent = MedAgent(name="fixture", code_execution_config=False)
        self.agent.configure_ehr_code_executor(execute_ehr_code)
        self.agent.register_function(function_map={"python": execute_ehr_code})

    def call(self, cell):
        return self.agent.execute_function({"name": "python", "arguments": json.dumps({"cell": cell})})

    def test_latest_function_execution_wins(self):
        for cell, success, expected in [
            ("answer = 0", True, "0"),
            ("answer = 2; raise ValueError('failed')", False, None),
            ("answer = 'Error label'", True, "Error label"),
            ("unassigned = 1", False, None),
            ("answer = 'final'", True, "final"),
        ]:
            with self.subTest(cell=cell):
                ok, reply = self.call(cell)
                self.assertEqual(ok, success)
                self.assertEqual(self.agent.last_executed_answer, expected)
                self.assertEqual(self.agent.last_execution_source, "python_function" if success else None)
                if success:
                    self.assertTrue(reply["content"].startswith("exitcode: 0 (execution succeeded)\n"))
                    self.assertIn("Saved answer: " + expected + "\n", reply["content"])
                    self.assertIn("plain assistant text", reply["content"])
                else:
                    self.assertTrue(reply["content"].startswith("exitcode: 1 (execution failed)\n"))
                    self.assertIn("saved answer has been cleared", reply["content"])
        self.agent.reset_execution_answer()
        self.assertIsNone(self.agent.last_executed_answer)

    def test_code_blocks_and_function_calls_share_one_answer(self):
        self.assertTrue(self.call("answer = 'function'")[0])
        self.assertNotEqual(self.agent.execute_code_blocks([("python", "value = 1")])[0], 0)
        self.assertIsNone(self.agent.last_executed_answer)
        self.assertEqual(self.agent.execute_code_blocks([("python", "answer = 'block'")])[0], 0)
        self.assertEqual(self.agent.last_execution_source, "python_code_block")
        for bad_args in ("{}", "null"):
            self.assertFalse(self.agent.execute_function({"name": "python", "arguments": bad_args})[0])
            self.assertIsNone(self.agent.last_executed_answer)
        self.assertTrue(self.call("answer = 'recovered'")[0])
        self.assertEqual(self.agent.last_executed_answer, "recovered")

    def test_deleted_answer_clears_saved_state_in_both_execution_paths(self):
        for route in ("function", "code_block"):
            with self.subTest(route=route):
                self.assertTrue(self.call("answer = 'previous'")[0])
                if route == "function":
                    ok, reply = self.call("answer = 1; del answer")
                    self.assertFalse(ok)
                    output = reply["content"]
                else:
                    exitcode, output = self.agent.execute_code_blocks([("python", "answer = 1; del answer")])
                    self.assertEqual(exitcode, 1)
                self.assertIn("Please save the final result to the variable 'answer'.", output)
                self.assertNotIn("KeyError", output)
                self.assertIsNone(self.agent.last_executed_answer)
                self.assertIsNone(self.agent.last_execution_source)
                self.assertTrue(self.call("answer = 'recovered'")[0])
                self.assertEqual(self.agent.last_executed_answer, "recovered")

    def test_successful_error_labels_do_not_trigger_diagnosis(self):
        self.agent.config_list = [{}]
        with patch.object(self.agent, "error_debugger") as diagnose:
            for answer in ("error", "Error label", "no errors"):
                with self.subTest(answer=answer):
                    ok, reply = self.call(f"answer = {answer!r}")
                    self.assertTrue(ok)
                    self.assertEqual(self.agent.last_executed_answer, answer)
                    self.assertNotIn("Potential Reasons:", reply["content"])
            diagnose.assert_not_called()

    def test_failed_status_triggers_diagnosis_without_an_error_keyword(self):
        self.agent.config_list = [{}]
        self.assertTrue(self.call("answer = 'previous'")[0])
        self.agent.register_function(function_map={"python": lambda cell: (1, "invalid query")})
        with patch.object(self.agent, "error_debugger", return_value="Check the query") as diagnose:
            ok, reply = self.call("answer = 1")
            self.assertFalse(ok)
            diagnose.assert_called_once_with({}, "answer = 1", "invalid query")
            self.assertIn("Potential Reasons: Check the query", reply["content"])
            self.assertIsNone(self.agent.last_executed_answer)
            self.assertIsNone(self.agent.last_execution_source)
