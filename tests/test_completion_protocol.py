import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from autogen import Agent, ConversableAgent

from benchmark import execute_ehr_code
from ehr_adapter import MedAgent
from evaluation import extract_prediction
from llm_config import EHR_SYSTEM_MESSAGE
from run_ehrsql import _run_agent_turn_with_support, _terminal


def python_call(cell):
    return {"content": "", "function_call": {"name": "python", "arguments": json.dumps({"cell": cell})}}


class CompletionProtocolTests(unittest.TestCase):
    def rollout(self, replies):
        """Exercise AutoGen's real synchronous chat loop without a model call."""
        pending = iter(replies)
        observed = []

        def scripted_reply(agent, messages, sender, config):
            observed.append(messages[-1])
            return True, next(pending)  # Any unexpected extra model turn fails the test.

        with TemporaryDirectory() as work:
            executor = Mock(wraps=execute_ehr_code)
            proxy = MedAgent(
                name="fixture_proxy", is_termination_msg=_terminal, human_input_mode="NEVER",
                max_consecutive_auto_reply=10, code_execution_config={"work_dir": work, "use_docker": False},
                config_list=[{}],
            )
            proxy.configure_ehr_code_executor(executor)
            proxy.register_function(function_map={"python": executor})
            proxy.benchmark_prompts = {"EHRAgent_Message_Prompt": "{examples}\n{knowledge}\n{question}"}
            proxy.update_memory(0, [])
            assistant = ConversableAgent(name="fixture_assistant", llm_config=False,
                                         system_message=EHR_SYSTEM_MESSAGE, human_input_mode="NEVER")
            assistant.register_reply(Agent, scripted_reply)
            with patch.object(proxy, "retrieve_knowledge", return_value="Synthetic fixture") as retrieval, \
                    patch.object(proxy, "error_debugger", return_value="Correct the complete solution"), \
                    patch("run_ehrsql.time.sleep", side_effect=AssertionError("Unexpected whole-task retry")):
                rows = _run_agent_turn_with_support(proxy, assistant, "Synthetic question", "")
            retrieval.assert_called_once()
            self.assertEqual(next(pending, None), None)
            self.assertEqual(proxy.chat_messages[assistant][-1]["content"], "TERMINATE")
            transcript = "\n----------------------------------------------------------\n".join(rows)
            return proxy, executor, observed, transcript

    def test_terminal_is_a_standalone_non_tool_message(self):
        for content in ("TERMINATE", "\n TERMINATE \n"):
            self.assertTrue(_terminal({"content": content}))
        for message in (
            {}, {"content": "do not TERMINATE"}, {"content": "```python\nTERMINATE\n```"},
            python_call("TERMINATE"),
            {"content": "TERMINATE", "function_call": {"name": "python", "arguments": "{}"}},
            {"content": "TERMINATE", "tool_calls": [{"id": "pending"}]},
        ):
            self.assertFalse(_terminal(message))

    def test_success_then_plain_terminate_preserves_raw_answer(self):
        proxy, executor, observed, transcript = self.rollout([python_call("answer = 'fixture'"), "TERMINATE"])
        self.assertEqual(executor.call_count, 1)
        self.assertEqual(len(observed), 2)
        self.assertIn("Saved answer: fixture", observed[-1]["content"])
        self.assertIn("plain assistant text", observed[-1]["content"])
        self.assertEqual(proxy.last_executed_answer, "fixture")
        self.assertEqual(extract_prediction(transcript), "fixture")

    def test_function_code_is_recorded_with_any_assistant_content(self):
        for content in (None, "", "Run the complete solution."):
            with self.subTest(content=content):
                cell = "answer = 3"
                message = python_call(cell)
                message["content"] = content
                _, executor, _, transcript = self.rollout([message, "TERMINATE"])
                executor.assert_called_once_with(cell=cell)
                self.assertIn(cell, transcript)
                if content:
                    self.assertIn(content, transcript)
                self.assertEqual(extract_prediction(transcript), "3")

    def test_success_allows_a_complete_correction(self):
        proxy, executor, _, transcript = self.rollout([
            python_call("answer = 1"), python_call("answer = 2"), "TERMINATE",
        ])
        self.assertEqual(executor.call_count, 2)
        self.assertEqual(proxy.last_executed_answer, "2")
        self.assertEqual(extract_prediction(transcript), "2")

    def test_later_failure_clears_the_answer_before_termination(self):
        for failing_cell in ("print(answer)", "TERMINATE", "answer = 1; del answer"):
            with self.subTest(cell=failing_cell):
                proxy, executor, observed, transcript = self.rollout([
                    python_call("answer = 'fixture'"), python_call(failing_cell), "TERMINATE",
                ])
                self.assertEqual(executor.call_count, 2)
                self.assertIn("saved answer has been cleared", observed[-1]["content"])
                self.assertIsNone(proxy.last_executed_answer)
                self.assertIsNone(proxy.last_execution_source)
                self.assertEqual(extract_prediction(transcript), "")

    def test_failure_can_recover_with_a_complete_solution(self):
        proxy, executor, _, transcript = self.rollout([
            python_call("answer = 1; raise ValueError('failure')"),
            python_call("answer = 2"), "TERMINATE",
        ])
        self.assertEqual(executor.call_count, 2)
        self.assertEqual(proxy.last_executed_answer, "2")
        self.assertEqual(extract_prediction(transcript), "2")

    def test_code_block_then_plain_terminate_preserves_the_answer(self):
        proxy, executor, _, transcript = self.rollout(["```python\nanswer = 3\n```", "TERMINATE"])
        self.assertEqual(executor.call_count, 1)
        self.assertEqual(proxy.last_execution_source, "python_code_block")
        self.assertEqual(extract_prediction(transcript), "3")

    def test_deleted_answer_in_code_block_is_feedback_not_a_whole_task_retry(self):
        replies = [python_call("answer = 'previous'"), "```python\nanswer = 1; del answer\n```"]
        for recover in (False, True):
            with self.subTest(recover=recover):
                correction = ["```python\nanswer = 2\n```"] if recover else []
                proxy, executor, observed, transcript = self.rollout(replies + correction + ["TERMINATE"])
                self.assertEqual(executor.call_count, 3 if recover else 2)
                self.assertIn("exitcode: 1 (execution failed)", observed[2]["content"])
                self.assertIn("Please save the final result to the variable 'answer'.", observed[2]["content"])
                self.assertNotIn("KeyError", observed[2]["content"])
                self.assertEqual(proxy.last_executed_answer, "2" if recover else None)
                self.assertEqual(extract_prediction(transcript), "2" if recover else "")
