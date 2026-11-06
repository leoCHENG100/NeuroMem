import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark import check_tables, execute_ehr_code
from check_smoke import check_smoke
from ehr_adapter import MedAgent
from runtime import FixedDemonstrationSelector
from tools.calculator import calculator_mode
from tools.tabtools import TABLE_FILES


class ReleaseContractTests(unittest.TestCase):
    def test_preflight_checks_files_and_readonly_table_names_not_csv_contents(self):
        with TemporaryDirectory() as tmp:
            tables = Path(tmp) / "mimic_iii"
            tables.mkdir()
            with sqlite3.connect(tables / "mimic_iii.db") as db:
                for table, filename in TABLE_FILES.items():
                    db.execute(f'CREATE TABLE "{table}" (n INTEGER)')
                    (tables / filename).write_bytes(b"\xff")
            with patch("benchmark.sqlite3.connect", wraps=sqlite3.connect) as connect:
                self.assertEqual(check_tables(tmp), Path(tmp).resolve())
                self.assertTrue(connect.call_args.args[0].endswith("?mode=ro"))
                self.assertTrue(connect.call_args.kwargs["uri"])
            (tables / next(iter(TABLE_FILES.values()))).unlink()
            with self.assertRaises(FileNotFoundError):
                check_tables(tmp)

    def test_fixed_demonstration_metadata_describes_the_actual_selector(self):
        agent = MedAgent(name="fixture", code_execution_config=False)
        agent.memory_selector = FixedDemonstrationSelector()
        examples = [{"question": f"Q{i}", "knowledge": "Synthetic knowledge", "code": "answer = 3"}
                    for i in range(4)]
        agent.update_memory(4, examples)
        text = agent.retrieve_examples("Q3")
        self.assertEqual(agent.last_selected_indexes, [0, 1, 2, 3])
        self.assertLess(text.index("Q0"), text.index("Q3"))
        self.assertEqual(agent.last_memory_selection_info["strategy"], "fixed_benchmark_demonstrations")
        self.assertEqual(set(agent.last_memory_selection_info), {
            "strategy", "configured_strategy", "fallback_used", "elapsed_sec", "memory_size",
            "num_shots", "selected_count",
        })

    def test_deleted_answer_returns_missing_answer_failure(self):
        expected = (1, "Error: Please save the final result to the variable 'answer'.")
        for cell in ("value = 1", "del answer", "answer = 1; del answer",
                     "answer = 1; globals().pop('answer')"):
            with self.subTest(cell=cell):
                self.assertEqual(execute_ehr_code(cell), expected)
        self.assertEqual(execute_ehr_code("answer = 1; del answer; answer = 0"), (0, "0"))

    def test_calculate_returns_disabled_feedback_without_service_calls(self):
        expected = (1, "Error: Exception: Invalid input query for Calculator. "
                       "Please check the input query or use other functions to do the computation.")
        with patch.dict(os.environ, {}, clear=True), patch("wolframalpha.Client") as service:
            for app_id in ("", "unused-test-id"):
                os.environ["WOLFRAMALPHA_APP_ID"] = app_id
                self.assertEqual(calculator_mode(), "offline")
                self.assertEqual(execute_ehr_code('answer = Calculate("1+1")'), expected)
            service.assert_not_called()

    def test_calculate_live_mode_requires_explicit_opt_in_and_app_id(self):
        with patch.dict(os.environ, {"NEUROMEM_ENABLE_WOLFRAM": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "WOLFRAMALPHA_APP_ID"):
                calculator_mode()
            os.environ["WOLFRAMALPHA_APP_ID"] = "unused-test-id"
            with patch("wolframalpha.Client") as service:
                service.return_value.query.return_value.results = iter([SimpleNamespace(text="2")])
                self.assertEqual(calculator_mode(), "live_wolfram")
                self.assertEqual(execute_ehr_code('answer = Calculate("1+1")'), (0, "2"))
                service.return_value.query.assert_called_once_with("1+1")

    def test_sql_tool_can_read_but_cannot_modify_the_external_database(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"EHRSQL_DATA_ROOT": tmp}):
            tables = Path(tmp) / "mimic_iii"
            tables.mkdir()
            db = sqlite3.connect(tables / "mimic_iii.db")
            db.execute("CREATE TABLE fixture (n INTEGER)")
            db.close()
            result = execute_ehr_code("answer = SQLInterpreter('DROP TABLE fixture')")
            self.assertEqual(result[0], 1)
            self.assertIn("readonly", result[1].lower())
            self.assertEqual(execute_ehr_code("answer = SQLInterpreter('SELECT COUNT(*) FROM fixture')[0][0]"),
                             (0, "0"))

    def test_smoke_requires_execution_and_marker_even_when_accuracy_is_zero(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"mode": "smoke", "task_count": 1, "completed_at_unix": 1,
                        "final": {"tasks": 1, "open_decisions": 0, "successes": 0}}
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            prediction = root / "prediction.txt"
            row = {"task_id": "synthetic", "prediction_path": prediction.name}
            (root / "evaluator_records.jsonl").write_text(json.dumps(row) + "\n")
            execution = "exitcode: 0 (execution succeeded)\nCode output: 4\n"
            marker = "Final Executed Answer ---> 4"
            prediction.write_text(execution + "\n----------------------------------------------------------\n" + marker)
            check_smoke(root)
            for transcript in (execution, marker, "Prediction-side rollout error: simulated"):
                prediction.write_text(transcript)
                with self.assertRaises(ValueError):
                    check_smoke(root)

    def test_function_smoke_requires_saved_answer_and_execution_source(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"mode": "smoke", "task_count": 1, "completed_at_unix": 1,
                        "final": {"tasks": 1, "open_decisions": 0, "successes": 0}}
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            prediction = root / "prediction.txt"
            records = root / "evaluator_records.jsonl"
            row = {"task_id": "synthetic", "prediction_path": prediction.name,
                   "final_execution_source": "python_function"}
            records.write_text(json.dumps(row) + "\n")
            separator = "\n----------------------------------------------------------\n"
            successful = separator.join(["4", "TERMINATE", "Final Executed Answer ---> 4"])
            prediction.write_text(successful)
            check_smoke(root)
            for transcript in (
                separator.join(["4", "TERMINATE"]),
                separator.join(["Error: simulated later failure", "TERMINATE"]),
                separator.join(["TERMINATE", "Final Executed Answer ---> "]),
                "Prediction-side rollout error: simulated" + separator + successful,
            ):
                with self.subTest(transcript=transcript):
                    prediction.write_text(transcript)
                    with self.assertRaises(ValueError):
                        check_smoke(root)
            prediction.write_text(successful)
            row["final_execution_source"] = None
            records.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                check_smoke(root)
