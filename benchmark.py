"""Local benchmark resources and the EHRAgent tool interface for MIMIC-III."""
import ast
from contextlib import closing
from pathlib import Path
import sqlite3

from tools import calculator, tabtools
from runtime import sha256_file

# The four benchmark demonstrations live in the upstream resource, outside B/M.
PROMPTS_SHA256 = "5b34b5e160be74c685f36a79000485be0f165bc3ce5961ff8ee4332fc05b26d2"


def load_prompts(path):
    path = Path(path).expanduser()
    if not path.is_file() or sha256_file(path) != PROMPTS_SHA256:
        raise ValueError("Supply the pinned EHRAgent prompts_mimic.py from the README.")
    fields = {"EHRAgent_4Shots_Knowledge", "EHRAgent_Message_Prompt", "RetrKnowledge", "CodeDebugger"}
    prompts = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in fields:
                prompts[target.id] = ast.literal_eval(node.value)
    if prompts.keys() != fields or not all(isinstance(v, str) for v in prompts.values()):
        raise ValueError("Incomplete benchmark prompt resource")
    return prompts


def check_tables(data_root):
    root = Path(data_root).expanduser().resolve()
    tables = root / "mimic_iii"
    needed = [tables / name for name in tabtools.TABLE_FILES.values()]
    needed.append(tables / "mimic_iii.db")
    missing = [str(item) for item in needed if not item.is_file()]
    if missing:
        raise FileNotFoundError("Missing benchmark files: " + ", ".join(missing))
    with closing(sqlite3.connect(needed[-1].as_uri() + "?mode=ro", uri=True)) as con:
        present = {row[0].lower() for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    absent = sorted(set(tabtools.TABLE_FILES) - present)
    if absent:
        raise ValueError("Missing SQLite tables: " + ", ".join(absent))
    return root


def execute_ehr_code(cell):
    missing = object()
    scope = {"answer": missing, "tabtools": tabtools, "calculator": calculator,
             "Calculate": calculator.WolframAlphaCalculator, "LoadDB": tabtools.db_loader,
             "FilterDB": tabtools.data_filter, "GetValue": tabtools.get_value,
             "SQLInterpreter": tabtools.sql_interpreter, "Calendar": tabtools.date_calculator}
    try:
        exec(str(cell), scope)
    except Exception as error:
        return 1, f"Error: {type(error).__name__}: {error}"
    answer = scope.get("answer", missing)
    if answer is missing:
        return 1, "Error: Please save the final result to the variable 'answer'."
    return 0, str(answer)
