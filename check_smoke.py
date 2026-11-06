"""Check a completed smoke run for evidence of execution, independently of accuracy."""
import argparse
import json
from pathlib import Path

from evaluation import extract_prediction


def check_smoke(output):
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    final = manifest.get("final", {})
    count = manifest.get("task_count", 0)
    if (manifest.get("mode") != "smoke" or count < 1 or final.get("tasks") != count
            or final.get("open_decisions") != 0 or "completed_at_unix" not in manifest):
        raise ValueError("Expected a completed smoke run with every decision finalized.")
    rows = [json.loads(line) for line in (output / "evaluator_records.jsonl").read_text().splitlines()]
    if len(rows) != count or len({r["task_id"] for r in rows}) != count:
        raise ValueError("Incomplete or duplicated smoke evaluation records.")
    for row in rows:
        transcript = (output / row["prediction_path"]).read_text(encoding="utf-8")
        if any(line.startswith("Prediction-side rollout error:") for line in transcript.splitlines()):
            raise ValueError(f'{row["prediction_path"]}: rollout error; inspect the provider and execution logs.')
        messages = transcript.split("\n----------------------------------------------------------\n")
        block_executed = any(msg.startswith("exitcode: 0 (execution succeeded)\n") for msg in messages[:-1])
        function_executed = row.get("final_execution_source") == "python_function"
        if (not (block_executed or function_executed)
                or not messages[-1].startswith("Final Executed Answer ---> ")
                or not extract_prediction(transcript)):
            raise ValueError(f'{row["prediction_path"]}: expected a successful execution and a nonempty final answer.')
    print(f"Smoke wiring passed for {count} tasks. Benchmark successes: {final['successes']}/{count}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        check_smoke(args.output)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Smoke check failed: {error}\n")
