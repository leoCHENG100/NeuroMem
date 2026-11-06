"""Exercise the public CLI against a local, scripted chat-completions endpoint."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread

from check_smoke import check_smoke
from tools.tabtools import TABLE_FILES


SMOKE_TASK_COUNT = 48


def check_delayed_updates(rows, final, credit_horizon):
    """Require full-horizon updates during the stream and a separate final flush."""

    settled = [decision for row in rows for decision in row["finalized_decisions"]]
    stream_updates = max((row["adam_step"] for row in rows), default=0)
    if not settled or stream_updates < 1:
        raise AssertionError("No full-horizon policy update occurred during the task stream")
    if any(decision["elapsed"] < credit_horizon for decision in settled):
        raise AssertionError("A stream decision finalized before its credit horizon")
    if final["open_decisions"] != 0 or final["adam_step"] <= stream_updates:
        raise AssertionError("The final flush must settle the remaining open decisions")
    return {"stream_updates": stream_updates, "updates_after_flush": final["adam_step"]}


def reply(messages):
    system = messages[0]["content"]
    prompt = messages[-1]["content"] or ""
    if "You are P_ttk" in system:
        return json.dumps({"knowledge": "Aggregate using SQL.", "applicability": "Aggregation tasks",
                           "reasoning_pattern": "Query then unpack the scalar.", "composition_keys": ["aggregation"],
                           "evidence": "Query executed.", "cautions": "Check the result shape.", "confidence": "high"})
    if "You are P_path" in system:
        def trace_list(label):
            return json.JSONDecoder().raw_decode(prompt.split(label, 1)[1].lstrip())[0]
        selected = trace_list("path is ordered and immutable:")
        others = trace_list("traces (not in the selected path):")
        return json.dumps({"selected_support_path": [{"memory_id": x["mem_id"], "role": "support",
                           "reason": "Reuse aggregation."} for x in selected],
                           "excluded_memories": [{"memory_id": x["mem_id"], "reason": "Outside path"} for x in others],
                           "composition_rationale": "Use the selected path.", "task_adaptation": "Run the query.",
                           "cautions": "Unpack the result.", "composed_context": "Use SQL aggregation.", "confidence": "high"})
    if "only use the functions" in system:
        if "execution succeeded" in prompt:
            return "TERMINATE"
        return "```python\nanswer = SQLInterpreter('SELECT SUM(n) FROM patients')[0][0]\n```"
    return "Use a scalar aggregate."


class LocalProvider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = json.dumps({"id": "local-check", "object": "chat.completion", "created": 1,
                              "model": request["model"], "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": reply(request["messages"])}}],
                              "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def smoke(root, prompts, model_dir, endpoint, device="cpu"):
    tables = root / "mimic_iii"
    tables.mkdir()
    with sqlite3.connect(tables / "mimic_iii.db") as db:
        for table, filename in TABLE_FILES.items():
            db.execute(f'CREATE TABLE "{table}" (n INTEGER)')
            (tables / filename).write_text("n\n", encoding="utf-8")
        db.execute("INSERT INTO patients VALUES (1), (2)")
    tasks = [{"id": f"synthetic-{i}", "question": f"Sum the sample values for check {i}.", "answer": ["3"]}
             for i in range(SMOKE_TASK_COUNT)]
    split = root / "tasks.json"
    split.write_text(json.dumps(tasks), encoding="utf-8")
    output = root / "output"
    runner = Path(__file__).resolve().parents[1] / "run_ehrsql.py"
    env = dict(os.environ, OPENAI_API_KEY="local-test-only", OPENAI_BASE_URL=endpoint,
               HF_HUB_OFFLINE="1", NEUROMEM_ENABLE_WOLFRAM="0",
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
               OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            del env[key]
    run = subprocess.run([sys.executable, str(runner), "--data-root", str(root), "--tasks", str(split),
                          "--model-dir", str(model_dir), "--prompts", str(prompts), "--device", device,
                          "--llm", "local-fixture", "--smoke", str(len(tasks)), "--output", str(output)],
                         cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
    if run.returncode:
        raise RuntimeError(run.stdout[-3500:])
    check_smoke(output)
    manifest = json.loads((output / "run_manifest.json").read_text())
    if device != "auto" and manifest["relation"]["device"] != device:
        raise AssertionError("CLI did not use the explicitly requested encoder device")
    final = manifest["final"]
    rows = [json.loads(line) for line in (output / "task_records.jsonl").read_text().splitlines()]
    if (final["successes"] != len(tasks) or final["open_decisions"] != 0 or final["adam_step"] < 1
            or not any(x["p_path"]["invoked"] for x in rows) or manifest["method"] != "NeuroMem"
            or manifest["calculate_mode"] != "offline"):
        raise AssertionError(f"CLI smoke failed: {final}")
    updates = check_delayed_updates(rows, final, manifest["settings"]["credit_horizon"])
    print(f"Policy update coverage: {updates}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    with TemporaryDirectory() as folder, ThreadingHTTPServer(("127.0.0.1", 0), LocalProvider) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            smoke(Path(folder), args.prompts.resolve(), args.model_dir.resolve(),
                  f"http://127.0.0.1:{server.server_port}/v1", device=args.device)
        finally:
            server.shutdown()
            worker.join()
    print("Offline CLI smoke passed: SQL execution, retrieval, evaluation, full-horizon updates, final flush.")


if __name__ == "__main__":
    main()
