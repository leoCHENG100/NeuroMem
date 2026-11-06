"""Run NeuroMem on an externally supplied EHRSQL/MIMIC-III stream."""
import argparse
import json
import os
from pathlib import Path
import pickle
import time

from benchmark import check_tables, execute_ehr_code, load_prompts
from evaluation import extract_prediction, judge, set_seed
from llm_config import EHR_SYSTEM_MESSAGE, agent_config, task_config
from neuromem_core import BudgetedSupportPathSelector, LocalBGERelationScorer, assert_prediction_only, confidence_value
from neuromem_prompts import OpenAIJSONPromptClient, build_ttk_candidate, prompt_manifest
from neuromem_policy import POLICY_NAME, SUCCESSOR_IDENTITY_MODE, NeuroMemPolicy
from path_composition import compose_with_recovery
from tools.calculator import calculator_mode
from runtime import (FixedDemonstrationSelector, active_memory, append_jsonl, memory_size_stats, model_directory_manifest,
                     select_path_and_credit, sha256_file, sha256_json, write_json)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Directory containing mimic_iii/.")
    parser.add_argument("--tasks", required=True, type=Path, help="Ordered preprocessed JSON split with answers.")
    parser.add_argument("--prompts", required=True, type=Path, help="Pinned upstream prompts_mimic.py.")
    parser.add_argument("--model-dir", required=True, type=Path, help="Local BAAI/bge-m3 directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--llm", default="gemini-3-flash-preview", help="OpenAI-compatible model identifier.")
    parser.add_argument("--output", type=Path, help="New directory for this run.")
    parser.add_argument("--smoke", type=int, metavar="N", help="Run only the first N tasks as a smoke check.")
    parser.add_argument("--check-inputs", action="store_true", help="Validate external inputs and load the encoder, then exit.")
    args = parser.parse_args()
    if not args.check_inputs and args.output is None:
        parser.error("--output is required for a run")
    return args


def load_tasks(path, limit):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Task file must contain a nonempty JSON list.")
    if limit is None and len(rows) != 581:
        raise ValueError("Full EHRSQL mode requires 581 ordered tasks. Use --smoke N for a subset.")
    if limit is not None:
        if not 1 <= limit <= len(rows):
            raise ValueError("--smoke must select between one task and the size of the split.")
        rows = rows[:limit]
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("id") or "").strip() or "answer" not in row:
            raise ValueError("Each task requires an id and an answer.")
        question = row.get("template") or row.get("question")
        if not isinstance(question, str) or not question.strip() or row["answer"] is None:
            raise ValueError("Each task requires a question/template and a reference answer.")
        task_id = str(row["id"])
        if task_id in ids:
            raise ValueError("Duplicate task id in ordered stream.")
        ids.add(task_id)
    return rows


def _terminal(message):
    message = message or {}
    return (str(message.get("content") or "").strip() == "TERMINATE"
            and not message.get("function_call") and not message.get("tool_calls"))



def _fixed_ehr_demonstrations(example_text) -> list[dict]:
    """Keep the benchmark's four fixed examples outside NeuroMem B/M."""

    result = []
    source_examples = example_text.split("\n\n")
    if len(source_examples) != 4:
        raise ValueError("EHRAgent benchmark prompt must contain exactly four fixed demonstrations")
    for index, item in enumerate(source_examples):
        chunk = item.split("Question:")[-1]
        question = chunk.split("\nKnowledge:\n")[0]
        chunk = chunk.split("\nKnowledge:\n")[-1]
        knowledge = chunk.split("\nSolution:")[0]
        code = chunk.split("\nSolution:")[-1]
        if not question.strip() or not knowledge.strip() or not code.strip():
            raise ValueError(f"Malformed fixed EHRAgent demonstration at index {index}")
        result.append({
            "mem_id": f"benchmark_prompt_demo_{index}",
            "question": question,
            "knowledge": knowledge,
            "code": code,
            "source": "fixed_benchmark_prompt_not_neuromem_memory",
        })
    return result


def _run_agent_turn_with_support(user_proxy, chatbot, question: str, support_context: str) -> list[str]:
    """Supply composed support context through the benchmark knowledge section."""

    last_error = None
    for attempt in range(1, 5):
        try:
            if hasattr(user_proxy, "reset_execution_answer"):
                user_proxy.reset_execution_answer()
            benchmark_knowledge = user_proxy.retrieve_knowledge(user_proxy.config_list[0], question)
            combined_knowledge = str(benchmark_knowledge or "").strip()
            if str(support_context or "").strip():
                combined_knowledge += (
                    "\n\nNeuroMem selected support-path guidance:\n" + str(support_context).strip()
                )
            user_proxy.initiate_chat(
                chatbot,
                message=question,
                fixed_knowledge=combined_knowledge,
            )
            logs = user_proxy._oai_messages
            rows = [str(question)]
            for agent in list(logs.keys()):
                for message in logs[agent]:
                    content = message.get("content")
                    if content is not None:
                        rows.append(str(content))
                    function_call = message.get("function_call")
                    if function_call:
                        arguments = function_call.get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                pass
                        if isinstance(arguments, dict):
                            arguments = arguments.get("cell", json.dumps(arguments, ensure_ascii=False))
                        rows.append(str(arguments))
            executed_answer = getattr(user_proxy, "last_executed_answer", None)
            if executed_answer is not None:
                rows.append("Final Executed Answer ---> " + str(executed_answer))
            return rows
        except Exception as error:
            last_error = error
            if attempt < 4:
                time.sleep(10 * attempt)
                continue
            raise
    return [str(last_error)]


def main():
    args = arguments()
    tasks = load_tasks(args.tasks, args.smoke)
    prompts = load_prompts(args.prompts)
    fixed_prompt_demonstrations = _fixed_ehr_demonstrations(prompts["EHRAgent_4Shots_Knowledge"])
    os.environ["EHRSQL_DATA_ROOT"] = str(check_tables(args.data_root))
    if not args.check_inputs:
        output_root = args.output.expanduser().resolve()
        if output_root.exists():
            raise FileExistsError(f"Choose a new output directory: {output_root}")
        config = task_config(args.llm)
        calculate_setting = calculator_mode()
    set_seed(0)
    scorer = LocalBGERelationScorer(args.model_dir, device=args.device)
    if args.check_inputs:
        scorer.encode("encoder readiness check")
        print(f"Inputs OK: {len(tasks)} tasks, four demonstrations, SQLite/CSV tables, local BGE-M3")
        return

    import autogen
    from ehr_adapter import MedAgent

    selector = BudgetedSupportPathSelector(scorer, expansion_budget=3072, candidate_pool_size=32)
    policy = NeuroMemPolicy(scorer, seed=0)
    persistent_memory = []
    prompt_client = OpenAIJSONPromptClient(config, max_tokens=1600, temperature=0.0)
    output_root.mkdir(parents=True, exist_ok=False)
    predictions_dir = output_root / "predictions"
    predictions_dir.mkdir()
    coding_dir = output_root / "coding"
    coding_dir.mkdir()
    chatbot = autogen.agentchat.AssistantAgent(
        name="chatbot",
        system_message=EHR_SYSTEM_MESSAGE,
        llm_config=agent_config(0, config),
    )
    user_proxy = MedAgent(name="user_proxy", is_termination_msg=_terminal, human_input_mode="NEVER",
                          max_consecutive_auto_reply=10, code_execution_config={"work_dir": str(coding_dir)},
                          config_list=[config])
    user_proxy.benchmark_prompts = prompts
    user_proxy.memory_strategy = FixedDemonstrationSelector.strategy
    user_proxy.memory_selector = FixedDemonstrationSelector()
    user_proxy.configure_ehr_code_executor(execute_ehr_code)
    user_proxy.register_function(function_map={"python": execute_ehr_code})

    task_ids = [str(row["id"]) for row in tasks]
    source_root = Path(__file__).resolve().parent
    manifest = {
        "method": POLICY_NAME, "successor_identity": SUCCESSOR_IDENTITY_MODE,
        "mode": "main" if args.smoke is None else "smoke", "task_count": len(tasks),
        "order": "dataset", "task_ids_sha256": sha256_json(task_ids), "tasks_sha256": sha256_file(args.tasks),
        "demonstrations_sha256": sha256_file(args.prompts), "prompt_modules": prompt_manifest(),
        "model": args.llm, "calculate_mode": calculate_setting,
        "semantic_model": model_directory_manifest(args.model_dir),
        "relation": scorer.metadata(), "policy_seed": 0, "llm_cache_seed": 0,
        "settings": {name: getattr(policy, name) for name in (
            "learning_rate", "gamma", "credit_horizon", "review_delay", "lambda_memory", "beta_redundancy",
            "pending_retrieval_threshold", "promotion_score_threshold", "promotion_min_utility",
            "max_pending_age", "merge_max_chars")},
        "search_budget": selector.expansion_budget, "candidate_pool_size": selector.candidate_pool_size,
        "source_sha256": {str(p.relative_to(source_root)): sha256_file(p) for p in
                          sorted([*source_root.glob("*.py"), *source_root.glob("tools/*.py")])},
        "created_at_unix": time.time(),
    }
    write_json(output_root / "run_manifest.json", manifest)
    records_path = output_root / "task_records.jsonl"
    evaluator_path = output_root / "evaluator_records.jsonl"
    started = time.time()
    successes = 0
    for turn, item in enumerate(tasks):
        turn_started = time.perf_counter()
        task_id = task_ids[turn]
        safe_id = task_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        question = item.get("template") or item.get("question")
        if not str(question or "").strip():
            raise ValueError(f"Task {task_id} has no template/question")
        question = str(question)

        # Retrieve and solve before updating memory. Lifecycle actions
        # at review step t affect the next interaction t+1.
        retrieval_memory = active_memory(persistent_memory, policy)
        selected_indexes, selected_traces, path_credit, successor_credit = select_path_and_credit(
            query=question,
            memory=retrieval_memory,
            selector=selector,
            credit_mode="prefix",
        )
        composition = compose_with_recovery(
            prompt_client,
            query=question,
            selected_traces=selected_traces,
            candidate_traces=[
                row for row in retrieval_memory if str(row.get("mem_id", "")) in set(selector.last_candidate_ids)
            ],
            task_context="EHRSQL executable clinical database task using the benchmark EHRAgent tools and output format.",
        )
        user_proxy.update_memory(len(fixed_prompt_demonstrations), fixed_prompt_demonstrations)

        rollout_started = time.perf_counter()
        try:
            log_rows = _run_agent_turn_with_support(
                user_proxy,
                chatbot,
                question,
                str(composition.get("composed_context", "")),
            )
        except Exception as error:
            log_rows = [f"Prediction-side rollout error: {type(error).__name__}: {error}"]
        rollout_sec = float(time.perf_counter() - rollout_started)
        prediction_text = "\n----------------------------------------------------------\n".join(map(str, log_rows))
        assert_prediction_only(prediction_text)
        prediction = extract_prediction(prediction_text)
        prediction_path = predictions_dir / f"{turn:05d}.prediction.txt"
        prediction_path.write_text(prediction_text, encoding="utf-8")

        has_answer = "answer" in item
        answer = item.get("answer")
        if isinstance(answer, list):
            answer = ", ".join(str(value) for value in answer)
        success = bool(has_answer and judge(prediction, answer))
        successes += int(success)
        observed_reward = float(success)
        task_reward = observed_reward
        credited = policy.credit_selected_path(successor_credit, task_reward, turn)
        pending_utility_updates = dict(policy.last_pending_utility_updates)
        finalized = policy.finalize_decisions(turn)

        candidate_token = f"ehr:{safe_id}:{turn}"
        candidate = build_ttk_candidate(
            prompt_client,
            task_family="EHRSQL executable clinical database question answering",
            query=question,
            trajectory=prediction_text,
            selected_traces=selected_traces,
            task_reward=task_reward,
            outcome_summary=(
                "The benchmark execution evaluator returned only scalar success feedback; "
                "no target-side text was supplied to the memory constructor."
            ),
            provenance={"benchmark": "EHRSQL", "task_id": task_id, "turn": turn},
        )
        policy.add_candidate(
            candidate_token,
            candidate,
            candidate_utility=task_reward,
            promotion_score=confidence_value(candidate["confidence"]),
            time_step=turn,
        )
        lifecycle_events = policy.review_and_apply(persistent_memory, turn)
        # A max-age candidate receives its scheduled review first.  Only a
        # still-retained candidate is then removed by the declared hard cap.
        expired = policy.expire_pending(turn)
        turn_sec = float(time.perf_counter() - turn_started)

        record = {
            "experiment_mode": manifest["mode"],
            "turn": turn,
            "task_id": task_id,
            "success": int(success),
            "observed_task_reward": observed_reward,
            "applied_task_reward": task_reward,
            "feedback_source_task_id": task_id,
            "selected_indexes": selected_indexes,
            "selected_memory_ids": [str(item["mem_id"]) for item in selected_traces],
            "path_search": selector.last_search_summary,
            "path_prefix": selector.last_path_details,
            "credit_mode": "prefix",
            "path_credit": path_credit,
            "successor_credit": successor_credit,
            "credited_decisions": credited,
            "pending_utility_updates": pending_utility_updates,
            "p_path": composition,
            "p_ttk_candidate": candidate,
            "candidate_token": candidate_token,
            "candidate_promotion_score": confidence_value(candidate["confidence"]),
            "candidate_utility": task_reward,
            "expired_pending": expired,
            "lifecycle_events": lifecycle_events,
            "finalized_decisions": finalized,
            "persistent_memory": memory_size_stats(persistent_memory),
            "active_memory": memory_size_stats(retrieval_memory),
            "buffer_entries": len(policy.pending),
            "open_decisions": len(policy.open_decisions),
            "baseline_count": policy.baseline_count,
            "baseline": policy.baseline,
            "adam_step": policy.adam_step,
            "rollout_sec": rollout_sec,
            "turn_sec_retrieval_through_post_response_memory": turn_sec,
        }
        append_jsonl(records_path, record)
        append_jsonl(evaluator_path, {
            "task_id": task_id,
            "final_execution_source": user_proxy.last_execution_source,
            "prediction_path": str(prediction_path.relative_to(output_root)),
            "gold_answer": answer if has_answer else None,
            "success": int(success),
        })
        print(
            f"[neuromem-ehrsql] turn={turn} success={int(success)} reward={task_reward:.3f} "
            f"path={len(selected_traces)} actions={[event['action'] for event in lifecycle_events]} "
            f"M={len(persistent_memory)} B={len(policy.pending)}",
            flush=True,
        )

    final_decisions = policy.finalize_decisions(len(tasks) + policy.credit_horizon, force=True)
    write_json(output_root / "final_persistent_memory.json", persistent_memory)
    write_json(output_root / "final_buffer_memory.json", policy.buffer_records(retrievable_only=False))
    write_json(output_root / "final_decisions.json", final_decisions)
    with (output_root / "policy_state.pkl").open("wb") as handle:
        pickle.dump(policy.state_dict(), handle)
    manifest["completed_at_unix"] = time.time()
    manifest["elapsed_sec"] = manifest["completed_at_unix"] - started
    manifest["final"] = {
        "tasks": len(tasks),
        "successes": successes,
        "accuracy": successes / len(tasks),
        "persistent_entries": len(persistent_memory),
        "buffer_entries": len(policy.pending),
        "open_decisions": len(policy.open_decisions),
        "policy_updates": policy.policy_updates,
        "adam_step": policy.adam_step,
        "baseline_count": policy.baseline_count,
        "baseline": policy.baseline,
    }
    manifest["manifest_content_sha256"] = sha256_json({key: value for key, value in manifest.items() if key != "manifest_content_sha256"})
    write_json(output_root / "run_manifest.json", manifest)
    print("[neuromem-ehrsql] completed " + json.dumps(manifest["final"], sort_keys=True), flush=True)




if __name__ == "__main__":
    main()
