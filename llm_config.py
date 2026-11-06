"""Credentials are read on demand. Never serialize the client configuration."""
import os


EHR_SYSTEM_MESSAGE = (
    "For coding tasks, only use the functions you have been provided with. "
    "Execute a complete, self-contained Python solution and assign the final result to 'answer'. "
    "Every execution starts in a fresh scope: variables from previous calls do not persist. "
    "The executor returns and saves the value of 'answer'; printing it is unnecessary. "
    "After a successful execution, inspect the returned answer. If it completes the task, "
    "reply with exactly TERMINATE as plain assistant text, with no function/tool call and no code block. "
    "TERMINATE is a conversation control message, never Python code. "
    "Do not call python again merely to print, repeat, or paraphrase an already obtained final answer. "
    "If the solution needs correction or execution failed, submit a new complete solution assigning 'answer'. "
    "A failed execution clears the saved answer. Execution success alone does not establish correctness."
)


def task_config(model):
    key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not key or not base_url:
        raise ValueError("Set OPENAI_API_KEY and OPENAI_BASE_URL for your authorized provider.")
    return {"model": model, "api_key": key, "base_url": base_url, "api_type": "openai",
            "max_retries": 5, "timeout": 180}


def agent_config(seed, config):
    return {
        "functions": [{"name": "python",
                       "description": (
                           "Run a complete Python solution in a fresh scope. Assign the result to 'answer'; "
                           "the executor returns and saves it automatically. Variables do not persist between calls. "
                           "After obtaining the final answer, send plain assistant text TERMINATE, not another call."
                       ),
                       "parameters": {"type": "object", "properties": {
                           "cell": {"type": "string", "description": "Valid Python code to execute."}},
                           "required": ["cell"]}}],
        "config_list": [config], "timeout": 120, "max_tokens": 1600,
        "cache_seed": seed, "temperature": 0,
    }
