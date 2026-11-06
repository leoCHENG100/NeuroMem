import time
import os
from typing import Dict, List, Tuple, Callable, Literal, Optional, Union
import openai
import json
from openai import OpenAI, AzureOpenAI
from autogen.agentchat import Agent, UserProxyAgent, ConversableAgent
from termcolor import colored
import Levenshtein


class MedAgent(UserProxyAgent):
    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        llm_config: Optional[Union[Dict, Literal[False]]] = False,
        system_message: Optional[Union[str, List]] = "",
        config_list: Optional[List[Dict]] = None,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
        )
        self.config_list = config_list
        self.question = ''
        self.code = ''
        self.knowledge = ''
        self.memory_strategy = "levenshtein"
        self.memory_selector = None
        self.last_selected_indexes: List[int] = []
        self.last_memory_selection_info: Dict = {}
        self.last_knowledge_retrieval_info: Dict = {}
        self._ehr_code_executor: Optional[Callable[[str], Tuple[int, str]]] = None
        self.last_executed_answer: Optional[str] = None
        self.last_execution_source: Optional[str] = None

        # Retry, timeout, and token limits for knowledge retrieval and diagnosis.
        self.chat_retry_patience = 3
        self.chat_retry_sleep = 20
        self.chat_timeout = 120
        self.chat_token_budgets = [256, 800, 1600]
        self.chat_debug_api = str(os.getenv("EHR_DEBUG_API", "0")).lower() in ("1", "true", "yes", "on")

    def configure_ehr_code_executor(self, executor: Callable[[str], Tuple[int, str]]):
        """Route Python code blocks through the dataset-aware EHR executor."""
        self._ehr_code_executor = executor
        self.reset_execution_answer()

    def reset_execution_answer(self):
        self.last_executed_answer = None
        self.last_execution_source = None

    def _record_ehr_result(self, exitcode, output, source):
        if exitcode == 0:
            self.last_executed_answer = str(output).strip()
            self.last_execution_source = source
        else:
            self.reset_execution_answer()

    def execute_code_blocks(self, code_blocks):
        """Execute Python blocks with EHR helpers when an executor is configured."""
        # Only the latest execution attempt may supply the evaluated answer.
        self.reset_execution_answer()
        if self._ehr_code_executor is None:
            return super().execute_code_blocks(code_blocks)

        if any(str(lang or "python").lower() not in ("python",) for lang, _ in code_blocks):
            return super().execute_code_blocks(code_blocks)

        logs_all = ""
        for index, (_, code) in enumerate(code_blocks):
            print(colored(f"\n>>>>>>>> EXECUTING EHR CODE BLOCK {index}...", "red"), flush=True)
            self.code = str(code)
            self.reset_execution_answer()
            exitcode, logs = self._ehr_code_executor(self.code)
            logs = str(logs)
            self._record_ehr_result(exitcode, logs, "python_code_block")
            logs_all += "\n" + logs
            if exitcode != 0:
                return int(exitcode), logs_all
        return 0, logs_all

    def _build_client(self, config):
        api_type = str(config.get("api_type", "openai")).lower()
        if api_type.startswith("azure"):
            return AzureOpenAI(
                api_key=config["api_key"],
                azure_endpoint=config["base_url"],
                api_version=config.get("api_version", "2024-02-01"),
            )
        return OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )

    def _debug_api_log(self, tag: str, payload):
        if not self.chat_debug_api:
            return
        try:
            print(f"[api-debug] {tag}: " + json.dumps(payload, ensure_ascii=False, default=str))
        except Exception:
            print(f"[api-debug] {tag}: {payload}")

    def _format_api_exception(self, err: Exception):
        out = {
            "type": type(err).__name__,
            "message": str(err),
        }
        for attr in ("status_code", "request_id", "code", "param", "body"):
            val = getattr(err, attr, None)
            if val is not None:
                out[attr] = val

        resp = getattr(err, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
            if status is not None:
                out["response_status_code"] = status

            headers = getattr(resp, "headers", None)
            if headers is not None and hasattr(headers, "get"):
                reqid = headers.get("x-request-id") or headers.get("request-id")
                if reqid:
                    out["response_request_id"] = reqid

            try:
                text = getattr(resp, "text", None)
                if text:
                    out["response_text"] = text
            except Exception:
                pass

            try:
                out["response_json"] = resp.json()
            except Exception:
                pass

        return out

    def _chat_with_retry(self, client, engine, messages, sleep_time=None):
        token_budgets = list(self.chat_token_budgets)
        total_patience = int(self.chat_retry_patience)
        patience = total_patience
        if sleep_time is None:
            sleep_time = float(self.chat_retry_sleep)
        last_error = None
        while patience > 0:
            patience -= 1
            attempt_no = total_patience - patience
            for max_tokens in token_budgets:
                try:
                    response = client.chat.completions.create(
                        model=engine,
                        messages=messages,
                        temperature=0,
                        max_tokens=max_tokens,
                        top_p=0.95,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                        timeout=self.chat_timeout,
                    )
                    choice = response.choices[0]
                    content = (choice.message.content or "").strip()

                    self._debug_api_log(
                        "chat_response",
                        {
                            "attempt": attempt_no,
                            "max_tokens": max_tokens,
                            "id": getattr(response, "id", None),
                            "model": getattr(response, "model", None),
                            "finish_reason": getattr(choice, "finish_reason", None),
                            "message_role": getattr(choice.message, "role", None),
                            "message_content": choice.message.content,
                            "tool_calls": getattr(choice.message, "tool_calls", None),
                            "usage": getattr(response, "usage", None),
                        },
                    )

                    if content and choice.finish_reason != "length":
                        return content
                except Exception as e:
                    last_error = e
                    self._debug_api_log(
                        "chat_exception",
                        {
                            "attempt": attempt_no,
                            "max_tokens": max_tokens,
                            "error": self._format_api_exception(e),
                        },
                    )
            if sleep_time > 0:
                time.sleep(sleep_time)
        if last_error is not None:
            print(last_error)
            self._debug_api_log("chat_last_error", self._format_api_exception(last_error))
        return ""

    def retrieve_knowledge(self, config, query):
        start_ts = time.perf_counter()
        RetrKnowledge = self.benchmark_prompts['RetrKnowledge']
        # Returns the related information to the given query.
        sleep_time = self.chat_retry_sleep
        openai.api_type = config.get("api_type", "openai")
        openai.api_base = config["base_url"]
        openai.api_version = config.get("api_version")
        openai.api_key = config["api_key"]
        engine = config["model"]
        query_message = RetrKnowledge.format(question=query)
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},
                    {"role":"user","content": query_message}]
        client = self._build_client(config)
        prediction = self._chat_with_retry(client, engine, messages, sleep_time=sleep_time)
        elapsed = float(time.perf_counter() - start_ts)
        self.last_knowledge_retrieval_info = {
            "mode": "retrieved",
            "model": engine,
            "elapsed_sec": elapsed,
            "query_chars": len(str(query or "")),
            "success": bool(prediction),
        }
        if prediction:
            return prediction
        self.last_knowledge_retrieval_info["success"] = False
        return "Fail to retrieve related knowledge, please try again later."

    def retrieve_examples(self, query):
        start_ts = time.perf_counter()
        selected_indexes = []
        used_strategy = self.memory_strategy
        fallback_used = False
        if self.memory_selector is not None:
            used_strategy = getattr(self.memory_selector, "strategy", self.memory_strategy)
            selected_indexes = self.memory_selector.select(query, self.memory, self.num_shots)

        if not selected_indexes:
            fallback_used = bool(self.memory_selector is not None)
            used_strategy = "levenshtein" if fallback_used else self.memory_strategy
            levenshtein_dist = {}
            for i in range(len(self.memory)):
                question = self.memory[i]["question"]
                levenshtein_dist[i] = Levenshtein.distance(query, question)
            levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
            selected_indexes = [levenshtein_dist[i][0] for i in range(min(self.num_shots, len(levenshtein_dist)))]

        self.last_selected_indexes = selected_indexes
        elapsed = float(time.perf_counter() - start_ts)
        self.last_memory_selection_info = {
            "strategy": used_strategy,
            "configured_strategy": self.memory_strategy,
            "fallback_used": bool(fallback_used),
            "elapsed_sec": elapsed,
            "memory_size": int(len(getattr(self, 'memory', []) or [])),
            "num_shots": int(getattr(self, 'num_shots', 0)),
            "selected_count": int(len(selected_indexes)),
        }
        examples = []
        for i in selected_indexes:
            template = "Question: {}\nKnowledge:\n{}\nSolution:\n{}\n".format(self.memory[i]["question"], self.memory[i]["knowledge"], self.memory[i]["code"])
            examples.append(template)
        examples = '\n'.join(examples)
        return examples

    def generate_init_message(self, **context):
        EHRAgent_Message_Prompt = self.benchmark_prompts['EHRAgent_Message_Prompt']
        self.question = context["message"]
        disable_knowledge = bool(context.get("disable_knowledge", False))
        fixed_knowledge = context.get("fixed_knowledge", None)
        if disable_knowledge:
            knowledge = ""
            self.last_knowledge_retrieval_info = {
                "mode": "disabled",
                "model": None,
                "elapsed_sec": 0.0,
                "query_chars": len(str(context["message"] or "")),
                "success": True,
            }
        elif fixed_knowledge is None:
            knowledge = self.retrieve_knowledge(self.config_list[0], context["message"])
        else:
            knowledge = str(fixed_knowledge)
            self.last_knowledge_retrieval_info = {
                "mode": "fixed",
                "model": None,
                "elapsed_sec": 0.0,
                "query_chars": len(str(context["message"] or "")),
                "success": True,
            }
        self.knowledge = knowledge

        examples = self.retrieve_examples(context["message"])

        if disable_knowledge:
            prompt_template = EHRAgent_Message_Prompt.replace("Knowledge:\n{knowledge}\n", "")
            init_message = prompt_template.format(examples=examples, question=context["message"])
        else:
            init_message = EHRAgent_Message_Prompt.format(examples=examples, knowledge=knowledge, question=context["message"])
        return init_message

    def send(self, message: Union[Dict, str], recipient: Agent, request_reply: Optional[bool]=None, silent: Optional[bool]=False):
        valid = self._append_oai_message(message, "assistant", recipient)
        if valid:
            recipient.receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    def initiate_chat(self, recipient: "ConversableAgent", clear_history: Optional[bool]=True, silent: Optional[bool]=False, **context,):
        self._prepare_chat(recipient, clear_history)
        self.send(self.generate_init_message(**context), recipient, silent=silent)

    def receive(
        self,
        message: Union[Dict, str],
        sender: Agent,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
    ):
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)

    def error_debugger(self, config, code, error_info):
        CodeDebugger = self.benchmark_prompts['CodeDebugger']
        # Diagnose the current code and execution feedback.
        sleep_time = self.chat_retry_sleep
        openai.api_type = config.get("api_type", "openai")
        openai.api_base = config["base_url"]
        openai.api_version = config.get("api_version")
        openai.api_key = config["api_key"]
        engine = config["model"]
        query_message = CodeDebugger.format(question=self.question, code=code, error_info=error_info)
        messages = [{"role":"system","content":"You are an AI assistant that helps people debug their code. Only list one most possible reason to the errors."},
                    {"role":"user","content": query_message}]
        client = self._build_client(config)
        prediction = self._chat_with_retry(client, engine, messages, sleep_time=sleep_time)
        if prediction:
            return prediction
        return "Fail to diagnose the reasons to the errors."

    def execute_function(self, func_call):
        """Execute a registered function and return status plus model feedback.

        EHR Python calls also record the latest successful answer. Execution
        failures clear that answer; completion guidance remains separate from
        the raw value used by the evaluator.
        """
        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)
        self.reset_execution_answer()

        is_exec_success = False
        if func is not None:
            # Extract arguments from a json-like string and put it into a dict.
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = None
                arguments_string = func_call["arguments"].split(': "')[-1]
                arguments_string = arguments_string.split('", ')[0]
                arguments = {"cell": arguments_string}
                content = f"Error: {e}\n There might be compilation errors in the code. Please check the code and try again."

            # Try to execute the function
            if isinstance(arguments, dict):
                print(
                    colored(f"\n>>>>>>>> EXECUTING FUNCTION {func_name}...", "magenta"),
                    flush=True,
                )
                try:
                    self.code = arguments["cell"]
                    result = func(**arguments)
                    if func_name == "python" and self._ehr_code_executor is not None:
                        exitcode, content = result
                        self._record_ehr_result(exitcode, content, "python_function")
                        is_exec_success = exitcode == 0
                    else:
                        content = result
                        is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
            else:
                content = "Error: Function arguments must be a JSON object."
        else:
            content = f"Error: Function {func_name} not found."
        if self.config_list and not is_exec_success:
            reasons = self.error_debugger(self.config_list[0], self.code, content)
            content = str(content) + '\nPotential Reasons: ' + str(reasons)

        if func_name == "python" and self._ehr_code_executor is not None:
            # Conversational feedback is separate from the raw saved answer.
            # Never treat execution success as benchmark correctness or force
            # termination: the model may still submit a genuine correction.
            if is_exec_success:
                content = (
                    "exitcode: 0 (execution succeeded)\nSaved answer: " + str(content)
                    + "\nThe executor has saved this answer. If it completes the task, reply with exactly "
                    "TERMINATE as plain assistant text, without a function/tool call or code block. "
                    "Do not call python just to print or repeat it. Only submit another complete solution "
                    "if correction is needed; each execution starts in a fresh scope."
                )
            else:
                content = (
                    "exitcode: 1 (execution failed)\n" + str(content)
                    + "\nThe saved answer has been cleared. To recover, execute a complete corrected solution "
                    "that assigns 'answer'. Variables from earlier calls do not persist."
                )

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }

    def update_memory(self, num_shots, memory):
        self.num_shots = num_shots
        self.memory = memory
        if self.memory_selector is not None:
            self.memory_selector.sync_memory_size(len(self.memory))
