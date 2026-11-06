# NeuroMem

Official EHRSQL implementation of **NeuroMem: A Neuroplastic Memory Framework for Lifelong Agents through Delayed Consolidation**.

Yingyi Cheng, Xueqiang Han, and Chen Yu

Huazhong University of Science and Technology · Hubei JJJC BigData Centre

NeurIPS 2026

Yingyi Cheng and Xueqiang Han contributed equally. Chen Yu is the corresponding author.

We introduce NeuroMem, a memory framework that learns to transform transient interaction experiences into compact, task-transferable knowledge through delayed consolidation. A trace remains provisional until later retrieval and task outcomes provide evidence of its utility.

This repository covers the EHRSQL/MIMIC-III setting using the EHRAgent task environment. It includes our memory framework, task adapter, evaluator, and tests; LoCoMo and comparison-method runners are outside its scope. Benchmark data, model weights, credentials, and generated experiment outputs are not included.

## Method overview

Our method separates memory construction, support-path retrieval, and learned consolidation (paper Sections 3.1–3.3).

1. **Construct provisional traces.** After a task, `P_ttk` distills the prediction-side trajectory and scalar evaluator feedback into a structured candidate in the transient buffer `B`. Its seven fields are `knowledge`, `applicability`, `reasoning_pattern`, `composition_keys`, `evidence`, `cautions`, and `confidence` (Appendix A.5). Reference answers are not supplied to the memory constructor.
2. **Retrieve compositional support.** Frozen BGE-M3 embeddings rank eligible traces from the buffer and persistent memory `M`. Budgeted search extends ordered, non-repeating paths. The path score averages query-to-first-trace similarity and successive trace-to-trace similarities (Eq. 9); each generated and scored extension consumes one budget unit. `P_path` formats the selected path as task context without changing its membership, order, or score.
3. **Assign delayed credit.** Nonnegative prefix-value gains determine each selected trace's attribution (Eqs. 10–12). A consolidation decision receives discounted future task rewards only through its exact successor trace (Eqs. 13–15); credit does not propagate to predecessor records.
4. **Learn consolidation.** A four-action softmax policy selects `PROMOTE`, `MERGE`, `RETAIN`, or `DISCARD` from admissible actions. Finalized delayed returns include action-transition penalties for persistent-memory growth and redundancy (Eq. 6). Online Adam applies a REINFORCE-style update using the mean of previously finalized returns as the baseline (Eq. 7).

Every run starts with an empty buffer, empty persistent memory, and a fresh policy. A full evaluation processes all 581 tasks in their supplied order; `--smoke N` processes only the first N tasks. The policy learns online; no pretrained consolidation checkpoint or separate offline policy-training stage is required. The four fixed EHRAgent demonstrations remain outside `B` and `M`.

## Main configuration

The following defaults match Table 5 of the paper. The retrieval budget counts scored path extensions, not model calls or tokens.

| Hyperparameter | Value | Code setting |
|---|---:|---|
| Retrieval budget | 3072 | `expansion_budget` |
| Candidate pool size | 32 | `candidate_pool_size` |
| Delayed-utility horizon | 32 | `credit_horizon` |
| Pending retrieval minimum utility | 0.60 | `pending_retrieval_threshold` |
| Promotion score threshold | 0.50 | `promotion_score_threshold` |
| Promotion minimum candidate utility | 0.55 | `promotion_min_utility` |
| Review delay | 12 | `review_delay` |
| Maximum transient-buffer age | 96 | `max_pending_age` |
| Memory-growth coefficient | 0.05 | `lambda_memory` |
| Redundancy coefficient | 0.10 | `beta_redundancy` |

The policy learning rate is 0.03 (Section 3.2). Additional implementation settings are a discount factor of 0.99, attribution epsilon of `1e-8`, and a maximum merged-trace length of 8192 characters. The entry point uses policy seed 0, model-response cache seed 0, and temperature 0. The run manifest records policy settings, seeds, the model identifier, and source and input fingerprints. Hosted model output is not guaranteed to be bitwise deterministic.

The paper evaluates DeepSeek-v3.2 and Gemini-3-Flash. Set `--llm` to the corresponding deployment identifier exposed by your authorized provider; the same identifier is used for all model stages. The CLI defaults to `gemini-3-flash-preview`.

## Setup and execution

Clone the repository, then follow the numbered steps below:

```bash
git clone https://github.com/leoCHENG100/NeuroMem.git
cd NeuroMem
```

If you already have the source, start from that directory.

### 1. Install

Use Linux x86-64, Python 3.12 and a fresh virtual environment. Run the following blocks in the same Bash session so the activated environment, paths and `INPUTS` array remain available. From the source directory, set up the CPU environment. Set `WORK` to a private working directory outside the source tree. The model and environment require several GB of disk.

**Memory and device planning.** The local FP32 BGE-M3 checkpoint is about 2.3 GB. Budget 16 GB of host RAM and, for GPU execution, 8 GB of free VRAM as a starting allocation, not measured minimum requirements. Long inputs (up to 8192 encoder tokens), CSV tables, and other processes can require more. Check both host resources and container/job memory limits; a 2 GB RAM limit is insufficient, and a GPU does not replace host RAM. If model loading is killed or runs out of memory, increase resources and repeat preflight. Keep encoder weights, precision, and sequence limits fixed when comparing runs.

Choose the PyTorch build before running the install block. It shows CPU installation. For CUDA 12.8 with a compatible NVIDIA driver, replace that CPU wheel install line with `python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128` in the fresh environment, then install the same requirements. Other supported GPU builds are listed in the [official PyTorch 2.9.0 instructions](https://pytorch.org/get-started/previous-versions/#v290).

```bash
export SRC="$(pwd -P)"
export WORK="$HOME/neuromem-work"
umask 077
mkdir -p "$WORK/resources"
python3.12 -m venv "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m pip install 'torch @ https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl'
python -m pip install -r "$SRC/requirements.txt"
python -m pip check
cd "$SRC"
python -B -m unittest discover -s tests -v
export DEVICE=cpu
```

Our tests cover exact successor credit, bounded MERGE eligibility, active-memory counting, non-overlapping candidate reviews, full-horizon settlement, numeric abstraction, structured-output recovery, and policy-state restoration. They also cover function/code-block answer tracking, complete execution transcripts, status-based error diagnosis, plain-message termination, continued correction, clearing after a later execution failure, and smoke acceptance for both execution paths. For CUDA, install a compatible PyTorch 2.9.0 CUDA build in place of the CPU wheel before installing the remaining requirements. The CLI also accepts `--device cpu` or `--device cuda`.

**Optional GPU readiness check.** Skip this block for CPU execution. After installing the CUDA build in the same environment, select CUDA and verify that both the driver and PyTorch can use it. A CPU-only PyTorch installation cannot use a GPU even if `nvidia-smi` lists one. This check needs no data or hosted model credentials; the encoder preflight below checks the actual BGE-M3 load and encoding.

```bash
export DEVICE=cuda
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable: check the PyTorch build, driver, and GPU allocation."
probe = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print("CUDA ready:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__)
PY
```

### 2. Get the external resources

Use an authorized copy of the [EHRAgent preprocessed EHRSQL data](https://github.com/wshi83/EhrAgent/tree/7dfcaf36c230d0b987a16aabe64b8d3024ccd8a6#data-preparation). Use the matching database, CSV tables and annotated split from the same distribution. Set `DATA_ROOT` to the **parent** of `mimic_iii`. The expected layout is:

```text
$DATA_ROOT/mimic_iii/
    valid_preprocessed.json    ordered JSON list; id, template or question, answer
    mimic_iii.db               SQLite benchmark database
    ADMISSIONS.csv, ...        all 17 benchmark CSV tables, uppercase filenames
```

Use the archive's existing `valid_preprocessed.json` in its original order. The raw `valid.json` contains a different task stream. Keep clinical data, model weights, generated predictions, and service credentials outside the source tree.

Fetch the **pinned** public EHRAgent prompt resource and a local snapshot of the frozen BGE-M3 encoder. The external prompt file supplies the four fixed demonstrations and task prompts. The loader verifies its SHA-256 and reads string literals without importing the upstream module.

```bash
cd "$WORK"
export DATA_ROOT="/absolute/path/to/ehrsql-ehragent"
export PROMPTS="$WORK/resources/prompts_mimic.py"
export BGE="$WORK/resources/bge-m3"
curl --fail --location --output "$PROMPTS" \
  https://raw.githubusercontent.com/wshi83/EhrAgent/7dfcaf36c230d0b987a16aabe64b8d3024ccd8a6/ehragent/prompts_mimic.py
hf download BAAI/bge-m3 --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --local-dir "$BGE" --include '*.json' 'pytorch_model.bin' 'sentencepiece.bpe.model'
```

**Verify the data before running.** From the external MIMIC-III directory, check the exact bytes of the 581-task split, SQLite database and all 17 CSVs against [the input checksums](checksums/mimic_iii.sha256). Expect 19 `OK` lines and exit status zero. On a mismatch, stop and re-extract the matching distribution before continuing.

```bash
(cd "$DATA_ROOT/mimic_iii" && sha256sum --strict --check "$SRC/checksums/mimic_iii.sha256")
```

The split SHA-256 is `c8d4ba1841346e339a5a3106ffc190bd6bb8df4c9bfb2d80ddfda8d112eb3a04`. It covers both contents and row order. In `run_manifest.json`, compare `tasks_sha256` to this value. The separate `task_ids_sha256` fingerprint is computed over the ordered JSON list of task IDs.

Then run the local preflight on the selected `DEVICE` (`cpu` by default, or `cuda` after the GPU readiness check). It loads the encoder, verifies the pinned prompt resource, checks task fields and unique IDs, verifies that the required CSV files exist, and opens the SQLite database read-only to check table names. It does not parse CSV contents or validate every column or row; the checksum verification above is a separate required step. This preflight needs no hosted model credentials. Expect `Inputs OK` and exit status zero before proceeding. Explicit `--device cuda` does not silently fall back to CPU if the GPU is unavailable.

```bash
INPUTS=(--data-root "$DATA_ROOT" --tasks "$DATA_ROOT/mimic_iii/valid_preprocessed.json" \
        --prompts "$PROMPTS" --model-dir "$BGE")
python "$SRC/run_ehrsql.py" "${INPUTS[@]}" --device "$DEVICE" --check-inputs
```

For an end-to-end wiring check **without clinical data or a paid service**, run the following command after obtaining the prompt resource and encoder. The test starts a loopback chat server, builds 48 synthetic tasks and a SQLite database in a temporary directory, and invokes the real CLI. It requires successful SQL execution and scoring, support-path composition, full-horizon policy updates during the task stream, and a separate final flush of the remaining open decisions. The default first review is at step 12 and its 32-step return window closes at step 44 (the 45th task). Expect `Offline CLI smoke passed` and exit status zero. The test removes its temporary outputs; its scripted success rate is not a model-performance measurement.

```bash
(cd "$SRC" && python -B -m tests.smoke_cli --prompts "$PROMPTS" --model-dir "$BGE" --device "$DEVICE")
```

### 3. Run with your model

Use a provider authorized for your data and an OpenAI-compatible Chat Completions endpoint supporting the `functions` argument used by EHRAgent. Pass your deployment identifier with `--llm`. Questions, task trajectories, tool feedback, fixed demonstrations, and retrieved memory may be sent to this provider. The agent executes model-generated Python on the local host; run it in an isolated account or container with read-only data, access only to the intended services, and a private writable working directory. The read-only SQL tools do not restrict arbitrary generated Python or shell execution.

By default, `Calculate` is disabled and returns a tool error without making a network request. Supplying a Wolfram App ID alone does not enable it. To study a configuration with live calculator access, explicitly set both `NEUROMEM_ENABLE_WOLFRAM=1` and `WOLFRAMALPHA_APP_ID`; this changes the agent's tool feedback. The run manifest records `calculate_mode` as `offline` or `live_wolfram`. The SQL and calendar tools open the external SQLite database read-only; persistent database writes through these tools fail.

```bash
cd "$WORK"
export OPENAI_BASE_URL="https://your-authorized-provider.example/v1"
read -rsp "Provider API key: " OPENAI_API_KEY; echo
export OPENAI_API_KEY
export MODEL_ID="gemini-3-flash-preview"
unset NEUROMEM_ENABLE_WOLFRAM

# Allocate a fresh working directory as well as a new output path.
SMOKE_DIR="$(mktemp -d "$WORK/smoke.XXXXXXXX")"
(cd "$SMOKE_DIR" && python "$SRC/run_ehrsql.py" "${INPUTS[@]}" --device "$DEVICE" --llm "$MODEL_ID" \
  --smoke 2 --output "$SMOKE_DIR/run")
python "$SRC/check_smoke.py" "$SMOKE_DIR/run"
```

**Smoke acceptance.** Continue when the checker exits zero. It requires a completed smoke manifest, one evaluation record per task, a successful execution through either a Python code block or a registered `python` function call, and a nonempty extracted final answer for every task, with no rollout error. It prints the benchmark successes separately; wrong answers can still demonstrate working execution. Also inspect the last assistant exchange for a plain `TERMINATE` to confirm the model completed normally rather than exhausting the reply limit. If the checker fails, inspect the named prediction transcript locally. A `Prediction-side rollout error` calls for checking the endpoint, model and function-calling support. A missing execution or final answer calls for inspecting the agent's code/tool exchange. Two tasks precede the first scheduled policy review; use the longer synthetic test above to check delayed updates.

Run the full stream only after smoke acceptance. AutoGen can reuse task-model responses from `.cache` under the process working directory. A fresh `--output` alone does not isolate that cache. Use a fresh working directory for the full run and every independent repetition:

```bash
FULL_DIR="$(mktemp -d "$WORK/full.XXXXXXXX")"
(cd "$FULL_DIR" && python "$SRC/run_ehrsql.py" "${INPUTS[@]}" --device "$DEVICE" --llm "$MODEL_ID" \
  --output "$FULL_DIR/run")
```

### 4. Read your results

The paper's EHRSQL metrics are task success rate and final persistent entry count (Appendix A.3, Eqs. 16–17). Read these from the **completed** `run_manifest.json`:

- `final.accuracy`: task success rate as a fraction, equal to `final.successes / final.tasks`.
- `final.persistent_entries`: the number of independently addressable active entries in persistent memory. It excludes buffered traces and inactive predecessors, and is not a token count or storage-size measure.

The manifest also records the model, seeds, settings, source and input fingerprints. Inspect `task_records.jsonl` for selected paths, per-task scores, lifecycle decisions and finalized returns. `evaluator_records.jsonl` contains local reference answers. The runner stores prediction-only transcripts separately, uses only the scalar score as feedback, and writes final memory, finalized decisions and policy state after the stream ends. No LLM judge or additional task-reward shaping is used.

All generated memories, transcripts, evaluator records, response caches, and optional API diagnostics may contain sensitive text. Numeric instance masking is not comprehensive de-identification. Keep these artifacts private and outside the source tree.

**Policy state.** `policy_state.pkl` stores policy configuration, parameters, optimizer and random-generator state, pending candidates, open decisions, finalized history, and successor identity metadata. `NeuroMemPolicy.load_state_dict` restores these values into an independent policy state; the caller must supply the same frozen relation scorer and the corresponding active persistent memory separately. This is a policy-state interface, not full experiment resumption: the CLI has no resume option, and the task position, task-agent conversation, and response cache are not restored by it. Load pickle files only from trusted sources.

**Final-answer extraction.** Both Python code blocks and registered `python` function calls must assign the result to `answer`. A successful execution saves that value; a later failed execution, including a missing or deleted `answer`, clears it. If a saved answer exists, the runner appends `Final Executed Answer ---> <value>` to the prediction transcript. The extractor takes the text after the last marker, up to the next transcript separator. A missing marker yields an empty prediction and zero reward. The evaluator record's `final_execution_source` is `python_function` or `python_code_block`, or null when no answer is saved.

**Completion protocol.** Each Python execution starts in a fresh scope and returns `answer` automatically. After inspecting a complete answer, the model should send exactly `TERMINATE` as a plain assistant message, without a tool call or code block. An extra `print(answer)` call is unnecessary and cannot reuse variables from an earlier call. Corrections must supply a complete solution; later failures still clear the saved answer. Execution success does not automatically terminate the conversation or imply a correct benchmark answer. Completion guidance is separate from the saved value and is not part of the prediction supplied to the evaluator.

**Answer comparison.** The evaluator parses literal values or comma-space-separated text, flattens nested answer lists, normalizes whitespace and case, maps `yes/true` to `1` and `no/false/none` to `0`, and normalizes integral strings such as `3.0` to `3`. It compares sorted answer units with duplicate counts preserved. For example, `b, a` matches `a, b`; `a` does not match `a, a`. Success requires a nonempty reference multiset and exact equality; that binary value is the task reward. See [the evaluator](evaluation.py).

## Algorithm and source map

| Stage | Implementation |
|---|---|
| Retrieve and assign path weights | `neuromem_core.py`, `runtime.py`. Frozen BGE-M3 raw cosine, top 32 candidates, up to 3072 scored extensions, normalized nonnegative prefix gains. |
| Construct candidate and support context | `neuromem_prompts.py`, `path_composition.py`. The candidate constructor uses scalar benchmark-correctness feedback, not a code-execution status. The support-context composer preserves the selected path and order. Invalid JSON and schema violations receive at most three generation attempts; P_path then uses the grounded immutable path. Provider failures use separate request retries and remain explicit errors. |
| Review the buffer | `neuromem_buffer.py`. First review after 12 interactions, maximum age 96, and at most one open nonterminal decision per candidate; terminal review remains available at the age cap. Pending retrieval requires utility 0.60. PROMOTE requires score 0.50 and utility 0.55. |
| Track action successors | `neuromem_policy.py`. Each MERGE creates an exact successor identity and replaces one active record in its logical persistent slot. Credit belongs to the retrieved successor only; inactive predecessors remain metadata. |
| Learn the four-action policy | `neuromem_core.py`. An affine softmax controller, delayed returns, transition penalties, cumulative finalized-return baseline, and Adam updates. Redundancy is mean positive nearest-neighbor cosine. MERGE requires immutable anchor-key overlap and non-duplicate content that survives the 8192-character output bound; truncation alone is not content gain. |
| Execute and score | `run_ehrsql.py`, `ehr_adapter.py`, `benchmark.py`, `tools/tabtools.py`, `evaluation.py`. SQL/Python execution supplies the predicted answer, which is compared as a normalized multiset to the external reference. The four benchmark demonstrations stay outside learned memory. |

## Citation

If you use NeuroMem in your research, please cite our paper:

```bibtex
@inproceedings{cheng2026neuromem,
  title     = {{NeuroMem}: A Neuroplastic Memory Framework for Lifelong Agents through Delayed Consolidation},
  author    = {Cheng, Yingyi and Han, Xueqiang and Yu, Chen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgments

Our EHRSQL task interface and tools build on [EHRAgent](https://github.com/wshi83/EhrAgent). We use [EHRSQL](https://github.com/glee4810/EHRSQL) for evaluation and frozen [BGE-M3](https://huggingface.co/BAAI/bge-m3) embeddings for memory retrieval. See [NOTICE.md](NOTICE.md) for component provenance and licensing information.
