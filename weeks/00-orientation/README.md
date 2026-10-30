# Week 0: From a prompt to generated text

**Outcome:** run GPT-2 locally, inspect its next-token predictions, and generate
50 tokens using a loop you can explain. No GPU, API key, or prior machine learning
experience is required. This implements Stage 0 of the [learning path](../../LEARNING_PATH.md).

Allow 2-3 hours for setup and the exercise, plus optional reading and experiments.
You need basic Python knowledge: functions, loops, lists, and command-line arguments.

## 1. The mental model

A language model repeatedly answers: **"Given these tokens, what token comes next?"**
It does not produce a whole reply in one forward pass.

| Term | Meaning in this exercise |
|------|--------------------------|
| Token | A piece of text, sometimes a whole word, sometimes punctuation or part of a word |
| Tokenizer | Converts text to integer token IDs and IDs back to text |
| Weights | Downloaded numerical parameters learned during training; we do not train or change them |
| Tensor | An array of numbers; its shape describes the size of each dimension |
| Forward pass | Run the token IDs through the model to obtain scores |
| Logits | Raw scores, one for each possible next token; they are not probabilities |
| Softmax | Converts logits into probabilities that sum to one |
| Sampling | Choose a token according to those probabilities |
| Autoregressive loop | Append the chosen token and run the model again |

For one prompt with `T` tokens and a vocabulary of `V` possible tokens:

```text
prompt text
  -> tokenizer
  -> input IDs [1, T]
  -> model forward pass
  -> logits [1, T, V]
  -> take the last position: logits[:, -1, :] [1, V]
  -> softmax and choose one token [1, 1]
  -> append it to the input [1, T+1]
  -> repeat
```

The first dimension is the batch size: we handle one prompt at a time. Each
position's scores predict the token *after* that position. Only the last position
predicts what comes after the entire prompt.

**Temperature** changes the sampling distribution: `softmax(logits / temperature)`.
Lower positive values concentrate probability on high-scoring tokens; higher
values flatten it. Our `temperature=0` setting uses the largest logit directly
(greedy decoding), rather than dividing by zero. Temperature does not change the
weights or make the model more knowledgeable.

## 2. Setup

Use a **64-bit Python 3.11-3.13** installation and a terminal. CPU is the default,
including on Apple Silicon; no CUDA setup is needed. Allow roughly 4 GB of free
RAM and 3 GB of disk space for a CPU environment, packages, and weights. A GPU-enabled
PyTorch installation can need substantially more disk space.

The first run downloads public GPT-2 weights (about 550 MB) from Hugging Face.
Internet access is needed for installation and that first run. Model downloads go
to the Hugging Face cache, not this repo. Later runs reuse them. There are no API
fees; your own machine or cloud compute may have costs.

GPT-2 is a 124-million-parameter **base completion model**, not an instruction-tuned
assistant. Incoherent, outdated, or biased continuations are possible; fluent text
is not evidence of correctness. Model details and the MIT license:
[GPT-2 model card](https://huggingface.co/openai-community/gpt2).
Use the built-in public prompt, not private workplace data.

Clone the repository if you do not already have it:

```powershell
git clone https://github.com/cemheren/LLM_Learning.git
cd LLM_Learning
```

### Windows PowerShell

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r weeks\00-orientation\requirements.txt
.\.venv\Scripts\python.exe weeks\00-orientation\week0.py
```

No environment activation or PowerShell execution-policy change is required.
For a smaller CPU-only installation, install PyTorch from its CPU wheel index
before installing the requirements. `--no-deps` applies only to this first wheel
download; the requirements command immediately afterward installs its dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r weeks\00-orientation\requirements.txt
```

### Having an LLM assistant set this up

Open this repository in your coding assistant and give it this instruction:

> Read AGENTS.md and weeks/00-orientation/README.md. Set up Week 0 in a local
> virtual environment, run the offline tests, then run the default GPT-2 example
> on CPU. Downloading public weights is OK. Report the environment and results.
> Do not change the exercise, commit, or push. Once ready, show me the debugger
> command so I can step through the loop myself.

`AGENTS.md` contains the exact commands and readiness criteria. An assistant
should report download failures honestly rather than substitute an untrained model.

### macOS or Linux

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r weeks/00-orientation/requirements.txt
.venv/bin/python weeks/00-orientation/week0.py
```

The commands below use Windows paths. On macOS/Linux use `.venv/bin/python`
instead of `.\.venv\Scripts\python.exe` and `/` for directory separators.

## 3. First run: inspect, don't just read the text

The script prints:

1. The model, resolved model revision, device, and PyTorch version.
2. Each prompt token's ID and decoded text. Quotes make leading spaces visible.
3. Input and logits shapes. GPT-2's vocabulary dimension should be **50,257**.
4. The five highest-probability next tokens for the original prompt at temperature 1.
5. The full prompt plus up to 50 new tokens at your chosen temperature.

For example, the shape labels look like this; `T` below stands for the actual
prompt length, not literal script output:

```text
Token IDs shape: (1, T) = (batch, prompt tokens)
Logits shape: (1, T, 50257) = (batch, prompt tokens, vocabulary)
Top 5 next tokens (unmodified distribution, temperature=1):
...
Generated 50 new tokens
--- Full text (prompt + continuation) ---
The best way to learn something new is...
```

**Checkpoints:** the decoded prompt tokens reconstruct your prompt; the logits'
middle dimension matches the input length; each printed probability is between
0% and 100%. The top five probabilities generally do **not** sum to 100%, because
the other vocabulary entries retain probability mass. Generation may stop before
50 tokens if the model selects its end-of-text token.

Exact continuations can differ across package versions and hardware, even with
the same seed. Judge success by these structural checks, not a matching sentence.

### Verified local run

Verified on Windows with Python **3.13.15**, CPU PyTorch **2.14.1+cpu**, and
Transformers **4.57.6**. PyYAML **6.0.3** was also verified with a prebuilt Windows
wheel; the requirements explicitly prohibit compiling that package from source.
The dependency file pins those releases; the
script defaults to GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
Other operating systems and Python versions listed above are expected to work,
but have not been tested here.

The default prompt produced input shape `(1, 8)`, logits shape `(1, 8, 50257)`,
and 50 new tokens. Its top next token was `' to'` at approximately 80.57%.
The continuation began:

```text
The best way to learn something new is to just ask questions, and you'll learn something about yourself.
```

All seven offline unit tests passed. This is an observed example, not an exact
text requirement for other machines.

## 4. Read the loop

Open `week0.py`. Start with `generate_tokens`, then `sample_next_token`.

```python
output = model(input_ids=token_ids, use_cache=False)
next_logits = output.logits[:, -1, :]
next_token = sample_next_token(next_logits, temperature, generator)
token_ids = torch.cat((token_ids, next_token), dim=1)
```

Every iteration reruns the **whole growing input**. That is intentionally simple
and inefficient. We explicitly turn KV caching off; implementing that optimization
belongs in Stage 2. `eval()` disables training-time behaviors such as dropout;
`torch.inference_mode()` avoids building a gradient graph. Neither changes the
model's learned weights.

The program never calls Hugging Face's `generate()`. Our loop implements its
essential idea, but not its batching, advanced sampling, or serving optimizations.
The complete prompt plus requested output must fit GPT-2's **1,024-token context
window**; the script rejects larger requests rather than silently truncating them.

### Step over it yourself

Once the normal run works, start the built-in Python debugger:

```powershell
.\.venv\Scripts\python.exe weeks\00-orientation\week0.py --debug --max-new-tokens 5 --temperature 0
```

The model loads and the token table prints first. Execution then pauses just
before the generation loop. At the `(Pdb)` prompt:

| Command | What it does |
|---------|--------------|
| `p token_ids.shape` | Inspect the starting input shape |
| `n` | Execute the highlighted line without stepping into library code |
| `p output.logits.shape` | After the forward-pass line runs, inspect all position scores |
| `p next_logits.shape` | After the slice runs, inspect the last-position scores |
| `p next_token` | After sampling runs, see the selected token ID |
| `p token_ids.shape` | After concatenation, see the sequence grow by one |
| `l` | Show nearby code |
| `c` | Continue until the program finishes |
| `q` | Quit the debugger |

Use `n` repeatedly and inspect variables only after their assignment has executed.
**Step over** the model call; stepping into PyTorch internals is not useful yet.
On the next iteration, notice that the input and logits sequence lengths both grow.
The arrow marks the line that will execute **next**, not the one just completed.
On the verified Python 3.13 run, three `n` commands from the initial pause reach
`next_logits = ...`; at that point `p output.logits.shape` prints
`torch.Size([1, 8, 50257])`. Asking for `output` before executing the model line
produces `NameError`; this is a debugger timing issue, not a failed forward pass.

### Visual Studio Code

Open the repository folder in VS Code and install its recommended **Python**
and **Python Debugger** extensions. On Windows, the workspace settings default
to the repository's `.venv`; if needed, use **Python: Select Interpreter** to
choose that environment. On macOS/Linux, select `.venv/bin/python` explicitly.

Open **Run and Debug (Ctrl+Shift+D)**, select
**Week 0: step through generation**, and press **F5**. The included Windows
launch configuration runs 15 tokens with greedy decoding and pauses
automatically inside `generate_tokens`; no manual breakpoint is required.
It uses the interpreter selected by the Python extension. The provided launch
and test paths use Windows separators; on macOS/Linux adjust those paths first.

Use **Step Over (F10)** to execute one line without entering PyTorch internals,
**Step Into (F11)** to enter your own `sample_next_token` function, and
**Continue (F5)** to finish. Inspect locals in the Variables panel, or evaluate
`token_ids.shape`, `output.logits.shape`, and `next_logits.shape` in the Debug
Console after their assignments execute. **Shift+F5** stops the session.

## 5. Experiments

Keep the prompt and seed fixed while changing just one variable.

```powershell
.\.venv\Scripts\python.exe weeks\00-orientation\week0.py --temperature 0
.\.venv\Scripts\python.exe weeks\00-orientation\week0.py --temperature 0.3
.\.venv\Scripts\python.exe weeks\00-orientation\week0.py --temperature 1.2
.\.venv\Scripts\python.exe weeks\00-orientation\week0.py --temperature 0.8 --seed 7
.\.venv\Scripts\python.exe weeks\00-orientation\week0.py --prompt "A computer program is"
```

Record your results in a `notes.md` alongside this exercise:

| Prompt | Temperature | Seed | What changed? |
|--------|-------------|------|---------------|
| Default | 0 | 42 | |
| Default | 0.3 | 42 | |
| Default | 1.2 | 42 | |
| Default | 0.8 | 7 | |

Explain why the printed top-five table stays the same for the same prompt even
when generation temperature changes: that table always uses temperature 1.
Changing a seed should not change greedy decoding. Positive-temperature runs can
sometimes choose the same tokens; a single sample is not a guarantee of diversity.

For repeatable experiments, pass the resolved commit printed by the script as
`--revision COMMIT_HASH` and record the package versions:

```powershell
.\.venv\Scripts\python.exe -m pip freeze
```

## 6. Check your understanding

- [ ] I can explain prompt -> token IDs -> logits -> sampling -> appended token.
- [ ] I can explain all three dimensions of the logits tensor.
- [ ] I can explain why taking `logits[:, -1, :]` is the right next-token prediction.
- [ ] My loop generates text without calling `generate()`.
- [ ] I compared temperatures and can explain greedy versus probabilistic selection.
- [ ] I can explain why this model is a completion engine, not a chat assistant.

Quick questions (try answering before opening):

<details>
<summary>Why not select 50 tokens from the first forward pass?</summary>

The first pass gives the distribution for the next token after the prompt.
Later predictions depend on the tokens actually selected so far. Sampling from
that same distribution 50 times ignores those dependencies.
</details>

<details>
<summary>Does a token ID of 500 mean it is more likely than token 100?</summary>

No. IDs are vocabulary lookup indices. Logits, not the numerical IDs, determine
the probabilities for this particular context.
</details>

<details>
<summary>Why can a high-temperature continuation be less coherent?</summary>

Higher temperature gives relatively more probability to lower-scoring tokens.
Those choices also change the context for every subsequent prediction.
</details>

## 7. Troubleshooting and offline checks

| Symptom | Action |
|---------|--------|
| `python` is missing or opens the Windows Store | Install 64-bit Python 3.11-3.13 and reopen the terminal |
| No compatible PyTorch wheel | Check your Python version and architecture; use a supported 64-bit Python |
| `building 'yaml._yaml' extension` / Microsoft Visual C++ required | Use the PyYAML wheel recovery commands below; a compiler is not required for the supported setup |
| `ModuleNotFoundError` | Run the requirements command with the same `.venv` Python used to run the script |
| Download or certificate error | Check access to Hugging Face and your network's proxy/certificate configuration; do not disable TLS verification |
| Slow first run | Wait for the weight download; later runs use the local cache |
| Low memory or slow generation | Close memory-heavy applications and try `--max-new-tokens 10`; no GPU is required |
| Context-window error | Shorten the prompt or reduce `--max-new-tokens` |
| Strange or repetitive text | Expected for a small base model; compare temperatures and prompts |
| Windows symlink-cache warning | Harmless for this exercise; caching still works but may use extra disk space. Administrator privileges are not required |
| Missing `hf_xet` warning | Ordinary HTTP downloads still work; no additional package is required for this exercise |

### If PyYAML asks for Microsoft C++ Build Tools

This means pip selected PyYAML source code rather than a prebuilt wheel. It does
not mean this exercise requires a compiler. An old pip, an unsupported Python
version/architecture, or a package mirror missing compatible wheels can cause
this; the compiler error alone does not identify which one applies.

Run these commands from the repository root using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: PyYAML==6.0.3
.\.venv\Scripts\python.exe -m pip install -r weeks\00-orientation\requirements.txt
```

The requirements now pin PyYAML and require its wheel, so unsupported setups fail
with a missing-distribution message instead of attempting a C++ build.
If the wheel command reports no matching distribution, check:

```powershell
.\.venv\Scripts\python.exe -c "import sys, platform, struct; print(sys.version); print(platform.machine()); print(struct.calcsize('P') * 8, 'bit')"
.\.venv\Scripts\python.exe -m pip --version
```

Use the supported 64-bit Python 3.11-3.13 environment. On Windows ARM devices,
this walkthrough's tested configuration is **x64 Python**, not native ARM Python.
If Python is supported but the configured index lacks the wheel, use your
network-approved package index or the documented fallback below. Do not install
C++ Build Tools as the first workaround.

### If PyPI's download CDN fails

On the verified machine, both Python and Windows curl failed TLS handshakes to
`files.pythonhosted.org`, while the official PyTorch CPU host and a public PyPI
mirror were reachable. The following fallback successfully installed the exercise.
Use it only if this mirror is permitted on your network; it is a third-party
package source operated by Tsinghua University, not the default PyPI service.

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r weeks\00-orientation\requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
.\.venv\Scripts\python.exe -m pip check
```

The separate commands keep dependency resolution on the chosen mirror. Adding
the PyTorch host as an extra index can select dependency links back to the
unreachable CDN. Do not disable TLS verification or change global pip settings.
If your network uses an approved internal mirror, use that instead.

After one successful model run, test that the cached exercise works without
Hugging Face network requests:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe weeks\00-orientation\week0.py
Remove-Item Env:HF_HUB_OFFLINE
```

On macOS/Linux, use `HF_HUB_OFFLINE=1 .venv/bin/python weeks/00-orientation/week0.py`.

Run the small tests after installing dependencies:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s weeks\00-orientation -p "test_*.py" -v
```

These instantiate a tiny random model locally: no weights or tokenizer downloads
are needed. They check the loop, not the quality of pretrained GPT-2.
If you cannot download weights during an offsite, run these checks, trace the
loop's shapes together, and pair with someone whose model is already cached.
That fallback does not replace the real-model completion checkpoint.

## 8. Team session (90 minutes, setup completed beforehand)

| Time | Activity |
|------|----------|
| 0-15 min | Walk through the mental model and predict tensor shapes |
| 15-35 min | Run the script and explain the token and probability output |
| 35-60 min | Work in pairs through the temperature experiments |
| 60-80 min | Trace the loop; explain why each next prediction needs the selected token |
| 80-90 min | Answer the questions and share one observation and one limitation |

**Facilitator preparation:** install dependencies and run the default command on
each participant machine before the session to populate caches. Verify the tests
run; have a screen-share partner ready for download failures. Ask participants
to predict changes before running experiments, rather than only reading results.

## Optional reading and next step

- [3Blue1Brown: Large Language Models explained briefly](https://www.3blue1brown.com/lessons/mini-llm)
- [3Blue1Brown: Transformers](https://www.3blue1brown.com/lessons/gpt)
- [3Blue1Brown: Attention](https://www.3blue1brown.com/lessons/attention)

These reinforce the explanation but are not prerequisites for running this exercise.
Next, Stage 1 opens the model's forward pass to explain embeddings, attention,
and the MLP. The later DeepInfer reading target for this stage is
`deepinfer/entrypoints/llm.py`: recognize the generation loop's production role,
not every implementation detail yet.
