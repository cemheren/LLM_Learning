# Local setup and learning exercises

This repository contains hands-on LLM inference exercises and team workshop
material. The curriculum is in `LEARNING_PATH.md`. The first runnable exercise
is `weeks/00-orientation/README.md`; read it before changing or running its code.

## Authoring future weeks: replication first

Optimize for a learner reproducing the exercise on their own computer, not for
the author's existing environment. Each week must be understandable on its own,
with explicit links to any earlier exercise it actually requires.

- Provide a shortest working path from a fresh clone to a meaningful result.
  Include copy-paste commands from the repo root for Windows PowerShell and
  macOS/Linux, with explicit virtual-environment interpreter paths.
- State supported Python versions, OS limitations, RAM, disk/download sizes,
  expected first-run time, and required network access before installation.
  Explain which steps need internet and which work after downloads are cached.
- Prefer a small public model and CPU baseline when they teach the objective.
  When GPU behavior is the objective, state the minimum GPU/VRAM and CUDA
  requirements and provide a specific hosted option and cost estimate.
  Label CPU simulations and precomputed results as fallbacks, not equivalent GPU
  validation.
- Keep dependencies minimal and declare them in the exercise folder. Document
  a tested dependency set and immutable model revision for reproduction; clearly
  separate tested configurations from merely expected compatibility.
  Do not silently upgrade an earlier week's environment if requirements conflict:
  document a separate per-exercise virtual environment instead.
- Use public sample inputs and sensible defaults. Avoid required credentials,
  private datasets, absolute author-machine paths, global installs, and system
  configuration changes. Explain caching and how to rerun without redownloading.
- Include an unattended smoke-test command, expected output or invariants, and
  common setup failures with fixes. Where practical, add small offline tests;
  distinguish these from a successful real-model or GPU run.
- Make code easy to inspect: explicit tensor operations, shape explanations,
  step-over/debugger instructions, and a few experiments changing one variable
  at a time. Define the concepts locally rather than requiring external videos.
- Include an assistant-ready setup prompt naming the files to read, commands to
  run, and completion criteria. Keep interactive debugging separate from setup.
- Before calling an exercise ready, follow its documented commands in a clean
  compatible environment without relying on undeclared global packages. Record
  the actual environment and results. If blocked, state what remains unverified;
  do not claim cross-platform support was tested when it was not.

## Set up Week 0

Run from the repository root. Use a local `.venv`, not globally installed packages.
Python must be 64-bit, version 3.11-3.13. CPU is supported and is the default.
Downloading public model weights is expected; no Hugging Face login is needed.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r weeks\00-orientation\requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s weeks\00-orientation -p "test_*.py" -v
.\.venv\Scripts\python.exe weeks\00-orientation\week0.py
```

macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r weeks/00-orientation/requirements.txt
.venv/bin/python -m unittest discover -s weeks/00-orientation -p "test_*.py" -v
.venv/bin/python weeks/00-orientation/week0.py
```

Reuse an existing compatible `.venv` if present. No API key, GPU, training,
external service, or editable package installation is required.
The first real run downloads about 550 MB of GPT-2 weights into the user's
Hugging Face cache. Read the exercise's resource estimates before installation.

## Verify readiness

- All offline unit tests pass.
- A real-model run prints token IDs, logits shaped `(1, T, 50257)`, five next-token
  probabilities, and a continuation of up to 50 tokens.
- The default run must work on CPU; do not replace it with a mock or a tiny random
  model to claim the real-model checkpoint is complete.
- Record the resolved model revision, installed dependency versions, and any
  setup failure when reporting what was actually verified.
- Do not launch `--debug` in an unattended setup run: it waits for interactive input.

If downloads fail, report the error and required network access; do not disable
TLS verification. Do not commit `.venv`, caches, downloaded weights, or credentials.
Do not commit or push changes unless requested.

## Teaching constraints

Keep the generation loop readable and explicit. Do not replace it with
`model.generate()`, add KV caching, or hide the tensor operations behind a serving
framework: those would skip this exercise's learning objective.
When a learner is ready, point them to the README's debugger walkthrough.
