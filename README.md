# Learning LLM Inference

A weekly, hands-on learning journal for understanding how large language model
inference works, from a first generation loop to serving and optimization.
The exercises and notes will also serve as reusable material for team learning
offsites.

## Goals

- Build intuition by writing and running code, not just reading.
- Understand the tradeoffs between latency, throughput, memory, and quality.
- Record reproducible experiments, results, and open questions each week.
- Turn completed exercises into workshops teammates can follow independently.

## Learning roadmap

The [LLM Inference Learning Path](LEARNING_PATH.md) is the outline for this repo:
10 stages over roughly six months at five hours per week, from a first sampling
loop to tracing requests through DeepInfer. It includes readings, hands-on builds,
completion checklists, and a glossary.

## Repository layout

- `weeks/`: weekly notes, exercises, and experiment code.
- `offsites/`: facilitated sessions adapted from completed weekly exercises.
- `LEARNING_PATH.md`: the stage-by-stage learning outline.

## Weekly workflow

1. Copy `weeks/TEMPLATE.md` into a folder such as `weeks/01-inference-basics/README.md`.
2. Set a concrete learning goal and add the exercise code alongside the notes.
3. Document setup and exact commands, including model, package versions, hardware,
   and expected runtime or cost where relevant.
4. Run the experiment, record results and limitations, and note what to try next.
5. Adapt useful exercises using `offsites/TEMPLATE.md`.

There is no shared runtime or dependency setup yet. Each exercise should document
its requirements; add shared tooling when there is a working exercise to support.

## Team learning offsites

Sessions should combine a short conceptual introduction, a runnable hands-on
exercise, and a discussion of observed tradeoffs. Include prerequisites,
checkpoints, expected outputs, and a fallback for participants without a GPU.
Keep facilitator notes separate from participant instructions.

## Reproducibility and public sharing

- Use public or appropriately licensed models and datasets; link their terms.
- Never commit credentials, private workplace material, model weights, or datasets.
- Record hardware, software versions, warmup, input/output lengths, and measurement
  method when reporting performance.
- Distinguish observations from hypotheses and avoid generalizing from one run.
