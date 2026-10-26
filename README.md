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

This is a starting plan, not a fixed schedule. Each week should produce a small
working example and a short writeup.

| Week | Focus | Suggested deliverable |
|------|-------|-----------------------|
| 01 | Inference basics: tokenization, model loading, generation | Run a small model and inspect tokens and generation settings |
| 02 | Autoregressive decoding | Implement a generation loop; compare greedy decoding and sampling |
| 03 | Attention and KV caching | Compare cached and uncached decoding for correctness, latency, and memory |
| 04 | Measuring inference | Measure prefill, time to first token, decode latency, and throughput |
| 05 | Batching and serving | Explore batching tradeoffs and build a minimal serving example |
| 06 | Precision and quantization | Compare memory use, speed, and output quality across supported precisions |
| 07 | Inference optimization | Explore an optimization such as optimized attention or speculative decoding |
| 08 | End-to-end capstone | Build a reproducible demo and package it as a team learning offsite |

## Repository layout

- `weeks/`: weekly notes, exercises, and experiment code.
- `offsites/`: facilitated sessions adapted from completed weekly exercises.

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
