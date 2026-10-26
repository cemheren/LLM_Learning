# LLM Inference Learning Path

Oct 20, 2026 · @Akif

About 6 months at ~5 hours a week takes you from zero to reading DeepInfer. There are 10 stages, and each ends with something you build plus a DeepInfer file you can then read. You already know the systems half (scheduling, caching, sharding, load balancing); the new half is the model and the GPU.

## How to use this path

Work the stages in order: each one assumes the ones before it. A stage is done when its build runs and its checklist is ticked, not when the reading is finished.

- **Pace.** The weeks assume about 5 hours a week, about 25 weeks in all. At 10 hours a week it takes about 3 months.
- **Hardware.** Stages 0–2 run on a laptop (CPU or Apple Silicon). From Stage 3 on you need an NVIDIA GPU. A free Colab T4 covers most builds, and one rented 16–24 GB cloud GPU covers all of them. DeepInfer itself needs SM100 (B200-class) GPUs, so you will read it rather than run it.
- **Keep a learning repo.** Use one folder per stage, holding the notebook, a short notes.md and any plots. That repo is the record of what you understand.
- **Ask for shape annotations.** When code is opaque, paste it to Claude and ask for each tensor's shape and meaning, line by line. Most inference code is confusing because the shapes are implicit.
- **Don't learn training.** Backpropagation, optimizers and fine-tuning are optional here. Inference is the forward pass plus systems engineering.

## Roadmap

Stage 4 is the longest and the most familiar: it is a scheduler and a memory allocator, which you already know how to reason about.

| # | Stage | Weeks | You'll be able to | Build | DeepInfer files it unlocks |
| --- | --- | --- | --- | --- | --- |
| 0 | Orientation | 1 | Explain prompt → tokens → next-token loop | Hand-written sampling loop around a small model | `entrypoints/llm.py` |
| 1 | Transformer internals | 3 | Read a model's forward pass, with shapes | GPT from scratch, then load GPT-2 weights | `models/deepseek.py`, `decoder_layer.py` |
| 2 | One request: KV cache | 2 | Estimate tokens/s from memory bandwidth | Add a KV cache to your GPT and measure it | `worker/cache_engine.py` |
| 3 | GPU fundamentals | 3 | Say whether work is compute-, memory- or overhead-bound | Roofline plot; CUDA graph speedup | `worker/cuda_graph_runner.py`, `tilelang_ops/` |
| 4 | Serving many requests | 4 | Design a batching scheduler with paged memory | Scheduler simulator; read and modify nano-vllm | `engine/llm_engine.py`, `csrc/engine/scheduler.hpp`, `block_manager.hpp` |
| 5 | Speculative decoding | 2 | Explain draft-and-verify and why outputs don't change | Toy speculative decoding with two model sizes | `layers/sampler.py`, `mtp_varlen.hpp`, `mtp/dflash.py` |
| 6 | Parallelism and MoE | 3 | Pick and justify a DP/TP/EP layout | Toy MoE layer with a load rebalancer | `v4_modules/mlp.py`, `eplb/` |
| 7 | DeepSeek's architecture | 3 | Explain each trick in V4 attention and what it saves | Annotated diagram of the V4 KV cache | `v4_modules/attn.py`, `yoco.py`, `compress.py` |
| 8 | The production system | 2 | Explain disaggregation, KV storage and async scheduling | Your own serving design doc | `worker/worker.py`, `doc/inference_deployment_qa.md` |
| 9 | Capstone | 2+ | Trace one request through Python and C++ | End-to-end trace doc | The whole repo |

## Stage 0 — Orientation (1 week)

**Goal:** describe everything that happens between sending a prompt and receiving the streamed reply.

**Concepts:** token, tokenizer, weights, forward pass, logits, sampling, temperature, the autoregressive loop (each new token is fed back in as input).

**Watch and read**

- [Large Language Models explained briefly](https://www.3blue1brown.com/lessons/mini-llm) (3Blue1Brown)
- [Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt), Deep Learning chapter 5
- [Attention in transformers, step-by-step](https://www.3blue1brown.com/lessons/attention), Deep Learning chapter 6
- [Inference Engineering, chapter 0](https://www.baseten.co/inference-engineering/book/00-inference.md) (Philip Kiely, Baseten): the three layers of an inference stack

**Build:** in a notebook on your laptop, load `Qwen/Qwen3-0.6B` (or GPT-2) with Hugging Face `transformers`. Tokenize a prompt, run one forward pass, and print the 5 most likely next tokens with their probabilities. Then write your own `while` loop that samples 50 tokens with a temperature setting, without calling `generate()`.

- [ ] I can explain why generation happens one token at a time
- [ ] My loop produces text, and changing the temperature visibly changes it

**Unlocks in DeepInfer:** `deepinfer/entrypoints/llm.py`. `LLM.generate` is your loop at production scale.

## Stage 1 — Transformer internals (3 weeks)

**Goal:** read a model's forward pass and state the shape and meaning of every tensor in it.

**Concepts:** embeddings; attention as queries, keys and values; multi-head attention; the MLP (feed-forward) block; residual connections; layer norm; positional encoding (RoPE).

**Watch and read**

- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) (Karpathy, about 2 hours). This is the core resource of the stage.
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (Karpathy)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar), for the pictures
- Optional: [building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0), if you want to see how neural nets learn. Inference doesn't need it.

**Build:** code along with the video until your character-level GPT trains on Shakespeare (CPU or Mac is fine). Then make your model's layer names match GPT-2, load the real GPT-2 weights, and check that your logits match Hugging Face's for the same prompt. [nanoGPT](https://github.com/karpathy/nanoGPT) is the reference if you get stuck.

- [ ] I can write self-attention from memory, with shapes
- [ ] My GPT-2 matches Hugging Face's logits to about 1e-4

**Unlocks in DeepInfer:** `models/deepseek.py` (`DeepseekModel.forward` is the loop over layers) and `models/decoder_layer.py` (each layer is attention, then MLP).

## Stage 2 — One request: KV cache, prefill vs decode (2 weeks)

**Goal:** predict, using arithmetic, roughly how many tokens per second a model will generate on given hardware.

**Concepts:** the KV cache (saved keys and values, so earlier tokens are never recomputed); **prefill**, which processes the whole prompt in one parallel pass; **decode**, which generates one token per step; why decode is limited by memory bandwidth rather than by arithmetic; TTFT and TBT; the rule of thumb that decode time per step is about the bytes of weights read divided by memory bandwidth.

**Read**

- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (kipply). Read it twice; it is the most important text in this path.
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (Google DeepMind): the "Transformer math" and "Inference" chapters. It uses TPUs as examples, but the reasoning transfers to GPUs.

**Build:** add a KV cache to your Stage 1 GPT-2. Measure tokens/s with and without it for prompts of 100, 1,000 and 4,000 tokens, and plot the results. Then compute the expected decode speed from your machine's memory bandwidth and compare it with what you measured.

- [ ] My estimate is within 2× of my measurement
- [ ] I can explain why serving a bigger batch makes each token cheaper in decode but not in prefill

**Unlocks in DeepInfer:** `worker/cache_engine.py` (the `KVCache` dataclass and the allocator), and the reason the README talks about prefill and decode as separate things.

## Stage 3 — GPU fundamentals (3 weeks)

**Goal:** look at a piece of GPU work and say whether arithmetic, memory traffic or launch overhead limits it, and which fix applies to each.

**Concepts:** streaming multiprocessors (SMs); HBM (main GPU memory) vs on-chip SRAM; kernels and kernel-launch overhead; matmul tiling; the roofline model; FlashAttention's main idea (keep data on-chip); low-precision numbers (FP8, FP4) with scale factors; CUDA graphs.

**Read and watch**

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (Horace He). It introduces the compute / memory / overhead framing.
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): the "Roofline" and "GPUs" chapters
- [GPU MODE lectures](https://github.com/gpu-mode/lectures): start with the profiling and Triton lectures and skip anything that feels like CUDA C++ for now
- [FlashAttention paper](https://arxiv.org/abs/2205.14135): read only the introduction and figure 1

**Build** (on a GPU from here on)

1. Benchmark `torch.matmul` across sizes and plot the TFLOPs you achieve against the GPU's peak.
2. Profile one decode step of your GPT-2 with `torch.profiler` and find the slowest kernels.
3. Capture that decode step in `torch.cuda.CUDAGraph` and measure the speedup at batch size 1.
4. Optional: write one [Triton](https://triton-lang.org/) kernel, starting with vector add and moving on to softmax.

- [ ] I have a roofline plot of my GPU with real measurements on it
- [ ] I measured the CUDA graph speedup and can explain where it comes from

**Unlocks in DeepInfer:** `worker/cuda_graph_runner.py`, the existence of `layers/tilelang_ops/` (custom kernels), and why the README leads with kernel libraries (FlashMLA, DeepGEMM, DeepEP). Don't read the kernels themselves yet.

## Stage 4 — Serving many requests: the engine (4 weeks)

**Goal:** design, implement and measure a batching scheduler that manages KV-cache memory in pages. This is the heart of every inference engine, DeepInfer included.

**Concepts:** static vs continuous batching (requests join and leave the batch at every step); paged KV cache (fixed-size blocks, a block table per request, reference counts); prefix caching (reusing KV for a shared prompt prefix); chunked prefill (splitting a long prompt so it doesn't stall everyone else's decode); preemption and recompute when memory runs out; the throughput-vs-latency trade-off.

**Read**

- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI '22): iteration-level (continuous) batching
- [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180): the KV cache as virtual memory
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310): chunked prefill
- [SGLang](https://arxiv.org/abs/2312.07104): RadixAttention prefix caching. Read section 3 only.

**Build**

1. **A simulator, no GPU needed.** Requests arrive with prompt and output lengths, and the KV budget is a fixed number of blocks. Each step costs a time based on your Stage 2 model. Implement FCFS continuous batching with preemption, then add chunked prefill. Report throughput, p50/p99 TTFT and p99 TBT.
2. **Read [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).** It is about 1,400 lines of Python and runs `Qwen3-0.6B` on an 8 GB GPU. Read `engine/` end to end, change one policy (for example, which request to preempt), and measure the effect with its `bench.py`.

- [ ] I can draw nano-vllm's step loop from memory
- [ ] My simulator shows chunked prefill lowering p99 TBT, at some cost to TTFT

**Unlocks in DeepInfer:** `engine/llm_engine.py` (`step()`), `csrc/engine/scheduler.hpp` and `block_manager.hpp`. nano-vllm uses the same file names (`llm_engine`, `scheduler`, `block_manager`, `model_runner`, `sequence`, `sampler`). DeepInfer is that skeleton plus everything in Stages 5–8.

## Stage 5 — Sampling and speculative decoding (2 weeks)

**Goal:** explain how guessing several tokens ahead speeds up decode without changing what the model outputs.

**Concepts:** logit processing (temperature, top-k, top-p); speculative decoding (a cheap drafter proposes k tokens and the big model checks them all in one pass); the accept/reject rule that keeps outputs statistically identical; acceptance rate; MTP (multi-token prediction) heads as built-in drafters; variable-length verification; structured output as a per-token mask over the vocabulary.

**Read**

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al.)
- [EAGLE](https://arxiv.org/abs/2401.15077): drafting from the big model's own hidden states
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437): the multi-token prediction section
- DeepInfer's `doc/inference_deployment_qa.md`: the "Speculative decoding" section, which you can now follow

**Build:** implement speculative decoding with two sizes from one model family (for example, `Qwen3-0.6B` drafting for `Qwen3-1.7B`). Sweep k from 1 to 8 and plot the acceptance rate and the end-to-end speedup.

- [ ] I have a plot of speedup against k, with the best k marked
- [ ] I can explain why checking 5 draft tokens costs about the same as generating 1 token in decode (hint: Stage 2)

**Unlocks in DeepInfer:** `layers/sampler.py` (`_verification`), `csrc/engine/mtp_varlen.hpp` (which drafts are worth verifying), `models/mtp/dflash.py` (the drafter) and `structure_output/`.

## Stage 6 — Many GPUs: parallelism and MoE (3 weeks)

**Goal:** given a model and a cluster, choose how to split the work across GPUs and justify the choice with numbers.

**Concepts:** data parallelism (DP), tensor parallelism (TP), pipeline parallelism, expert parallelism (EP), context parallelism (CP); collectives (all-reduce, all-gather, all-to-all); NVLink and scale-up domains vs the network; Mixture-of-Experts (a router sends each token to a few of many expert sub-networks); load imbalance and redundant experts (EPLB).

**Read**

- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/): the "Sharding" and "Applied inference" chapters
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066): why many small experts
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437): the MoE, load-balancing and deployment sections
- [DeepEP README](https://github.com/deepseek-ai/DeepEP): what dispatch and combine do

**Build:** write a toy MoE layer in PyTorch with a router and top-k selection. Run real text through it, log the load on each expert, and implement a greedy rebalancer that replicates hot experts across "GPUs" (simulated as lists). Measure the max/mean load ratio before and after rebalancing. Optional, with 2 GPUs: perform the dispatch with `torch.distributed.all_to_all`.

- [ ] My rebalancer measurably lowers the max/mean load ratio
- [ ] I can explain why DeepSeek runs decode at EP64–128 but prefill at EP2–8

**Unlocks in DeepInfer:** `v4_modules/mlp.py` (`DeepseekMoE.forward`: gate, dispatch, experts, combine), `model_executor/eplb/` and `parallel_utils/parallel_state.py`.

## Stage 7 — DeepSeek's architecture tricks (3 weeks)

**Goal:** read DeepInfer's V4 attention code and name what each trick saves: memory, compute or bandwidth.

**Concepts:** MLA (cache a small compressed vector instead of full keys and values); sparse attention (a lightweight indexer picks the top-k past tokens worth attending to); sliding-window attention (attend only to the last N tokens); KV compression ratios; YOCO (cache once in early layers and let later layers reuse it, so prefill can skip work); hyper-connections and Engram, but only enough to recognize them in the code.

**Read**

- [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434): the MLA section
- [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp): DeepSeek Sparse Attention and its "lightning indexer"
- [You Only Cache Once (YOCO)](https://arxiv.org/abs/2405.05254)
- DeepInfer's `doc/inference_deployment_qa.md`: the "KV cache storage" and "KV cache hits and recomputation" sections

**Build:** go through `V4Attention.forward` in `v4_modules/attn.py` and annotate every line with tensor shapes and where each piece lives (paged block cache, per-request state slot, or indexer cache). Turn the result into a one-page diagram of the V4 KV cache.

- [ ] My diagram matches `StateBlock` and `KVCache` in `worker/cache_engine.py`
- [ ] I can explain what "SWA bounded replay" recomputes and why that is cheap

**Unlocks in DeepInfer:** `v4_modules/attn.py`, `compress.py`, `index_attn.py`, `yoco.py`, `cross_layer.py` and `worker/cache_engine.py`.

## Stage 8 — The production system around the engine (2 weeks)

**Goal:** explain how a fleet of engines serves trillions of tokens a day, and which parts DeepInfer leaves out.

**Concepts:** prefill/decode (PD) disaggregation, meaning separate clusters for each phase with KV cache shipped between them; KV-cache storage tiers (GPU memory, host memory, a distributed file system); load balancing by compute quota rather than request count; asynchronous (overlapped) CPU/GPU scheduling; autoscaling and failure domains in large EP groups; batch-invariant, deterministic inference.

**Read**

- [DistServe](https://arxiv.org/abs/2401.09670): why prefill and decode are split
- [Mooncake](https://arxiv.org/abs/2407.00079): a KV-cache-centric serving architecture
- [3FS](https://github.com/deepseek-ai/3FS): the file system DeepSeek stores KV cache on
- DeepInfer's `doc/inference_deployment_qa.md`, end to end. By now every section should make sense.

**Build:** write a 2-page design doc titled "How I would serve model X on my infrastructure". Use the QA doc's numbers (GPU counts, storage per GPU, TTLs, load-balancer rules) and map each component to infrastructure you run today.

- [ ] My design doc names the bottleneck of each tier and its SLO
- [ ] I can explain zero-overhead scheduling: what overlaps with what, and why sequence lengths lag one step

**Unlocks in DeepInfer:** `worker/worker.py` (`execute_model_step`: forward N, then post-process N−1, then sample N) and the empty hook methods in `LLMEngine` (`init_pd_split`, `do_save_kvcache`) where the unreleased PD and KV-storage code plugs in.

## Stage 9 — Capstone: read DeepInfer (2+ weeks)

**Goal:** trace one request from `LLM.generate` to a returned token, across Python and C++, and name every place where DeepInfer goes beyond nano-vllm.

**Reading order** (each step builds on the one before)

1. `README.md`, then `doc/inference_deployment_qa.md`
2. `examples/offline_inference.py`: how an engine is configured and launched
3. `deepinfer/entrypoints/llm.py`, then `engine/llm_engine.py` (`add_request`, `step`, `post_process`)
4. `deepinfer/worker/worker.py` (`execute_model_step`), then `worker/model_runner.py`
5. `csrc/engine/scheduler.hpp` (`_schedule`, `_schedule_prefill`, `_schedule_decode`), `block_manager.hpp`, `mtp_varlen.hpp`
6. `model_executor/models/deepseek.py`, then `decoder_layer.py`
7. `v4_modules/attn.py`, then `v4_modules/mlp.py`
8. `layers/sampler.py`, then `worker/cache_engine.py`
9. `tests/`, to see what the authors consider important enough to check

**Build:** a trace doc, either a sequence diagram or numbered steps, following one request through prefill, a few decode steps with speculation, and completion. Include which queue the request sits in, when its KV blocks are allocated and freed, and which process does what. If you ever get B200-class GPUs, run `examples/offline_inference.py` and `examples/benchmark.py`.

- [ ] My trace doc covers prefill, decode and finish
- [ ] I can list the 3 biggest differences between DeepInfer and nano-vllm, and the reason for each

## Glossary

These are the terms from the DeepInfer summary, ordered by the stage where you learn them. Each analogy is an approximation to get you started; the stage gives you the real thing.

| Term | Plain meaning | Backend analogy | Stage |
| --- | --- | --- | --- |
| Token | A word piece; the unit models read, write and bill by | A protocol frame | 0 |
| Logits / sampling | A score for every vocabulary token, then a pick of one | Weighted random choice over a scored candidate list | 0 |
| Temperature | How random that pick is | A jitter knob | 0 |
| Forward pass | One run of input through all layers | One request through a fixed pipeline of stages | 1 |
| Attention | Each token looks up relevant earlier tokens | A soft key-value lookup over the conversation so far | 1 |
| Layer / MLP | Repeated block: attention, then a per-token transform | A middleware chain applied N times | 1 |
| Prefill | Processing the whole prompt in one parallel pass | Bulk load or index build: compute-heavy | 2 |
| Decode | Generating one token per step | A chatty request/response loop: latency-bound | 2 |
| KV cache | Saved per-token state so earlier tokens are never recomputed | A per-session memoization cache that grows with the conversation | 2 |
| TTFT / TBT | Time to first token / time between tokens | p50/p99 latency SLOs for a streaming API | 2 |
| HBM | The GPU's main memory | RAM; its bandwidth is the bottleneck in decode | 2 |
| Memory- vs compute-bound | Waiting on reading data vs on arithmetic | IO-bound vs CPU-bound | 3 |
| Kernel | A function that runs on the GPU | A stored procedure executed on the device | 3 |
| CUDA graph | Record a sequence of kernel launches once, replay it with one call | A prepared statement, or a batched RPC | 3 |
| FP8 / FP4 / MXFP8 | Numbers stored in fewer bits, with shared scale factors | Lossy compression trading precision for bandwidth | 3 |
| SM100 | NVIDIA Blackwell (B200-class) GPUs | A specific instance SKU | 3 |
| FlashMLA / DeepGEMM / DeepEP | DeepSeek's kernel libraries for attention, matmul and expert communication | Hand-tuned native drivers | 3 |
| Continuous batching | Requests join and leave the batch at every step | A connection pool rather than waiting for a full batch | 4 |
| Paged KV / block manager | KV cache in fixed-size blocks, with a block table per request | Virtual memory pages; a slab allocator with ref counts | 4 |
| Prefix caching | Reusing KV for identical prompt prefixes across requests | A content-addressed cache for shared system prompts | 4 |
| Chunked prefill | Splitting a long prompt across steps so decodes don't stall | Time-slicing a big job to protect tail latency | 4 |
| Preemption | Evicting a running request when memory runs out, recomputing it later | Load shedding / OOM eviction with retry | 4 |
| Scheduler | Picks which requests run in the next step | A job scheduler with admission control | 4 |
| Speculative decoding | A cheap drafter guesses k tokens; the big model verifies them in one pass | Optimistic concurrency, or CPU speculative execution | 5 |
| MTP / draft tokens / DFlash | Built-in drafter heads, their guesses, and DeepSeek's drafter | A prefetcher and its prefetched pages | 5 |
| Confidence-scheduled verification (DSpark) | Verify only the guesses likely to be accepted | Adaptive prefetch depth | 5 |
| Structured output | Masking out tokens that would break a grammar or JSON schema | Schema validation enforced on every write | 5 |
| MoE / expert / router (gate) | Many sub-networks; a router sends each token to a few | Shards plus a router that picks shards per key | 6 |
| DP (data parallelism) | Independent replicas, each serving its own requests | Horizontal scaling behind a load balancer | 6 |
| TP (tensor parallelism) | Every matrix split across GPUs; combine results each layer | One query split across nodes with a scatter-gather per stage | 6 |
| EP (expert parallelism) | Experts spread across GPUs; tokens are sent to their expert's GPU | A sharded service with fan-out | 6 |
| CP (context parallelism) | One long prompt split across GPUs | Partitioning a single huge job | 6 |
| Dispatch / combine (all-to-all) | Sending tokens to expert GPUs and bringing results back | The shuffle phase in MapReduce | 6 |
| EPLB | Replicate hot experts and rebalance them periodically | Hot-key replication and shard rebalancing | 6 |
| NVLink / scale-up domain | Very fast GPU-to-GPU links within a node or rack | Same-rack network vs cross-datacenter | 6 |
| MLA / compressed KV | Caching a small compressed vector instead of full keys and values | Storing a compressed representation | 7 |
| SWA (sliding-window attention) | Attending only to the last N tokens | A ring buffer | 7 |
| Sparse attention / top-k indexer | A cheap index picks which past tokens to attend to | A secondary index before a full scan | 7 |
| YOCO / SWA bounded replay | Later layers reuse earlier layers' cache; recompute only a short suffix | A materialized view computed once and read downstream | 7 |
| PD disaggregation | Separate clusters for prefill and decode, with KV cache shipped between | Splitting the write path from the read path | 8 |
| ZOS / async scheduling | The CPU prepares step N+1 while the GPU runs step N | Pipelining / double buffering | 8 |
| Batch invariance | Results don't depend on which other requests share the batch | Reproducible builds | 8 |

## Sources

- [DeepInfer](https://github.com/deepseek-ai/DeepInfer) and its `doc/inference_deployment_qa.md`
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) (Karpathy)
- [3Blue1Brown deep learning lessons](https://www.3blue1brown.com/lessons/gpt)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) ([repo](https://github.com/jax-ml/scaling-book))
- [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)
- [GPU MODE lectures](https://github.com/gpu-mode/lectures)
- [Inference Engineering](https://www.baseten.co/inference-engineering/book/00-inference.md) (Philip Kiely, Baseten)
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) (Lilian Weng): an optional survey to read after Stage 3
- Papers: [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), [PagedAttention](https://arxiv.org/abs/2309.06180), [Speculative Decoding](https://arxiv.org/abs/2211.17192), [DistServe](https://arxiv.org/abs/2401.09670), [YOCO](https://arxiv.org/abs/2405.05254)