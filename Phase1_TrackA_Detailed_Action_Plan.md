# PHASE 1 — TRACK A

## Inference Internals Deep Dive Action Plan

*A practical, week-by-week guide to mastering local AI inference from the ground up*
*Covers: llama.cpp codebase study, GGUF format internals, and hands-on profiling*

---

| | |
|---|---|
| **Duration** | 12 weeks (Months 1–3) |
| **Time Commitment** | 10–15 hours per week |
| **Primary Tool** | llama.cpp (C/C++) |
| **Deliverable** | Model-agnostic inference service + resource report |
| **Prerequisites** | C/C++ or Rust reading ability, Linux CLI comfort |

---

## Overview and Goals

This plan breaks down Phase 1 Track A of the AI Systems Security Engineering Roadmap into twelve weeks of structured, hands-on work. Each week has a clear focus, concrete tasks, and measurable outputs. The goal is not just theoretical understanding but the ability to trace, inspect, and reason about every layer of the inference stack.

By the end of this track, you will be able to explain exactly what happens from the moment a GGUF model file is opened to the moment a generated token reaches the output buffer, and you will have built tooling that proves your understanding.

---

## How to Use This Plan

Each week lists focus areas, tasks, and outputs. Tasks marked with a wrench emoji (🔧) are hands-on builds. Tasks marked with a book emoji (📖) are reading/study. Tasks marked with a microscope (🔬) are profiling and investigation. You do not need to complete everything perfectly before moving on—progress over perfection.

---

## Weeks 1–2: Environment Setup and llama.cpp Orientation

### Objective

Get a working development environment, compile llama.cpp from source, run your first inference, and begin understanding the project structure and build system.

### Tasks

#### Environment Setup

- 🔧 Set up a Linux development machine (native or VM). Ubuntu 22.04+ recommended. If using macOS, ensure Xcode command line tools are installed.
- 🔧 Install build dependencies: cmake, gcc/clang, make, git, python3 (for conversion scripts), and optional CUDA toolkit if you have an NVIDIA GPU.
- 🔧 Install profiling tools: htop, perf, strace, ltrace, and nvidia-smi (if applicable). Verify each tool works.
- 🔧 Download 2–3 small GGUF models from Hugging Face for testing. Recommended starting points: a Q4_0 and a Q4_K_M quantized variant of a 7B-parameter model (e.g., Mistral 7B or Llama 3 8B).

#### Build and First Run

- 🔧 Clone the llama.cpp repository. Read the README and Makefile/CMakeLists.txt before building.
- 🔧 Compile with cmake in both Debug and Release modes. Note the difference in binary sizes and build flags.
- 🔧 Run inference using the main CLI binary with explicit parameters: set context size (-c), batch size (-b), thread count (-t), and temperature. Run the same prompt with different parameter combinations.
- 🔬 While inference runs, observe resource usage in a second terminal: htop for CPU/memory, nvidia-smi for GPU, and note resident memory size (RSS) vs. virtual memory size (VSZ).

#### Project Structure Orientation

- 📖 Map the top-level directory structure. Identify key directories: src/ (core), examples/ (CLI tools), ggml/ or ggml-backend/ (tensor library), common/ (shared utilities).
- 📖 Read through the Makefile or CMakeLists.txt to understand build targets, compile flags, and backend selection (CPU, CUDA, Metal, Vulkan).
- 📖 Identify the main entry points: which source file contains main() for the CLI? Trace a single function call from main to the first model-related operation.

### 🎯 Week 1–2 Output

A compiled llama.cpp binary in both Debug and Release modes. A short log documenting: build flags used, model files downloaded, first inference run parameters, and observed resource usage (screenshot or text notes). Annotated directory map of the llama.cpp project.

---

## Weeks 3–4: Model Loading and Memory Architecture

### Objective

Understand how a GGUF model file goes from disk into usable memory, including the mmap vs. full-load decision, memory layout, and how tensors are organized.

### Tasks

#### Model Loading Path

- 📖 Trace the execution path from the CLI entry point through model loading. Find the function that opens the GGUF file and reads the header. Follow it to where tensors are mapped into memory.
- 📖 Study how llama.cpp decides between mmap and full memory load. Find the relevant flags and conditions in the source code. Understand why mmap matters for large models: it lets the OS page in tensor data on demand rather than allocating everything upfront.
- 🔬 Run inference with mmap enabled and disabled (--mlock flag, or modifying source). Compare RSS memory usage and startup time for each mode. Log the difference.

#### Memory Layout

- 📖 Study how tensors (weight matrices) are laid out in memory after loading. How does the code index into specific layers? Where does the KV cache get allocated relative to model weights?
- 🔬 Use strace to observe system calls during model loading: open, mmap, madvise, mlock. Count and categorize the calls. This reveals the OS-level mechanics.
- 🔧 Write a small C program that mmap()s a GGUF file, reads the first few bytes of the header, and prints the magic number and version. This forces you to understand the file-to-memory interface at the syscall level.

#### Backend and Compute Graph

- 📖 Read through the ggml backend abstraction. Understand how the same model can run on CPU, CUDA, or Metal through a backend interface. Identify the key functions: tensor creation, graph computation, memory allocation per backend.
- 📖 Locate where the compute graph is built for a forward pass. How does llama.cpp represent the transformer architecture as a directed graph of tensor operations?

### 🎯 Week 3–4 Output

An annotated call trace from the CLI entry point through model loading to tensor allocation (document as a text/diagram with function names, file references, and brief descriptions of each step). A comparison table: mmap vs. full load showing RSS, startup time, and strace syscall counts. Your small C program that reads GGUF header bytes.

---

## Weeks 5–6: GGUF Format Deep Dive

### Objective

Understand the binary structure of GGUF files at the byte level. Build a parsing tool that can inspect any GGUF file and report its contents. This knowledge is foundational for the capstone's model attestation and tampering detection features.

### Tasks

#### GGUF Specification Study

- 📖 Read the GGUF specification document (available in the llama.cpp repository or ggml wiki). Map out the binary layout: magic number, version, tensor count, metadata key-value pairs, tensor info array, padding/alignment, and tensor data blob.
- 📖 Study the metadata schema: what keys are standard (general.architecture, general.name, tokenizer.ggml.model)? What data types are supported (uint8, int32, float32, string, array)? How are strings length-prefixed?
- 🔬 Open a GGUF file in a hex editor (xxd or hexdump). Manually identify the magic bytes (0x47475546 for "GGUF"), the version field, tensor count, and the first metadata key-value pair. Annotate the hex dump.

#### Quantization Schemes

- 📖 Study the quantization types supported by GGUF: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K_M, Q5_K_S, Q6_K, and the newer importance-matrix quantizations. Understand the tradeoff: smaller quantization = less memory + faster inference, but lower output quality.
- 📖 Read how block quantization works in ggml: weights are grouped into blocks (typically 32 values), each block gets a scale factor, and individual weights are stored as low-bit integers relative to that scale. This is how a 4-bit quantized model stores tensor data.
- 🔬 Compare the same model in Q4_0 vs. Q4_K_M vs. Q8_0: file sizes, peak memory during inference, tokens/second, and subjective output quality on the same prompt. Build a comparison table.

#### Build a GGUF Parser

- 🔧 Write a command-line tool (in C, Rust, or Python) that takes a GGUF file path and prints: magic/version, all metadata key-value pairs, tensor count, and for each tensor: name, dimensions, quantization type, and byte offset in the file.
- 🔧 Extend the parser to compute and print the SHA-256 hash of the entire file and of the tensor data section separately. This is your first step toward model integrity verification (critical for the capstone's secure model loading pipeline).
- 🔧 Test your parser against at least 3 different GGUF files (different models, different quantizations). Verify your output matches the metadata shown by llama.cpp's own model info output.

### 🎯 Week 5–6 Output

A working GGUF parser tool that dumps full file metadata, tensor info, and SHA-256 hashes. An annotated hex dump of a GGUF header. A quantization comparison table (file size, RAM, tokens/sec, quality notes) for at least 3 quantization types.

---

## Weeks 7–8: The Inference Loop and KV Cache

### Objective

Understand the core inference loop: how a prompt is tokenized, how the forward pass computes attention and generates logits, how the KV cache avoids redundant computation, and how tokens are sampled from the output distribution.

### Tasks

#### Tokenization

- 📖 Study how llama.cpp loads and applies the tokenizer. Find where the vocabulary is read from the GGUF metadata. Trace a string input through the tokenization function to see how it becomes a sequence of integer token IDs.
- 🔬 Use llama.cpp's tokenize utility (or write a small wrapper) to tokenize several prompts and inspect the output. Compare tokenization of the same text across different models (different vocabularies, different BPE merges).

#### Forward Pass and Attention

- 📖 Trace the inference loop: from token IDs through embedding lookup, layer-by-layer transformer forward pass (attention + feed-forward), to final logit computation. Identify the main function that orchestrates one forward pass.
- 📖 Study the attention computation: how are Q, K, V matrices computed from input embeddings? How is the attention mask applied? How does the output get projected back? Read the ggml operations involved (ggml_mul_mat, ggml_soft_max, etc.).
- 🔬 Add logging or use a debugger (gdb) to print tensor shapes at each step of one forward pass for one layer. Document the shapes: input embedding → Q/K/V → attention output → feed-forward → layer output.

#### KV Cache Mechanics

- 📖 Study the KV cache implementation: where is it allocated? How does it grow as the context window fills? What happens when the context limit is reached (truncation, sliding window, ring buffer)?
- 🔬 Profile memory usage during a long generation (e.g., 2048 tokens). Plot or log how RSS memory grows as the KV cache fills. Calculate: for a given model and context size, how much memory does the KV cache consume? Verify your calculation against actual measurement.
- 📖 Understand the security implications: the KV cache contains representations of all prior context. If an attacker can read process memory, the KV cache leaks the full conversation. Note this for Phase 2 isolation work.

#### Token Sampling

- 📖 Study the sampling functions: how logits become token probabilities (softmax), how temperature scaling works, how top-k and top-p (nucleus) sampling filter the distribution, and how repetition penalties are applied.
- 🔬 Run the same prompt at temperature 0.0 (greedy), 0.7, and 1.5. Observe how output changes. Run at temperature 0.7 with different top-p values. Document the behavioral difference and explain it in terms of the sampling code you read.

### 🎯 Week 7–8 Output

A documented trace of one complete forward pass showing tensor shapes at each layer. A KV cache memory analysis: predicted vs. actual memory growth over context length. A sampling parameter experiment log showing output differences and the code-level explanation for each.

---

## Weeks 9–10: Performance Profiling and System Interaction

### Objective

Move from reading code to measuring real system behavior. Profile inference at the OS and hardware level. Build the resource consumption report that feeds into the capstone deliverable.

### Tasks

#### CPU and Memory Profiling

- 🔬 Use perf stat to capture hardware counters during inference: instructions retired, cache misses (L1, L2, LLC), branch mispredictions, and cycles per instruction (CPI). Compare these across quantization types.
- 🔬 Use perf record + perf report to generate a CPU profile. Identify the hottest functions during inference. Which ggml kernel consumes the most cycles? Is it matrix multiplication, attention computation, or something else?
- 🔬 Profile memory allocation patterns: use strace to count mmap/munmap/brk calls during one inference session. Use /proc/[pid]/smaps to get a detailed memory map showing which regions are the model weights, KV cache, and heap.

#### Thread and Concurrency Analysis

- 🔬 Run inference with different thread counts (-t 1, -t 4, -t 8, etc.) and measure tokens/second for each. Find the scaling curve and the point of diminishing returns. Explain the result: what limits parallelism in transformer inference?
- 📖 Study how llama.cpp parallelizes matrix multiplication across threads. Find the thread pool implementation. Understand the batch processing mode: how does processing a batch of tokens differ from one-at-a-time generation?

#### GPU Profiling (if applicable)

- 🔬 If using CUDA: measure GPU utilization, memory bandwidth, and kernel execution time using nvidia-smi and nvprof/nsight-compute. Identify how many layers are offloaded to GPU vs. remaining on CPU.
- 🔬 Experiment with partial GPU offloading (-ngl flag for number of GPU layers). Find the optimal split point for your hardware by measuring total tokens/second at different offload levels.

#### Build the Resource Report

- 🔧 Create a structured resource consumption report covering: peak RSS memory, tokens/second (prompt processing and generation separately), CPU utilization per core, GPU utilization and VRAM usage, cache miss rates, and thread scaling curve. Test across at least 2 models and 2 quantization types.

### 🎯 Week 9–10 Output

A structured resource consumption report (the deliverable referenced in the capstone milestone). A perf CPU profile identifying the top 5 hottest functions during inference. Thread scaling analysis with a plot or table. If GPU is used, an optimal layer offloading analysis.

---

## Weeks 11–12: Capstone Layer 1 — Inference Service Build

### Objective

Build the first layer of the capstone project: a model-agnostic inference service that wraps llama.cpp for local inference and includes a provider abstraction for cloud API fallback. This is the working foundation that Phases 2–4 will build on top of.

### Tasks

#### Architecture Design

- 🔧 Design the provider abstraction interface. Define a common trait/interface that both the local llama.cpp backend and a cloud API backend implement: load model, send prompt, receive streaming tokens, report resource usage.
- 🔧 Design the routing logic: how does the system decide whether to use local inference or fall back to a cloud API? Criteria should include: prompt length vs. local context window, estimated token cost for cloud, model capability matching, and a user-configurable preference (local-first, cloud-first, or cost-optimized).
- 📖 Review how OpenClaw's provider routing works (and failed). Your design should use API key authentication only—no subscription token proxying, no OAuth workarounds. Keys stored encrypted at rest.

#### Implementation

- 🔧 Implement the local backend by wrapping llama.cpp's C API (or using the server mode with HTTP). The wrapper should handle: model loading with the verified GGUF path, prompt tokenization, inference execution with configurable parameters, and token streaming to the caller.
- 🔧 Implement a cloud backend stub that calls a single provider's API (e.g., Anthropic or OpenAI) with proper API key authentication. It should implement the same interface as the local backend.
- 🔧 Implement the router that accepts an inference request, evaluates it against routing criteria, selects a backend, executes the request, and returns the result along with a metadata struct (which backend was used, tokens consumed, latency, estimated cost).
- 🔧 Integrate your GGUF parser from Week 5–6 as the model inspection step during loading. Before inference begins, the system should log model metadata (name, architecture, quantization type, tensor count, SHA-256 hash).

#### Testing and Documentation

- 🔧 Write integration tests: local model loads and responds, cloud fallback triggers when local model can't handle the request, routing selects the correct backend for different prompt types.
- 🔧 Write a README documenting the architecture, how to add new backends, and the routing decision criteria. Include an architecture diagram.

### 🎯 Week 11–12 Output (Capstone Milestone)

A working model-agnostic inference service in Rust or C with: local llama.cpp backend, cloud API fallback with proper API key auth, cost-aware routing logic, model metadata inspection on load (including SHA-256), a structured resource consumption report per request, and a provider abstraction that supports adding new backends without changing the router. This is the Phase 1 capstone deliverable.

---

## Quick Reference: Key Files in llama.cpp

This table maps the concepts you need to study to the likely source file locations in the llama.cpp repository. File paths may shift between releases, so use these as starting points and search the codebase if they've moved.

| Concept | Where to Look | What to Study |
|---|---|---|
| CLI entry point | `examples/main/main.cpp` | Argument parsing, model load call, inference loop invocation |
| Model loading | `src/llama.cpp` or `llama-model.cpp` | GGUF header parsing, mmap setup, tensor allocation |
| GGUF parsing | `ggml/src/gguf.c` or `ggml.c` | Binary format reading, metadata extraction |
| KV cache | `src/llama.cpp` (search "kv_cache") | Cache allocation, growth, context limit handling |
| Sampling | `common/sampling.cpp` | Temperature, top-k, top-p, repetition penalty |
| Backend abstraction | `ggml/src/ggml-backend.c` | Tensor allocation per device, graph execution |
| Compute graph | `src/llama.cpp` (search "build_graph") | Transformer layer as ggml operations |
| Thread pool | `ggml/src/ggml.c` (search "thread") | Work splitting for matrix multiply |

---

## Suggested Weekly Schedule

For a 12-hour week (adjust proportionally for your actual hours):

| Day | Activity | Hours |
|---|---|---|
| Monday | Source code reading and annotation | 2 |
| Tuesday | Hands-on lab (building, profiling, experiments) | 2.5 |
| Wednesday | Hands-on lab continued | 2.5 |
| Thursday | Capstone work (design docs, coding, testing) | 2.5 |
| Saturday | Review, notes, write-up, and weekend deep dive | 2.5 |

---

## Tip: Keep a Learning Journal

Maintain a Markdown file (or repo) where you log each session: what you studied, what you built, what surprised you, and what questions remain open. This becomes invaluable during Phase 5 when you write technical content, and it forces you to articulate understanding rather than just pattern-matching.

---

## Success Criteria

At the end of Week 12, you should be able to confidently answer all of the following questions. If you can't answer one, loop back to the relevant week.

- What system calls does llama.cpp use to load a model into memory, and why?
- What is the binary layout of a GGUF file, and how do you verify its integrity?
- How does Q4_K_M quantization work at the block level, and what's the quality tradeoff vs. Q8_0?
- What is the KV cache, how does it grow during generation, and what are the memory implications?
- How does temperature affect the softmax output distribution, and how does top-p further filter it?
- What are the top 3 hottest functions during inference on your hardware, and why?
- How does thread count affect throughput, and what's the bottleneck that limits scaling?
- How would you detect if a GGUF model file had been tampered with after download?
- What data is stored in the KV cache, and what are the security implications if process memory is readable by another process?
- How does your inference service decide when to route to local vs. cloud, and what's the auth model for cloud keys?

---

*End of Phase 1 Track A — Detailed Action Plan*
