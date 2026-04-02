# Learning Path 1: llama.cpp Internals

Understanding how llama.cpp works under the hood — from model loading to token output.

---

## Recommended Reading Order

### Phase 1: Start Here — The Simplest End-to-End Example
**Goal:** See the full pipeline in ~100 lines before diving into any internals.

- **`examples/simple/simple.cpp`** — Read this first, top to bottom. It shows the complete API call sequence:
  1. `llama_model_load_from_file()` — load model
  2. `llama_tokenize()` — text to tokens
  3. `llama_init_from_model()` — create context (allocates KV cache)
  4. `llama_sampler_chain_init()` + `llama_sampler_chain_add()` — set up sampling
  5. `llama_decode()` — forward pass
  6. `llama_sampler_sample()` — pick next token
  7. Loop until EOS

- **`include/llama.h`** — Skim the public C API. Focus on the `llama_model_params` and `llama_context_params` structs to understand what knobs exist (especially `use_mmap`, `n_gpu_layers`, `n_ctx`).

### Phase 2: Model Loading (mmap vs. full load)
**Goal:** Understand how a GGUF file becomes tensors in memory.

Read in this order:

1. **`src/llama-model-loader.h`** — The `llama_model_loader` class definition. Note the `weights_map`, `mappings`, and the `use_mmap` flag.

2. **`src/llama-model-loader.cpp`** — The heavy lifting:
   - Constructor (~line 502): calls `gguf_init_from_file()` to parse GGUF metadata (tensor names, shapes, types, offsets). No data loaded yet.
   - `init_mappings()` (~line 898): if `use_mmap=true`, creates `llama_mmap` objects that memory-map the file. If false, prepares for direct I/O.
   - `load_all_data()` (~line 971): the key function. Two paths:
     - **mmap path**: tensor data pointers point directly into the mapped file region
     - **full-load path**: reads from disk into host buffers, async-uploads to GPU

3. **`src/llama-mmap.h` / `src/llama-mmap.cpp`** — The `llama_mmap` wrapper around the OS `mmap()` syscall. Short and focused.

4. **`src/llama-model.cpp`** — `load_tensors()` (~line 2693): where tensors are created per layer and assigned to devices (CPU vs GPU split based on `n_gpu_layers` and available VRAM).

**Call chain summary:**
```
llama_model_load_from_file()        [llama.cpp:1029]
  -> llama_model_load()              [llama.cpp:828]
    -> llama_model_loader()          [llama-model-loader.cpp:502]  -- parse GGUF
    -> model->load_tensors(ml)       [llama-model.cpp:2693]        -- allocate buffers
    -> ml.init_mappings()            [llama-model-loader.cpp:898]   -- mmap or prepare I/O
    -> ml.load_all_data()            [llama-model-loader.cpp:971]   -- data into memory
```

### Phase 3: KV Cache
**Goal:** Understand how past token representations are stored and reused.

1. **`src/llama-kv-cache.h`** — The `llama_kv_cache` class. Key concepts:
   - `kv_layer` struct: holds one `ggml_tensor * k` and `ggml_tensor * v` per layer
   - `llama_kv_cells`: tracks which positions/sequences occupy each cache slot

2. **`src/llama-kv-cells.h`** — Cell-level metadata: position tracking, sequence assignment, shift accumulation.

3. **`src/llama-kv-cache.cpp`** (~3000 lines) — Key functions:
   - `prepare(ubatches)` — find free slots for incoming tokens (ring buffer search)
   - `find_slot(ubatch)` — locate contiguous cache space
   - `apply_ubatch()` — write token metadata into selected cells
   - `cpy_k()` / `cpy_v()` — store K/V tensors into cache during inference
   - `update()` — apply queued shifts and updates after computation

### Phase 4: Sampling
**Goal:** Understand how logits become a chosen token.

1. **`src/llama-sampler.cpp`** (~3885 lines) — All sampling strategies implemented here:
   - Each sampler implements the `llama_sampler_i` interface (especially `apply()`)
   - `llama_sampler_chain` applies samplers sequentially
   - Key samplers to study: `greedy`, `dist` (random), `top_k`, `top_p`, `temp`, `min_p`
   - `llama_sampler_sample()` (~line near end): builds candidate array from logits, applies chain, returns selected token

2. **`common/sampling.h`** — Higher-level wrapper used by examples. Shows how sampler chains are typically configured.

### Phase 5: The Inference Loop (Decode)
**Goal:** Understand the forward pass from tokens to logits.

1. **`src/llama-context.h`** (~line 129-130) — `encode()` and `decode()` declarations. Note the `llama_context` holds the KV cache (`memory`), logits buffer, and embeddings buffer.

2. **`src/llama-context.cpp`** (~line 3335) — `decode()` implementation:
   - Splits batch into micro-batches (`ubatches`)
   - Calls `kv_cache->prepare()` to allocate cache slots
   - Builds the computation graph (the transformer forward pass)
   - Executes the graph on the backend (CPU/GPU)
   - Stores K/V into cache, outputs logits

3. **`src/llama-model.cpp`** — `build_graph()` / the graph-building functions show how attention, FFN, and normalization layers are wired together using ggml ops.

---

## Key Structs to Keep in Mind

| Struct | Where | What |
|--------|-------|------|
| `llama_model_params` | `include/llama.h` | `use_mmap`, `n_gpu_layers`, tensor split |
| `llama_context_params` | `include/llama.h` | `n_ctx`, `n_batch`, type_k/type_v |
| `llama_batch` | `include/llama.h` | Token IDs + positions + sequence IDs |
| `llama_token_data_array` | `include/llama.h` | Candidate tokens for sampling |
| `llama_model_loader` | `src/llama-model-loader.h` | GGUF parsing, weight mapping |
| `llama_kv_cache` | `src/llama-kv-cache.h` | K/V storage per layer |
| `llama_sampler_chain` | `src/llama-sampler.cpp` | Sequential sampler pipeline |

---

## Verification / Hands-On Exercises

1. **Build and run the simple example** with a small model (e.g., TinyLlama GGUF). Add `printf` statements in key functions to see the call order.
2. **Toggle `use_mmap`** in model params and observe memory behavior (use `htop` or Activity Monitor).
3. **Add a print in `llama_sampler_sample()`** to see the top-5 candidates and their probabilities before final selection.
4. **Set a breakpoint in `load_all_data()`** to see the mmap vs. read path in action.
5. **Print KV cache utilization** after each decode step by logging `kv_cache` cell occupancy.
