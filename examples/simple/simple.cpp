// ============================================================================
// llama.cpp Simple Example — Fully Annotated for Learning
// ============================================================================
//
// This file demonstrates the COMPLETE API call sequence for running inference
// with llama.cpp. The pipeline is:
//
//   1. ggml_backend_load_all()       — load compute backends (CPU, CUDA, Metal, etc.)
//   2. llama_model_load_from_file()  — load GGUF model file into memory
//   3. llama_model_get_vocab()       — get the vocabulary (tokenizer) from the model
//   4. llama_tokenize()              — convert text prompt into token IDs
//   5. llama_init_from_model()       — create a context (allocates KV cache, scratch buffers)
//   6. llama_sampler_chain_init()    — create a sampler pipeline for choosing next tokens
//   7. llama_decode()                — run the transformer forward pass (tokens → logits)
//   8. llama_sampler_sample()        — pick the next token from the logits
//   9. Loop steps 7-8 until done
//  10. Free everything
//
// ============================================================================

// #include brings in external code (like "import" in Python/JS)
#include "llama.h"       // The main llama.cpp public API — all llama_* functions
#include <clocale>       // C locale functions (number formatting)
#include <cstdio>        // C standard I/O: printf, fprintf, fflush
#include <cstring>       // C string functions: strcmp
#include <string>        // C++ std::string — a safer, resizable string type
#include <vector>        // C++ std::vector — a dynamic array (like Python list)

// "static" means this function is only visible within this file.
// "void" means it returns nothing. "int" and "char **" are parameter types.
// "char ** argv" is an array of C-strings (the command-line arguments).
static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

// main() is the entry point. argc = argument count, argv = argument values.
int main(int argc, char ** argv) {
    // Set numeric locale to "C" so decimal points are always "." (not "," in some locales)
    std::setlocale(LC_NUMERIC, "C");

    // ========================================================================
    // STEP 0: Parse command-line arguments
    // ========================================================================
    // These are local variables with default values. std::string is like a
    // Python string; int is a 32-bit integer.

    std::string model_path;                    // path to the .gguf model file (required)
    std::string prompt = "Hello my name is";   // default prompt text
    int ngl = 99;                              // GPU layers to offload (99 = as many as possible)
    int n_predict = 32;                        // how many new tokens to generate

    // --- Argument parsing block ---
    // The outer { } creates a scope — variables declared inside (like "i") are
    // destroyed when the scope ends, keeping the main function's namespace clean.
    {
        int i = 1; // start at 1 because argv[0] is the program name itself
        // "for" loop: start at i=1, continue while i < argc, increment i each iteration
        for (; i < argc; i++) {
            // strcmp compares two C-strings; returns 0 if they are equal
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    // "++i" increments i FIRST, then uses it — so this reads the NEXT argument
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1; // return 1 from main = exit with error code
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    // "try/catch" handles errors. std::stoi converts string to int.
                    // If the string isn't a valid number, stoi throws an exception,
                    // which catch(...) catches (the "..." means catch any exception).
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // Any argument that isn't a flag is treated as the start of the prompt
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        // If there are remaining arguments, join them as the prompt string
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";  // "+=" on std::string appends (like Python's +=)
                prompt += argv[i];
            }
        }
    }

    // ========================================================================
    // STEP 1: Load compute backends
    // ========================================================================
    // ggml is the tensor/compute library underneath llama.cpp. This call
    // discovers and loads all available backends: CPU, CUDA (NVIDIA GPU),
    // Metal (Apple GPU), Vulkan, etc. Must be called before loading a model.
    ggml_backend_load_all();

    // ========================================================================
    // STEP 2: Load the model from a GGUF file
    // ========================================================================
    // First, create a params struct with default values, then customize it.
    // In C++, "llama_model_params" is a struct (a bundle of named fields).
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl; // how many transformer layers to run on GPU

    // Load the model. This is the heaviest step — it:
    //   1. Parses the GGUF file header (tensor names, shapes, quantization types)
    //   2. Memory-maps (mmap) or reads the weight data into RAM
    //   3. Optionally uploads layers to GPU VRAM based on n_gpu_layers
    //
    // "llama_model *" means a pointer to a llama_model object.
    // Pointers are like references/handles — they hold a memory address.
    // ".c_str()" converts std::string to a C-style char* that the C API expects.
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    // NULL (or nullptr) means the pointer is empty — the load failed.
    if (model == NULL) {
        // fprintf(stderr, ...) prints to standard error (not stdout).
        // __func__ is a built-in macro that expands to the current function name ("main").
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // ========================================================================
    // STEP 3: Get the vocabulary (tokenizer) from the loaded model
    // ========================================================================
    // The vocab is embedded in the GGUF file. It defines how text maps to token IDs.
    // "const" means this pointer won't be modified — it's read-only.
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ========================================================================
    // STEP 4: Tokenize the prompt (text → token IDs)
    // ========================================================================
    // Tokenization is done in TWO calls:
    //
    // Call 1: Pass NULL buffer and size 0 to ask "how many tokens would this produce?"
    //         The function returns the NEGATIVE count (hence the leading minus sign).
    //         This is a common C pattern — negative return = "I need this many slots".
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    //                                                                                      ^^^^  ^^^^
    //                                                               add_special (BOS token) ┘      │
    //                                                           parse_special (handle <s> etc.) ────┘

    // Call 2: Now allocate a vector of the right size and actually tokenize.
    // std::vector<llama_token> is a dynamic array of token IDs (int32).
    // prompt_tokens(n_prompt) creates the vector with n_prompt elements.
    std::vector<llama_token> prompt_tokens(n_prompt);
    // .data() returns a raw pointer to the vector's internal array.
    // .size() returns the number of elements.
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // ========================================================================
    // STEP 5: Create an inference context
    // ========================================================================
    // The context holds everything needed for inference:
    //   - KV cache (stores past key/value attention states so we don't recompute them)
    //   - Logits buffer (output probabilities for each token in the vocabulary)
    //   - Scratch/work buffers for the computation graph
    //
    // A single model can have MULTIPLE contexts (e.g., for parallel requests).

    llama_context_params ctx_params = llama_context_default_params();

    // n_ctx = total context window size (how many tokens the model can "see").
    // We need room for the prompt + the tokens we'll generate.
    ctx_params.n_ctx = n_prompt + n_predict - 1;

    // n_batch = max tokens processed in a single llama_decode() call.
    // For the prompt, we process all tokens at once (parallel prefill).
    // During generation, we process 1 token at a time.
    ctx_params.n_batch = n_prompt;

    // Enable performance counters (timing stats printed at the end)
    ctx_params.no_perf = false;

    // Create the context. Internally this:
    //   1. Allocates the KV cache (sized to n_ctx)
    //   2. Allocates compute buffers on the appropriate backend (CPU/GPU)
    //   3. Sets up the scheduler for splitting work across backends
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // ========================================================================
    // STEP 6: Set up the sampling pipeline
    // ========================================================================
    // Sampling = choosing which token to output next from the model's logits
    // (the raw output scores for every token in the vocabulary).
    //
    // llama.cpp uses a "sampler chain" — a pipeline of samplers applied in order.
    // Common chain: temperature → top_k → top_p → pick one token.
    // Here we use just "greedy" (always pick the highest-probability token).

    // "auto" tells the compiler to figure out the type automatically
    // (it's llama_sampler_chain_params). Like Python's duck typing but at compile time.
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false; // enable timing stats

    // Create an empty sampler chain
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // Add a greedy sampler to the chain (picks the token with highest logit)
    // Other options: llama_sampler_init_temp(), llama_sampler_init_top_k(), etc.
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Print the prompt text by converting each token ID back to its text piece.
    // "for (auto id : prompt_tokens)" is a range-based for loop (like Python's "for id in prompt_tokens").
    // "auto" deduces the type (llama_token, which is int32_t).
    for (auto id : prompt_tokens) {
        char buf[128];  // a stack-allocated buffer of 128 bytes for the text piece
        // llama_token_to_piece converts a token ID → its text representation
        // Returns the number of bytes written, or negative on error.
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        // std::string(buf, n) creates a string from the first n bytes of buf
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // ========================================================================
    // STEP 7: Prepare the initial batch
    // ========================================================================
    // A "batch" packages tokens to be processed together by llama_decode().
    // llama_batch_get_one() creates a simple batch from a contiguous array of tokens.
    // For the prompt, ALL tokens go in one batch (parallel prefill).
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // --- Encoder-decoder model support (e.g., T5, BART) ---
    // Most LLMs (GPT, LLaMA) are decoder-only, so this block is skipped.
    // For encoder-decoder models: encode the prompt first, then start decoding.
    if (llama_model_has_encoder(model)) {
        // llama_encode() runs the encoder forward pass
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        // Get the special "start of decoder" token
        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab); // fallback to BOS (beginning of sequence)
        }

        // Replace the batch with just the decoder start token
        // "&decoder_start_token_id" takes the address of this variable (creates a pointer to it)
        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // ========================================================================
    // STEP 8: The main generation loop (decode → sample → repeat)
    // ========================================================================
    // This is where text generation actually happens. Each iteration:
    //   1. llama_decode() — run the transformer forward pass on the batch
    //      - First iteration: processes ALL prompt tokens at once (prefill)
    //      - Subsequent iterations: processes just 1 new token (autoregressive generation)
    //      - Internally: builds a ggml computation graph, executes it on CPU/GPU,
    //        stores K/V states in the cache, outputs logits
    //   2. llama_sampler_sample() — select the next token from the output logits
    //   3. Check for end-of-generation (EOS/EOT tokens)
    //   4. Prepare a new batch with just the sampled token for the next iteration

    const auto t_main_start = ggml_time_us(); // microsecond timer for benchmarking
    int n_decode = 0;          // count of generated tokens (for speed calculation)
    llama_token new_token_id;  // will hold each newly sampled token

    // Loop until we've generated enough tokens.
    // n_pos tracks our position in the context window.
    // Note: the loop has NO increment expression — n_pos is updated inside the body.
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {

        // --- DECODE: Run the transformer forward pass ---
        // This is the most expensive call. It:
        //   1. Looks up token embeddings
        //   2. Runs through all transformer layers (attention + feed-forward)
        //   3. Stores K and V tensors in the KV cache for future steps
        //   4. Produces logits (unnormalized scores) for the NEXT token
        //
        // On the first call, batch has all prompt tokens (e.g., 5 tokens).
        // On subsequent calls, batch has just 1 token (the one we just sampled).
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens; // advance our position by however many tokens we just processed

        // --- SAMPLE: Choose the next token ---
        // The { } block scopes the sampling variables.
        {
            // llama_sampler_sample() does:
            //   1. Reads the logits from the context (for the last token position, index -1)
            //   2. Builds a "candidates" array: one entry per vocab token with its logit score
            //   3. Applies each sampler in the chain (here just greedy → picks max)
            //   4. Returns the chosen token ID
            new_token_id = llama_sampler_sample(smpl, ctx, -1);
            //                                              ^^
            //                       -1 means "logits for the last token in the batch"

            // Check if this is an end-of-generation token (EOS, EOT, etc.)
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break; // stop generating
            }

            // Convert the token ID back to text and print it immediately
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout); // force output to display NOW (don't buffer it)

            // Prepare the next batch with JUST the sampled token.
            // "&new_token_id" creates a pointer to this single token.
            // On the next loop iteration, llama_decode() will process only this 1 token,
            // using the KV cache for all previous tokens (no recomputation needed).
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    // ========================================================================
    // STEP 9: Print performance stats
    // ========================================================================
    const auto t_main_end = ggml_time_us();

    // Calculate and print generation speed (tokens per second)
    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);  // print sampler timing stats
    llama_perf_context_print(ctx);   // print context timing stats (prompt eval + generation)
    fprintf(stderr, "\n");

    // ========================================================================
    // STEP 10: Clean up — free all allocated resources
    // ========================================================================
    // In C/C++, you must manually free memory (no garbage collector).
    // Order matters: free things that depend on others first.
    llama_sampler_free(smpl);   // free the sampler chain
    llama_free(ctx);            // free the context (KV cache, buffers)
    llama_model_free(model);    // free the model (weights, vocab)

    return 0; // 0 = success
}
