#pragma once

#include "llama-context.h"
#include "ggml.h"
#include "stdint.h"

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Returns the projected memory use (model + context + compute) in bytes
// for the given device within this context. Returns 0 if the device is not used.
LLAMA_API uint64_t llama_context_device_memory(
        const struct llama_context * ctx,
        ggml_backend_dev_t           device);
