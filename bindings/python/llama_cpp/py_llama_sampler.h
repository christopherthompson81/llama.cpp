#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llama.h"
#include "py_llama_context.h"

namespace py = pybind11;

// LlamaSampler wrapper class
class PyLlamaSampler {
private:
    llama_sampler* sampler;

public:
    PyLlamaSampler();
    ~PyLlamaSampler();

    void add_greedy();
    void add_top_k(int k);
    void add_top_p(float p);
    void add_temperature(float temp);
    void add_mirostat(float tau, float eta, int m);
    /**
     * @brief Add a grammar-based sampler to the chain
     * @param grammar_str The grammar rules as a string
     */
    void add_grammar(const std::string& grammar_str);
    
    /**
     * @brief Sample a token from the context's logits
     * @param ctx_ptr Pointer to the llama_context
     * @param last_tokens Vector of previous tokens (unused in current implementation)
     * @return The sampled token
     */
    llama_token sample(void* ctx_ptr, const std::vector<llama_token>& last_tokens = {});
    
    /**
     * @brief Get the raw context pointer from a PyLlamaContext
     * @param ctx The Python context wrapper
     * @return Raw pointer to the underlying llama_context
     */
    void* get_context_ptr(PyLlamaContext& ctx);
    void accept(llama_token token);
};
