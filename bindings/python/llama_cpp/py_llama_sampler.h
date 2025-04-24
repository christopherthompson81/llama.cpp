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

    void add_top_k(int k);
    void add_top_p(float p);
    void add_temperature(float temp);
    void add_mirostat(float tau, float eta, int m);
    void add_grammar(const std::string& grammar_str);
    llama_token sample(void* ctx_ptr, const std::vector<llama_token>& last_tokens);
    void* get_context_ptr(PyLlamaContext& ctx);
    void accept(llama_token token);
};
