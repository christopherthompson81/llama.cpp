#include "py_llama_sampler.h"

PyLlamaSampler::PyLlamaSampler() {
    struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
    sampler = llama_sampler_chain_init(params);
    if (!sampler) {
        throw std::runtime_error("Failed to create sampler");
    }
}

PyLlamaSampler::~PyLlamaSampler() {
    if (sampler) {
        llama_sampler_free(sampler);
    }
}

void PyLlamaSampler::add_top_k(int k) {
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(k));
}

void PyLlamaSampler::add_top_p(float p) {
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(p, 1));
}

void PyLlamaSampler::add_temperature(float temp) {
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
}

void PyLlamaSampler::add_mirostat(float tau, float eta, int m) {
    const struct llama_vocab* vocab = llama_model_get_vocab(nullptr);
    int n_vocab = llama_vocab_n_tokens(vocab);
    llama_sampler_chain_add(sampler, llama_sampler_init_mirostat(n_vocab, 42, tau, eta, m));
}

void PyLlamaSampler::add_grammar(const std::string& grammar_str) {
    const struct llama_vocab* vocab = llama_model_get_vocab(nullptr);
    llama_sampler_chain_add(sampler, llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root"));
}

llama_token PyLlamaSampler::sample(void* ctx_ptr, const std::vector<llama_token>& last_tokens) {
    llama_context* ctx = static_cast<llama_context*>(ctx_ptr);
    return llama_sampler_sample(sampler, ctx, -1);
}

void* PyLlamaSampler::get_context_ptr(PyLlamaContext& ctx) {
    return ctx.get_context_ptr();
}

void PyLlamaSampler::accept(llama_token token) {
    llama_sampler_accept(sampler, token);
}
