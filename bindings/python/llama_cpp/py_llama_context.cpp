#include "py_llama_context.h"

PyLlamaContext::PyLlamaContext(PyLlamaModel& model, const py::dict& params_dict) : model(model) {
    llama_context_params params = llama_context_default_params();

    // Set parameters from Python dict
    if (params_dict.contains("n_ctx")) {
        params.n_ctx = params_dict["n_ctx"].cast<int>();
    }
    if (params_dict.contains("n_batch")) {
        params.n_batch = params_dict["n_batch"].cast<int>();
    }
    if (params_dict.contains("n_threads")) {
        params.n_threads = params_dict["n_threads"].cast<int>();
    }
    if (params_dict.contains("n_threads_batch")) {
        params.n_threads_batch = params_dict["n_threads_batch"].cast<int>();
    }
    if (params_dict.contains("rope_scaling_type")) {
        params.rope_scaling_type = (enum llama_rope_scaling_type) params_dict["rope_scaling_type"].cast<int>();
    }
    if (params_dict.contains("rope_freq_base")) {
        params.rope_freq_base = params_dict["rope_freq_base"].cast<float>();
    }
    if (params_dict.contains("rope_freq_scale")) {
        params.rope_freq_scale = params_dict["rope_freq_scale"].cast<float>();
    }
    if (params_dict.contains("yarn_ext_factor")) {
        params.yarn_ext_factor = params_dict["yarn_ext_factor"].cast<float>();
    }
    if (params_dict.contains("yarn_attn_factor")) {
        params.yarn_attn_factor = params_dict["yarn_attn_factor"].cast<float>();
    }
    if (params_dict.contains("yarn_beta_fast")) {
        params.yarn_beta_fast = params_dict["yarn_beta_fast"].cast<float>();
    }
    if (params_dict.contains("yarn_beta_slow")) {
        params.yarn_beta_slow = params_dict["yarn_beta_slow"].cast<float>();
    }
    if (params_dict.contains("defrag_thold")) {
        params.defrag_thold = params_dict["defrag_thold"].cast<float>();
    }
    if (params_dict.contains("embeddings")) {
        params.embeddings = params_dict["embeddings"].cast<bool>();
    }
    if (params_dict.contains("offload_kqv")) {
        params.offload_kqv = params_dict["offload_kqv"].cast<bool>();
    }

    ctx = llama_init_from_model(model.get_model(), params);
    if (!ctx) {
        throw std::runtime_error("Failed to create context");
    }
}

PyLlamaContext::~PyLlamaContext() {
    if (ctx) {
        llama_free(ctx);
    }
}

void* PyLlamaContext::get_context_ptr() const {
    return static_cast<void*>(ctx);
}

bool PyLlamaContext::decode(const PyLlamaBatch& batch) {
    return llama_decode(ctx, batch.get_batch()) == 0;
}

py::array_t<float> PyLlamaContext::get_logits() {
    float* logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    
    return py::array_t<float>(
        {n_vocab},                  // Shape
        {sizeof(float)},            // Strides
        logits,                     // Data pointer
        py::capsule(nullptr, [](void*) {})  // No ownership transfer
    );
}

py::array_t<float> PyLlamaContext::get_embeddings() {
    float* embeddings = llama_get_embeddings(ctx);
    if (!embeddings) {
        throw std::runtime_error("Embeddings not available. Make sure to set embeddings=True in context params.");
    }
    
    int n_embd = llama_model_n_embd(llama_get_model(ctx));
    
    return py::array_t<float>(
        {n_embd},                   // Shape
        {sizeof(float)},            // Strides
        embeddings,                 // Data pointer
        py::capsule(nullptr, [](void*) {})  // No ownership transfer
    );
}

void PyLlamaContext::kv_cache_clear() {
    llama_kv_self_clear(ctx);
}

void PyLlamaContext::set_n_threads(int n_threads, int n_threads_batch) {
    llama_set_n_threads(ctx, n_threads, n_threads_batch);
}
