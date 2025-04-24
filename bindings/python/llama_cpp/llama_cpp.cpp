#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "llama.h"

namespace py = pybind11;

// Forward declarations of wrapper classes
class PyLlamaModel;
class PyLlamaContext;
class PyLlamaBatch;
class PyLlamaSampler;
class PyLlamaVocab;

// LlamaModel wrapper class
class PyLlamaModel {
private:
    llama_model* model;
    bool owns_model;

public:
    PyLlamaModel(const std::string& path) {
        llama_model_params params = llama_model_default_params();
        model = llama_model_load_from_file(path.c_str(), params);
        if (!model) {
            throw std::runtime_error("Failed to load model from " + path);
        }
        owns_model = true;
    }

    PyLlamaModel(llama_model* model, bool owns_model = false) : model(model), owns_model(owns_model) {}

    ~PyLlamaModel() {
        if (owns_model && model) {
            llama_model_free(model);
        }
    }

    llama_model* get_model() const {
        return model;
    }

    int n_ctx_train() const {
        return llama_model_n_ctx_train(model);
    }

    int n_embd() const {
        return llama_model_n_embd(model);
    }

    int n_layer() const {
        return llama_model_n_layer(model);
    }

    int n_head() const {
        return llama_model_n_head(model);
    }

    int n_head_kv() const {
        return llama_model_n_head_kv(model);
    }

    PyLlamaVocab get_vocab() const;

    std::string meta_val(const std::string& key) const {
        char buf[256];
        int ret = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
        return ret >= 0 ? std::string(buf) : "";
    }

    std::string chat_template() const {
        const char* tmpl = llama_model_chat_template(model, nullptr);
        return tmpl ? std::string(tmpl) : "";
    }
};

// LlamaContext wrapper class
class PyLlamaContext {
private:
    llama_context* ctx;
    PyLlamaModel model;

public:
    PyLlamaContext(PyLlamaModel& model, const py::dict& params_dict) : model(model) {
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

    ~PyLlamaContext() {
        if (ctx) {
            llama_free(ctx);
        }
    }
    
    void* get_context_ptr() const {
        return static_cast<void*>(ctx);
    }

    bool decode(const PyLlamaBatch& batch);

    py::array_t<float> get_logits() {
        float* logits = llama_get_logits(ctx);
        int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        
        return py::array_t<float>(
            {n_vocab},                  // Shape
            {sizeof(float)},            // Strides
            logits,                     // Data pointer
            py::capsule(nullptr, [](void*) {})  // No ownership transfer
        );
    }

    py::array_t<float> get_embeddings() {
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

    void kv_cache_clear() {
        llama_kv_self_clear(ctx);
    }

    void set_n_threads(int n_threads, int n_threads_batch) {
        llama_set_n_threads(ctx, n_threads, n_threads_batch);
    }
};

// LlamaBatch wrapper class
class PyLlamaBatch {
private:
    llama_batch batch;
    bool owns_batch;

public:
    PyLlamaBatch(int32_t n_tokens, int32_t embd = 0, int32_t n_seq_max = 1) {
        batch = llama_batch_init(n_tokens, embd, n_seq_max);
        owns_batch = true;
    }

    PyLlamaBatch(llama_batch batch, bool owns_batch = false) : batch(batch), owns_batch(owns_batch) {}

    ~PyLlamaBatch() {
        if (owns_batch) {
            llama_batch_free(batch);
        }
    }

    llama_batch get_batch() const {
        return batch;
    }

    py::array_t<llama_token> get_tokens() const {
        return py::array_t<llama_token>(
            {batch.n_tokens},
            {sizeof(llama_token)},
            batch.token,
            py::capsule(nullptr, [](void*) {})
        );
    }

    void set_tokens(py::array_t<llama_token> tokens) {
        if (tokens.size() > batch.n_tokens) {
            throw std::runtime_error("Token array too large for batch");
        }
        
        auto r = tokens.unchecked<1>();
        for (py::ssize_t i = 0; i < r.shape(0); i++) {
            batch.token[i] = r(i);
        }
        // Update n_tokens to match the actual number of tokens
        batch.n_tokens = r.shape(0);
    }

    py::array_t<llama_pos> get_positions() const {
        return py::array_t<llama_pos>(
            {batch.n_tokens},
            {sizeof(llama_pos)},
            batch.pos,
            py::capsule(nullptr, [](void*) {})
        );
    }

    void set_positions(py::array_t<llama_pos> positions) {
        if (positions.size() > batch.n_tokens) {
            throw std::runtime_error("Positions array too large for batch");
        }
        
        auto r = positions.unchecked<1>();
        for (py::ssize_t i = 0; i < r.shape(0); i++) {
            batch.pos[i] = r(i);
        }
    }

    py::array_t<int32_t> get_n_seq_id() const {
        return py::array_t<int32_t>(
            {batch.n_tokens},
            {sizeof(int32_t)},
            batch.n_seq_id,
            py::capsule(nullptr, [](void*) {})
        );
    }

    void set_n_seq_id(py::array_t<int32_t> n_seq_id) {
        if (n_seq_id.size() > batch.n_tokens) {
            throw std::runtime_error("n_seq_id array too large for batch");
        }
        
        auto r = n_seq_id.unchecked<1>();
        for (py::ssize_t i = 0; i < r.shape(0); i++) {
            batch.n_seq_id[i] = r(i);
        }
    }

    py::array_t<int8_t> get_logits() const {
        return py::array_t<int8_t>(
            {batch.n_tokens},
            {sizeof(int8_t)},
            batch.logits,
            py::capsule(nullptr, [](void*) {})
        );
    }

    void set_logits(py::array_t<int8_t> logits) {
        if (logits.size() > batch.n_tokens) {
            throw std::runtime_error("Logits array too large for batch");
        }
        
        auto r = logits.unchecked<1>();
        for (py::ssize_t i = 0; i < r.shape(0); i++) {
            batch.logits[i] = r(i);
        }
    }
};

// LlamaSampler wrapper class
class PyLlamaSampler {
private:
    llama_sampler* sampler;

public:
    PyLlamaSampler() {
        struct llama_sampler_chain_params params = llama_sampler_chain_default_params();
        sampler = llama_sampler_chain_init(params);
        if (!sampler) {
            throw std::runtime_error("Failed to create sampler");
        }
    }

    ~PyLlamaSampler() {
        if (sampler) {
            llama_sampler_free(sampler);
        }
    }

    void add_top_k(int k) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(k));
    }

    void add_top_p(float p) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(p, 1));
    }

    void add_temperature(float temp) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
    }

    void add_mirostat(float tau, float eta, int m) {
        const struct llama_vocab* vocab = llama_model_get_vocab(nullptr);
        int n_vocab = llama_vocab_n_tokens(vocab);
        llama_sampler_chain_add(sampler, llama_sampler_init_mirostat(n_vocab, 42, tau, eta, m));
    }

    void add_grammar(const std::string& grammar_str) {
        const struct llama_vocab* vocab = llama_model_get_vocab(nullptr);
        llama_sampler_chain_add(sampler, llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root"));
    }

    llama_token sample(void* ctx_ptr, const std::vector<llama_token>& last_tokens) {
        llama_context* ctx = static_cast<llama_context*>(ctx_ptr);
        return llama_sampler_sample(sampler, ctx, -1);
    }
    
    // Helper method to get the raw context pointer
    void* get_context_ptr(PyLlamaContext& ctx) {
        return ctx.get_context_ptr();
    }

    void accept(llama_token token) {
        llama_sampler_accept(sampler, token);
    }
};

// LlamaVocab wrapper class
class PyLlamaVocab {
private:
    const llama_model* model;
    const llama_vocab* vocab;
    bool owns_model;

public:
    PyLlamaVocab(llama_model* model, bool owns_model = false) : model(model), owns_model(owns_model) {
        vocab = llama_model_get_vocab(model);
    }

    ~PyLlamaVocab() {
        if (owns_model && model) {
            llama_model_free(const_cast<llama_model*>(model));
        }
    }

    std::vector<llama_token> tokenize(const std::string& text, bool add_bos = false, bool special = true) const {
        std::vector<llama_token> tokens(text.size() + 4); // +4 for safety
        int n_tokens = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), add_bos, special);
        
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), add_bos, special);
        }
        
        if (n_tokens < 0) {
            throw std::runtime_error("Tokenization failed");
        }
        
        tokens.resize(n_tokens);
        return tokens;
    }

    std::string detokenize(const std::vector<llama_token>& tokens) const {
        std::string result;
        for (const auto& token : tokens) {
            result += token_to_piece(token);
        }
        return result;
    }

    std::string token_to_piece(llama_token token) const {
        std::vector<char> buf(8, 0); // Start with a small buffer
        int n_chars = llama_token_to_piece(vocab, token, buf.data(), buf.size(), 0, true);
        
        if (n_chars < 0) {
            buf.resize(-n_chars);
            n_chars = llama_token_to_piece(vocab, token, buf.data(), buf.size(), 0, true);
        }
        
        if (n_chars < 0) {
            throw std::runtime_error("Token to piece conversion failed");
        }
        
        return std::string(buf.data(), n_chars);
    }

    llama_token bos_token() const {
        return llama_vocab_bos(vocab);
    }

    llama_token eos_token() const {
        return llama_vocab_eos(vocab);
    }

    int n_tokens() const {
        return llama_vocab_n_tokens(vocab);
    }
};

// Implementation of methods that require forward declarations
PyLlamaVocab PyLlamaModel::get_vocab() const {
    return PyLlamaVocab(model, false);
}

bool PyLlamaContext::decode(const PyLlamaBatch& batch) {
    return llama_decode(ctx, batch.get_batch()) == 0;
}

// Helper function for chat templates
std::string py_chat_apply_template(
    const PyLlamaModel& model,
    const std::vector<std::map<std::string, std::string>>& messages,
    const std::string& custom_template = "") {
    
    // Convert messages to C format
    std::vector<llama_chat_message> c_messages(messages.size());
    for (size_t i = 0; i < messages.size(); i++) {
        const auto& msg = messages[i];
        
        auto role_it = msg.find("role");
        auto content_it = msg.find("content");
        
        c_messages[i].role = role_it != msg.end() ? role_it->second.c_str() : "";
        c_messages[i].content = content_it != msg.end() ? content_it->second.c_str() : "";
    }
    
    // Allocate a buffer for the result
    std::vector<char> buf(65536, 0); // Start with 64KB buffer
    
    // Apply template
    int result = llama_chat_apply_template(
        custom_template.empty() ? nullptr : custom_template.c_str(),
        c_messages.data(),
        c_messages.size(),
        true,  // add_ass
        buf.data(),
        buf.size()
    );
    
    if (result < 0) {
        throw std::runtime_error("Failed to apply chat template");
    }
    
    if (result >= (int)buf.size()) {
        // Buffer was too small, resize and try again
        buf.resize(result + 1);
        result = llama_chat_apply_template(
            custom_template.empty() ? nullptr : custom_template.c_str(),
            c_messages.data(),
            c_messages.size(),
            true,  // add_ass
            buf.data(),
            buf.size()
        );
        
        if (result < 0 || result >= (int)buf.size()) {
            throw std::runtime_error("Failed to apply chat template after resizing buffer");
        }
    }
    
    return std::string(buf.data(), result);
}

PYBIND11_MODULE(llama_cpp, m) {
    m.doc() = "Python bindings for llama.cpp";
    
    // Expose LlamaModel
    py::class_<PyLlamaModel>(m, "LlamaModel")
        .def(py::init<const std::string&>())
        .def_property_readonly("n_ctx_train", &PyLlamaModel::n_ctx_train)
        .def_property_readonly("n_embd", &PyLlamaModel::n_embd)
        .def_property_readonly("n_layer", &PyLlamaModel::n_layer)
        .def_property_readonly("n_head", &PyLlamaModel::n_head)
        .def_property_readonly("n_head_kv", &PyLlamaModel::n_head_kv)
        .def_property_readonly("vocab", &PyLlamaModel::get_vocab)
        .def_property_readonly("vocab_size", [](PyLlamaModel& model) {
            return model.get_vocab().n_tokens();
        })
        .def("meta_val", &PyLlamaModel::meta_val)
        .def("chat_template", &PyLlamaModel::chat_template)
        .def("create_batch", [](PyLlamaModel& model, int n_tokens, int embd, int n_seq_max) {
            return PyLlamaBatch(n_tokens, embd, n_seq_max);
        }, py::arg("n_tokens"), py::arg("embd") = 0, py::arg("n_seq_max") = 1)
        .def("create_context", [](PyLlamaModel& model, py::kwargs kwargs) {
            py::dict params_dict;
            for (auto item : kwargs) {
                params_dict[item.first] = item.second;
            }
            return PyLlamaContext(model, params_dict);
        })
        .def("tokenize", [](PyLlamaModel& model, const std::string& text, bool add_bos, bool special) {
            return model.get_vocab().tokenize(text, add_bos, special);
        }, py::arg("text"), py::arg("add_bos") = false, py::arg("special") = true)
        .def("detokenize", [](PyLlamaModel& model, const std::vector<llama_token>& tokens) {
            return model.get_vocab().detokenize(tokens);
        });
    
    // Expose LlamaContext
    py::class_<PyLlamaContext>(m, "LlamaContext")
        .def(py::init<PyLlamaModel&, const py::dict&>())
        .def("decode", &PyLlamaContext::decode)
        .def("get_logits", &PyLlamaContext::get_logits)
        .def("get_embeddings", &PyLlamaContext::get_embeddings)
        .def("kv_cache_clear", &PyLlamaContext::kv_cache_clear)
        .def("set_n_threads", &PyLlamaContext::set_n_threads);
    
    // Expose LlamaBatch
    py::class_<PyLlamaBatch>(m, "LlamaBatch")
        .def(py::init<int32_t, int32_t, int32_t>())
        .def_property("tokens", &PyLlamaBatch::get_tokens, &PyLlamaBatch::set_tokens)
        .def_property("positions", &PyLlamaBatch::get_positions, &PyLlamaBatch::set_positions)
        .def_property("n_seq_id", &PyLlamaBatch::get_n_seq_id, &PyLlamaBatch::set_n_seq_id)
        .def_property("logits", &PyLlamaBatch::get_logits, &PyLlamaBatch::set_logits);
    
    // Expose LlamaSampler
    py::class_<PyLlamaSampler>(m, "LlamaSampler")
        .def(py::init<>())
        .def("add_top_k", &PyLlamaSampler::add_top_k)
        .def("add_top_p", &PyLlamaSampler::add_top_p)
        .def("add_temperature", &PyLlamaSampler::add_temperature)
        .def("add_mirostat", &PyLlamaSampler::add_mirostat)
        .def("add_grammar", &PyLlamaSampler::add_grammar)
        .def("sample", &PyLlamaSampler::sample)
        .def("accept", &PyLlamaSampler::accept)
        .def("get_context_ptr", &PyLlamaSampler::get_context_ptr);
    
    // Expose LlamaVocab
    py::class_<PyLlamaVocab>(m, "LlamaVocab")
        .def("tokenize", &PyLlamaVocab::tokenize, py::arg("text"), py::arg("add_bos") = false, py::arg("special") = true)
        .def("detokenize", &PyLlamaVocab::detokenize)
        .def("token_to_piece", &PyLlamaVocab::token_to_piece)
        .def_property_readonly("bos_token", &PyLlamaVocab::bos_token)
        .def_property_readonly("eos_token", &PyLlamaVocab::eos_token)
        .def_property_readonly("n_tokens", &PyLlamaVocab::n_tokens);
    
    // Expose helper functions
    m.def("chat_apply_template", &py_chat_apply_template, 
          py::arg("model"), py::arg("messages"), py::arg("custom_template") = "");
}
