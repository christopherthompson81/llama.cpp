#include "py_llama_model.h"
#include "py_llama_vocab.h"

PyLlamaModel::PyLlamaModel(const std::string& path) {
    llama_model_params params = llama_model_default_params();
    model = llama_model_load_from_file(path.c_str(), params);
    if (!model) {
        throw std::runtime_error("Failed to load model from " + path);
    }
    owns_model = true;
}

PyLlamaModel::PyLlamaModel(llama_model* model, bool owns_model) : model(model), owns_model(owns_model) {}

PyLlamaModel::~PyLlamaModel() {
    if (owns_model && model) {
        llama_model_free(model);
    }
}

llama_model* PyLlamaModel::get_model() const {
    return model;
}

int PyLlamaModel::n_ctx_train() const {
    return llama_model_n_ctx_train(model);
}

int PyLlamaModel::n_embd() const {
    return llama_model_n_embd(model);
}

int PyLlamaModel::n_layer() const {
    return llama_model_n_layer(model);
}

int PyLlamaModel::n_head() const {
    return llama_model_n_head(model);
}

int PyLlamaModel::n_head_kv() const {
    return llama_model_n_head_kv(model);
}

PyLlamaVocab PyLlamaModel::get_vocab() const {
    return PyLlamaVocab(model, false);
}

std::string PyLlamaModel::meta_val(const std::string& key) const {
    char buf[256];
    int ret = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    return ret >= 0 ? std::string(buf) : "";
}

std::string PyLlamaModel::chat_template() const {
    const char* tmpl = llama_model_chat_template(model, nullptr);
    return tmpl ? std::string(tmpl) : "";
}
