#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "llama.h"

namespace py = pybind11;

// Forward declarations
class PyLlamaVocab;

// LlamaModel wrapper class
class PyLlamaModel {
private:
    llama_model* model;
    bool owns_model;

public:
    PyLlamaModel(const std::string& path);
    PyLlamaModel(llama_model* model, bool owns_model = false);
    ~PyLlamaModel();

    llama_model* get_model() const;
    int n_ctx_train() const;
    int n_embd() const;
    int n_layer() const;
    int n_head() const;
    int n_head_kv() const;
    PyLlamaVocab get_vocab() const;
    std::string meta_val(const std::string& key) const;
    std::string chat_template() const;
};
