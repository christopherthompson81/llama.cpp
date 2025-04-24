#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llama.h"

namespace py = pybind11;

// LlamaVocab wrapper class
class PyLlamaVocab {
private:
    const llama_model* model;
    const llama_vocab* vocab;
    bool owns_model;

public:
    PyLlamaVocab(llama_model* model, bool owns_model = false);
    ~PyLlamaVocab();

    std::vector<llama_token> tokenize(const std::string& text, bool add_bos = false, bool special = true) const;
    std::string detokenize(const std::vector<llama_token>& tokens) const;
    std::string token_to_piece(llama_token token) const;
    llama_token bos_token() const;
    llama_token eos_token() const;
    int n_tokens() const;
};
