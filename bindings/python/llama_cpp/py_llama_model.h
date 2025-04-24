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
    /**
     * @brief Load a model from a file
     * @param path Path to the model file
     */
    PyLlamaModel(const std::string& path);
    
    /**
     * @brief Wrap an existing model
     * @param model Pointer to the model to wrap
     * @param owns_model Whether this object should free the model on destruction
     */
    PyLlamaModel(llama_model* model, bool owns_model = false);
    
    /**
     * @brief Destructor - frees the model if owns_model is true
     */
    ~PyLlamaModel();

    /**
     * @brief Get the underlying model pointer
     * @return Pointer to the llama_model
     */
    llama_model* get_model() const;
    
    /**
     * @brief Get the context size the model was trained with
     * @return Training context size
     */
    int n_ctx_train() const;
    
    /**
     * @brief Get the embedding dimension of the model
     * @return Embedding dimension
     */
    int n_embd() const;
    int n_layer() const;
    int n_head() const;
    int n_head_kv() const;
    PyLlamaVocab get_vocab() const;
    /**
     * @brief Get a metadata value from the model
     * @param key The metadata key to look up
     * @return The metadata value as a string
     */
    std::string meta_val(const std::string& key) const;
    std::string chat_template() const;
};
