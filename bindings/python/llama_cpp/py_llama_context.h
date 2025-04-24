#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "llama.h"
#include "py_llama_model.h"
#include "py_llama_batch.h"

namespace py = pybind11;

// LlamaContext wrapper class
class PyLlamaContext {
private:
    llama_context* ctx;
    PyLlamaModel model;

public:
    /**
     * @brief Construct a context for the given model with specified parameters
     * @param model The model to create a context for
     * @param params_dict Dictionary of context parameters
     */
    PyLlamaContext(PyLlamaModel& model, const py::dict& params_dict);
    
    /**
     * @brief Destructor - frees the context
     */
    ~PyLlamaContext();
    
    /**
     * @brief Get the raw context pointer
     * @return Pointer to the underlying llama_context
     */
    void* get_context_ptr() const;
    
    /**
     * @brief Process a batch of tokens through the model
     * @param batch The batch of tokens to process
     * @return True on success, false on failure
     */
    bool decode(const PyLlamaBatch& batch);
    
    /**
     * @brief Get the logits from the last decode operation
     * @return NumPy array of logits
     */
    py::array_t<float> get_logits();
    
    /**
     * @brief Get the embeddings from the last decode operation
     * @return NumPy array of embeddings
     */
    py::array_t<float> get_embeddings();
    void kv_cache_clear();
    void set_n_threads(int n_threads, int n_threads_batch);
};
