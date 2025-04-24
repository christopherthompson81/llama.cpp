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
    PyLlamaContext(PyLlamaModel& model, const py::dict& params_dict);
    ~PyLlamaContext();
    
    void* get_context_ptr() const;
    bool decode(const PyLlamaBatch& batch);
    py::array_t<float> get_logits();
    py::array_t<float> get_embeddings();
    void kv_cache_clear();
    void set_n_threads(int n_threads, int n_threads_batch);
};
