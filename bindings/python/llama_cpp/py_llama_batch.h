#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "llama.h"

namespace py = pybind11;

// LlamaBatch wrapper class
class PyLlamaBatch {
private:
    llama_batch batch;
    bool owns_batch;

public:
    PyLlamaBatch(int32_t n_tokens, int32_t embd = 0, int32_t n_seq_max = 1);
    PyLlamaBatch(llama_batch batch, bool owns_batch = false);
    ~PyLlamaBatch();

    llama_batch get_batch() const;
    py::array_t<llama_token> get_tokens() const;
    void set_tokens(py::array_t<llama_token> tokens);
    py::array_t<llama_pos> get_positions() const;
    void set_positions(py::array_t<llama_pos> positions);
    py::array_t<int32_t> get_n_seq_id() const;
    void set_n_seq_id(py::array_t<int32_t> n_seq_id);
    py::array_t<int8_t> get_logits() const;
    void set_logits(py::array_t<int8_t> logits);
};
