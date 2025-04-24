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
    int32_t n_seq_max; // Store the max sequences per token
    int32_t embd;      // Store the embedding dimension size

public:
    /**
     * @brief Construct a new batch with the specified capacity
     * @param n_tokens Maximum number of tokens this batch can hold
     * @param embd Embedding dimension (if non-zero, token embeddings will be used instead of token IDs)
     * @param n_seq_max Maximum number of sequences per token
     */
    PyLlamaBatch(int32_t n_tokens, int32_t embd = 0, int32_t n_seq_max = 1);
    
    /**
     * @brief Construct a wrapper around an existing llama_batch
     * @param batch The existing batch to wrap
     * @param owns_batch Whether this object should free the batch on destruction
     */
    PyLlamaBatch(llama_batch batch, bool owns_batch = false);
    
    /**
     * @brief Destructor - frees the batch if owns_batch is true
     */
    ~PyLlamaBatch();

    /**
     * @brief Get the underlying llama_batch structure
     * @return The llama_batch structure
     */
    llama_batch get_batch() const;
    
    /**
     * @brief Get the token IDs in this batch
     * @return NumPy array of token IDs
     */
    py::array_t<llama_token> get_tokens() const;
    
    /**
     * @brief Set the token IDs in this batch
     * @param tokens NumPy array of token IDs
     */
    void set_tokens(py::array_t<llama_token> tokens);
    
    /**
     * @brief Get the positions of tokens in this batch
     * @return NumPy array of positions
     */
    py::array_t<llama_pos> get_positions() const;
    
    /**
     * @brief Set the positions of tokens in this batch
     * @param positions NumPy array of positions
     */
    void set_positions(py::array_t<llama_pos> positions);
    
    /**
     * @brief Get the number of sequence IDs per token
     * @return NumPy array of sequence ID counts
     */
    py::array_t<int32_t> get_n_seq_id() const;
    
    /**
     * @brief Set the number of sequence IDs per token
     * @param n_seq_id NumPy array of sequence ID counts
     */
    void set_n_seq_id(py::array_t<int32_t> n_seq_id);
    
    /**
     * @brief Get the logits flags for each token
     * @return NumPy array of logits flags
     */
    py::array_t<int8_t> get_logits() const;
    
    /**
     * @brief Set the logits flags for each token
     * @param logits NumPy array of logits flags
     */
    void set_logits(py::array_t<int8_t> logits);
};
