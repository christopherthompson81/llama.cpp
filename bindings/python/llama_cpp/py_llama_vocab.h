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
    /**
     * @brief Construct a vocabulary wrapper
     * @param model Pointer to the model containing the vocabulary
     * @param owns_model Whether this object should free the model on destruction
     */
    PyLlamaVocab(llama_model* model, bool owns_model = false);
    
    /**
     * @brief Destructor - frees the model if owns_model is true
     */
    ~PyLlamaVocab();

    /**
     * @brief Tokenize a text string into token IDs
     * @param text The input text to tokenize
     * @param add_bos Whether to prepend a BOS token
     * @param special Whether to handle special tokens
     * @return Vector of token IDs
     */
    std::vector<llama_token> tokenize(const std::string& text, bool add_bos = false, bool special = true) const;
    
    /**
     * @brief Convert token IDs back to text
     * @param tokens Vector of token IDs to convert
     * @return The reconstructed text
     */
    std::string detokenize(const std::vector<llama_token>& tokens) const;
    
    /**
     * @brief Convert a single token ID to its text representation
     * @param token The token ID to convert
     * @return The text piece for this token
     */
    std::string token_to_piece(llama_token token) const;
    
    /**
     * @brief Get the beginning-of-sequence token ID
     * @return The BOS token ID
     */
    llama_token bos_token() const;
    
    /**
     * @brief Get the end-of-sequence token ID
     * @return The EOS token ID
     */
    llama_token eos_token() const;
    int n_tokens() const;
};
