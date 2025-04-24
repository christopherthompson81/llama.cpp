#include "py_llama_vocab.h"

PyLlamaVocab::PyLlamaVocab(llama_model* model, bool owns_model) : model(model), owns_model(owns_model) {
    vocab = llama_model_get_vocab(model);
}

PyLlamaVocab::~PyLlamaVocab() {
    if (owns_model && model) {
        llama_model_free(const_cast<llama_model*>(model));
    }
}

std::vector<llama_token> PyLlamaVocab::tokenize(const std::string& text, bool add_bos, bool special) const {
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

std::string PyLlamaVocab::detokenize(const std::vector<llama_token>& tokens) const {
    std::string result;
    for (const auto& token : tokens) {
        result += token_to_piece(token);
    }
    return result;
}

std::string PyLlamaVocab::token_to_piece(llama_token token) const {
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

llama_token PyLlamaVocab::bos_token() const {
    return llama_vocab_bos(vocab);
}

llama_token PyLlamaVocab::eos_token() const {
    return llama_vocab_eos(vocab);
}

int PyLlamaVocab::n_tokens() const {
    return llama_vocab_n_tokens(vocab);
}
