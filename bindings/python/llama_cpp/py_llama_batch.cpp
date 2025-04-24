#include "py_llama_batch.h"

PyLlamaBatch::PyLlamaBatch(int32_t n_tokens, int32_t embd, int32_t n_seq_max) {
    batch = llama_batch_init(n_tokens, embd, n_seq_max);
    owns_batch = true;
}

PyLlamaBatch::PyLlamaBatch(llama_batch batch, bool owns_batch) : batch(batch), owns_batch(owns_batch) {}

PyLlamaBatch::~PyLlamaBatch() {
    if (owns_batch) {
        llama_batch_free(batch);
    }
}

llama_batch PyLlamaBatch::get_batch() const {
    return batch;
}

py::array_t<llama_token> PyLlamaBatch::get_tokens() const {
    return py::array_t<llama_token>(
        {batch.n_tokens},
        {sizeof(llama_token)},
        batch.token,
        py::capsule(nullptr, [](void*) {})
    );
}

void PyLlamaBatch::set_tokens(py::array_t<llama_token> tokens) {
    auto r = tokens.unchecked<1>();
    
    // Check if the array is too large
    if (r.shape(0) > batch.n_tokens) {
        // Resize the batch to accommodate the tokens
        llama_batch new_batch = llama_batch_init(r.shape(0), batch.embd, batch.n_seq_max);
        
        // Copy existing data if needed
        if (batch.n_tokens > 0) {
            // Copy tokens that will fit
            size_t copy_size = std::min(static_cast<size_t>(batch.n_tokens), static_cast<size_t>(r.shape(0)));
            std::memcpy(new_batch.token, batch.token, copy_size * sizeof(llama_token));
            
            // Copy other arrays if they exist
            if (batch.pos && new_batch.pos) {
                std::memcpy(new_batch.pos, batch.pos, copy_size * sizeof(llama_pos));
            }
            if (batch.n_seq_id && new_batch.n_seq_id) {
                std::memcpy(new_batch.n_seq_id, batch.n_seq_id, copy_size * sizeof(int32_t));
            }
            if (batch.seq_id && new_batch.seq_id) {
                // This is a 2D array, need to copy each row
                for (int i = 0; i < batch.n_tokens; i++) {
                    if (i < copy_size) {
                        std::memcpy(new_batch.seq_id[i], batch.seq_id[i], 
                                    batch.n_seq_max * sizeof(llama_seq_id));
                    }
                }
            }
            if (batch.logits && new_batch.logits) {
                std::memcpy(new_batch.logits, batch.logits, copy_size * sizeof(int8_t));
            }
        }
        
        // Free the old batch
        if (owns_batch) {
            llama_batch_free(batch);
        }
        
        // Use the new batch
        batch = new_batch;
        owns_batch = true;
    }
    
    // Copy the tokens
    for (py::ssize_t i = 0; i < r.shape(0); i++) {
        batch.token[i] = r(i);
    }
    
    // Update n_tokens to match the actual number of tokens
    batch.n_tokens = r.shape(0);
}

py::array_t<llama_pos> PyLlamaBatch::get_positions() const {
    return py::array_t<llama_pos>(
        {batch.n_tokens},
        {sizeof(llama_pos)},
        batch.pos,
        py::capsule(nullptr, [](void*) {})
    );
}

void PyLlamaBatch::set_positions(py::array_t<llama_pos> positions) {
    if (positions.size() > batch.n_tokens) {
        throw std::runtime_error("Positions array too large for batch");
    }
    
    auto r = positions.unchecked<1>();
    for (py::ssize_t i = 0; i < r.shape(0); i++) {
        batch.pos[i] = r(i);
    }
}

py::array_t<int32_t> PyLlamaBatch::get_n_seq_id() const {
    return py::array_t<int32_t>(
        {batch.n_tokens},
        {sizeof(int32_t)},
        batch.n_seq_id,
        py::capsule(nullptr, [](void*) {})
    );
}

void PyLlamaBatch::set_n_seq_id(py::array_t<int32_t> n_seq_id) {
    if (n_seq_id.size() > batch.n_tokens) {
        throw std::runtime_error("n_seq_id array too large for batch");
    }
    
    auto r = n_seq_id.unchecked<1>();
    for (py::ssize_t i = 0; i < r.shape(0); i++) {
        batch.n_seq_id[i] = r(i);
    }
}

py::array_t<int8_t> PyLlamaBatch::get_logits() const {
    return py::array_t<int8_t>(
        {batch.n_tokens},
        {sizeof(int8_t)},
        batch.logits,
        py::capsule(nullptr, [](void*) {})
    );
}

void PyLlamaBatch::set_logits(py::array_t<int8_t> logits) {
    if (logits.size() > batch.n_tokens) {
        throw std::runtime_error("Logits array too large for batch");
    }
    
    auto r = logits.unchecked<1>();
    for (py::ssize_t i = 0; i < r.shape(0); i++) {
        batch.logits[i] = r(i);
    }
}
