#include "py_llama_batch.h"

PyLlamaBatch::PyLlamaBatch(int32_t n_tokens, int32_t embd, int32_t n_seq_max) : n_seq_max(n_seq_max), embd(embd) { // Initialize member variables
    batch = llama_batch_init(n_tokens, embd, n_seq_max);
    owns_batch = true;
}

// Constructor for wrapping an existing batch - needs to handle n_seq_max somehow if resizing is ever needed from this state.
// For now, let's assume it's not resized or n_seq_max isn't critical for wrapped batches.
// If resizing becomes an issue for wrapped batches, this constructor might need adjustment.
PyLlamaBatch::PyLlamaBatch(llama_batch batch, bool owns_batch) : batch(batch), owns_batch(owns_batch), n_seq_max(1), embd(0) {
    // WARNING: Assuming n_seq_max = 1 and embd = 0 for wrapped batches. This might be incorrect
    // if the wrapped batch supports more/embeddings and resizing is attempted later.
    // A better approach might be to require these values when wrapping.
}

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
        // Use the embd size and n_seq_max stored in this PyLlamaBatch instance
        llama_batch new_batch = llama_batch_init(r.shape(0), this->embd, this->n_seq_max);

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
                        // Copy the actual number of sequence IDs for this token
                        std::memcpy(new_batch.seq_id[i], batch.seq_id[i],
                                    batch.n_seq_id[i] * sizeof(llama_seq_id));
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

py::array_t<llama_seq_id> PyLlamaBatch::get_sequence_ids() const {
    if (!batch.seq_id || !batch.seq_id[0] || batch.n_tokens == 0) {
        // Return an empty array if seq_id is null, its first row is null, or batch is empty
        return py::array_t<llama_seq_id>();
    }
    // Assumption: seq_id points to a contiguous block of memory for all seq_ids,
    // starting from batch.seq_id[0]. This is how llama_batch_init allocates it.
    return py::array_t<llama_seq_id>(
        {static_cast<py::ssize_t>(batch.n_tokens), static_cast<py::ssize_t>(this->n_seq_max)}, // Shape (n_tokens, n_seq_max)
        {sizeof(llama_seq_id) * this->n_seq_max, sizeof(llama_seq_id)}, // Strides
        batch.seq_id[0], // Pointer to the start of the first row's data
        py::capsule(this, [](void*) {}) // Use 'this' or another non-null pointer for base to indicate ownership/validity tied to PyLlamaBatch
    );
}

void PyLlamaBatch::set_sequence_ids(py::array_t<llama_seq_id> seq_ids) {
    if (!batch.seq_id) {
         throw std::runtime_error("Batch seq_id buffer is null. Cannot set sequence IDs.");
    }
    // Ensure the input is C-contiguous and has the correct type
    auto seq_ids_cont = seq_ids.cast<py::array_t<llama_seq_id, py::array::c_style | py::array::forcecast>>();

    if (seq_ids_cont.ndim() != 2) {
        throw std::runtime_error("Sequence IDs array must be 2-dimensional.");
    }
    // Check dimensions against the *current* batch.n_tokens and stored n_seq_max
    // The number of rows in the input must match the current batch.n_tokens
    if (seq_ids_cont.shape(0) != batch.n_tokens) {
         throw std::runtime_error("Sequence IDs array rows (" + std::to_string(seq_ids_cont.shape(0)) +
                                  ") must match current batch.n_tokens (" + std::to_string(batch.n_tokens) + ").");
    }
    if (seq_ids_cont.shape(1) > this->n_seq_max) {
         throw std::runtime_error("Sequence IDs array columns (" + std::to_string(seq_ids_cont.shape(1)) +
                                  ") exceed batch n_seq_max capacity (" + std::to_string(this->n_seq_max) + ").");
    }

    auto r = seq_ids_cont.unchecked<2>();
    for (py::ssize_t i = 0; i < r.shape(0); ++i) { // Iterate up to batch.n_tokens (which == r.shape(0))
        if (batch.seq_id[i] != nullptr) {
            // Copy the row data provided in the input array
            std::memcpy(batch.seq_id[i], r.data(i, 0), r.shape(1) * sizeof(llama_seq_id));
            // If input has fewer columns than n_seq_max, the remaining seq_ids in the batch buffer for this token are untouched.
        } else {
             // This shouldn't happen if llama_batch_init worked correctly
             throw std::runtime_error("Batch seq_id row pointer is null for token index " + std::to_string(i));
        }
    }
    // This setter copies data based on the input array's shape.
    // It's crucial that batch.n_tokens was set correctly *before* calling this setter.
}
