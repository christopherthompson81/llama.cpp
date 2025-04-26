#include "py_llama_quant.h"

PyLlamaQuant::PyLlamaQuant() {
    params = llama_model_quantize_default_params();
}

PyLlamaQuant::~PyLlamaQuant() {
    // Nothing for the destructor yet
}

void PyLlamaQuant::_zeros(std::ofstream & file, size_t n) {
    zeros(file, n);
}

void PyLlamaQuant::tensor_dequantize_impl(
    struct ggml_tensor * tensor,
    std::vector<no_init<float>> & output,
    std::vector<std::thread> & workers,
    const size_t nelements,
    const int nthread
) {
    llama_tensor_dequantize_impl(tensor, output, workers, nelements, nthread);
}

ggml_type PyLlamaQuant::tensor_get_type(
    quantize_state_impl & qs,
    ggml_type new_type,
    const ggml_tensor * tensor,
    llama_ftype ftype
) {
    return llama_tensor_get_type(qs, new_type, tensor, ftype);
}

size_t PyLlamaQuant::tensor_quantize_impl(
    enum ggml_type new_type,
    const float * f32_data,
    void * new_data,
    const int64_t chunk_size,
    int64_t nrows,
    int64_t n_per_row,
    const float * imatrix,
    std::vector<std::thread> & workers,
    const int nthread
) {
    return llama_tensor_quantize_impl(new_type, f32_data, new_data, chunk_size, nrows, n_per_row, imatrix, workers, nthread);
}

void PyLlamaQuant::model_quantize_impl(
    const std::string & fname_inp,
    const std::string & fname_out
) {
    llama_model_quantize_impl(fname_inp, fname_out, &params);
}

struct llama_model_quantize_params PyLlamaQuant::model_quantize_default_params() {
    return llama_model_quantize_default_params();
}

uint32_t PyLlamaQuant::model_quantize(
        const char * fname_inp,
        const char * fname_out
) {
    return llama_model_quantize(fname_inp, fname_out, &params);
}
