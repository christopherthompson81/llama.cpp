#include "py_llama_quant.h"

PyLlamaQuant::PyLlamaQuant() {
    std::string name;
    llama_ftype ftype;
    std::string desc;
}

PyLlamaQuant::~PyLlamaQuant() {
    // Nothing for the destructor yet
}

void PyLlamaQuant::zeros(std::ofstream & file, size_t n) {
    zeros(file, n);
}

/*
struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;

    int n_k_quantized = 0;
    int n_fallback    = 0;

    bool has_imatrix = false;

    // used to figure out if a model shares tok_embd with the output weight
    bool has_output = false;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
        {}
};
*/

void PyLlamaQuant::llama_tensor_dequantize_impl(
    struct ggml_tensor * tensor,
    std::vector<no_init<float>> & output,
    std::vector<std::thread> & workers,
    const size_t nelements,
    const int nthread
) {
    llama_tensor_dequantize_impl(tensor, output, workers, nelements, nthread);
}

ggml_type PyLlamaQuant::llama_tensor_get_type(
    quantize_state_impl & qs,
    ggml_type new_type,
    const ggml_tensor * tensor,
    llama_ftype ftype
) {
    return llama_tensor_get_type(qs, new_type, tensor, ftype);
}

size_t PyLlamaQuant::llama_tensor_quantize_impl(
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

void PyLlamaQuant::llama_model_quantize_impl(
    const std::string & fname_inp,
    const std::string & fname_out,
    const llama_model_quantize_params * params
) {
    llama_model_quantize_impl(fname_inp, fname_out, params);
}

/*
struct llama_model_quantize_params llama_model_quantize_default_params() {
    struct llama_model_quantize_params result = {
        //.nthread                     = 0,
        //.ftype                       = LLAMA_FTYPE_MOSTLY_Q5_1,
        //.output_tensor_type          = GGML_TYPE_COUNT,
        //.token_embedding_type        = GGML_TYPE_COUNT,
        //.allow_requantize            = false,
        //.quantize_output_tensor      = true,
        //.only_copy                   = false,
        //.pure                        = false,
        //.keep_split                  = false,
        //.imatrix                     = nullptr,
        //.kv_overrides                = nullptr,
    };

    return result;
}
*/

uint32_t PyLlamaQuant::llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params
) {
    return llama_model_quantize(fname_inp, fname_out, params);
}
