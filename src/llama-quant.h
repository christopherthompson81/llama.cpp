#pragma once

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>

static void zeros(std::ofstream & file, size_t n);

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

static void llama_tensor_dequantize_impl(
    struct ggml_tensor * tensor,
    std::vector<no_init<float>> & output,
    std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
);

static ggml_type llama_tensor_get_type(
    quantize_state_impl & qs,
    ggml_type new_type,
    const ggml_tensor * tensor,
    llama_ftype ftype
);

static size_t llama_tensor_quantize_impl(
    enum ggml_type new_type,
    const float * f32_data,
    void * new_data,
    const int64_t chunk_size,
    int64_t nrows,
    int64_t n_per_row,
    const float * imatrix,
    std::vector<std::thread> & workers,
    const int nthread
);

static void llama_model_quantize_impl(
    const std::string & fname_inp,
    const std::string & fname_out,
    const llama_model_quantize_params * params
);

struct llama_model_quantize_params llama_model_quantize_default_params();

uint32_t llama_model_quantize(
    const char * fname_inp,
    const char * fname_out,
    const llama_model_quantize_params * params
);