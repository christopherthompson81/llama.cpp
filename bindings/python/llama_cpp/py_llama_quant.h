#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>

#include "llama.h"
#include "llama-impl.h"
#include "llama-quant.h"
#include "py_llama_context.h"

namespace py = pybind11;

// LlamaQuant wrapper class
class PyLlamaQuant {
private:
    llama_model_quantize_params params;

public:
    PyLlamaQuant();
    ~PyLlamaQuant();

    void _zeros(std::ofstream & file, size_t n);

    void PyLlamaQuant::tensor_dequantize_impl(
        struct ggml_tensor * tensor,
        std::vector<no_init<float>> & output,
        std::vector<std::thread> & workers,
        const size_t nelements,
        const int nthread
    );

    ggml_type PyLlamaQuant::tensor_get_type(
        quantize_state_impl & qs,
        ggml_type new_type,
        const ggml_tensor * tensor,
        llama_ftype ftype
    );
    
    /**
     * @brief Perform the quantization of f32_data to a GGML type
     * @param new_type The target quantization type
     * @return The quantized size of the data
     */
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
    );

    void PyLlamaQuant::model_quantize_impl(
        const std::string & fname_inp,
        const std::string & fname_out
    );

    struct llama_model_quantize_params PyLlamaQuant::model_quantize_default_params();

    uint32_t PyLlamaQuant::model_quantize(
            const char * fname_inp,
            const char * fname_out
    );
};
