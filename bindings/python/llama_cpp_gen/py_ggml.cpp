#include <pybind11/pybind11.h>

#include "ggml.h"

namespace py = pybind11;

PYBIND11_MODULE(ggml, m) {
    m.def("ggml_abort", &ggml_abort);

    py::enum_<ggml_status>(m, "ggml_status")
        .value("GGML_STATUS_ALLOC_FAILED", ggml_status::GGML_STATUS_ALLOC_FAILED)
        .value("GGML_STATUS_FAILED", ggml_status::GGML_STATUS_FAILED)
        .value("GGML_STATUS_SUCCESS", ggml_status::GGML_STATUS_SUCCESS)
        .value("GGML_STATUS_ABORTED", ggml_status::GGML_STATUS_ABORTED);

    m.def("ggml_status_to_string", &ggml_status_to_string);

    m.def("ggml_fp16_to_fp32", &ggml_fp16_to_fp32);

    m.def("ggml_fp32_to_fp16", &ggml_fp32_to_fp16);

    m.def("ggml_fp16_to_fp32_row", &ggml_fp16_to_fp32_row);

    m.def("ggml_fp32_to_fp16_row", &ggml_fp32_to_fp16_row);

    py::class_<ggml_bf16_t>(m, "ggml_bf16_t")
        .def(py::init<>())
        .def_readwrite("bits", &ggml_bf16_t::bits);

    m.def("ggml_fp32_to_bf16", &ggml_fp32_to_bf16);

    m.def("ggml_bf16_to_fp32", &ggml_bf16_to_fp32);

    m.def("ggml_bf16_to_fp32_row", &ggml_bf16_to_fp32_row);

    m.def("ggml_fp32_to_bf16_row_ref", &ggml_fp32_to_bf16_row_ref);

    m.def("ggml_fp32_to_bf16_row", &ggml_fp32_to_bf16_row);

    py::enum_<ggml_type>(m, "ggml_type")
        .value("GGML_TYPE_F32", ggml_type::GGML_TYPE_F32)
        .value("GGML_TYPE_F16", ggml_type::GGML_TYPE_F16)
        .value("GGML_TYPE_Q4_0", ggml_type::GGML_TYPE_Q4_0)
        .value("GGML_TYPE_Q4_1", ggml_type::GGML_TYPE_Q4_1)
        .value("GGML_TYPE_Q5_0", ggml_type::GGML_TYPE_Q5_0)
        .value("GGML_TYPE_Q5_1", ggml_type::GGML_TYPE_Q5_1)
        .value("GGML_TYPE_Q8_0", ggml_type::GGML_TYPE_Q8_0)
        .value("GGML_TYPE_Q8_1", ggml_type::GGML_TYPE_Q8_1)
        .value("GGML_TYPE_Q2_K", ggml_type::GGML_TYPE_Q2_K)
        .value("GGML_TYPE_Q3_K", ggml_type::GGML_TYPE_Q3_K)
        .value("GGML_TYPE_Q4_K", ggml_type::GGML_TYPE_Q4_K)
        .value("GGML_TYPE_Q5_K", ggml_type::GGML_TYPE_Q5_K)
        .value("GGML_TYPE_Q6_K", ggml_type::GGML_TYPE_Q6_K)
        .value("GGML_TYPE_Q8_K", ggml_type::GGML_TYPE_Q8_K)
        .value("GGML_TYPE_IQ2_XXS", ggml_type::GGML_TYPE_IQ2_XXS)
        .value("GGML_TYPE_IQ2_XS", ggml_type::GGML_TYPE_IQ2_XS)
        .value("GGML_TYPE_IQ3_XXS", ggml_type::GGML_TYPE_IQ3_XXS)
        .value("GGML_TYPE_IQ1_S", ggml_type::GGML_TYPE_IQ1_S)
        .value("GGML_TYPE_IQ4_NL", ggml_type::GGML_TYPE_IQ4_NL)
        .value("GGML_TYPE_IQ3_S", ggml_type::GGML_TYPE_IQ3_S)
        .value("GGML_TYPE_IQ2_S", ggml_type::GGML_TYPE_IQ2_S)
        .value("GGML_TYPE_IQ4_XS", ggml_type::GGML_TYPE_IQ4_XS)
        .value("GGML_TYPE_I8", ggml_type::GGML_TYPE_I8)
        .value("GGML_TYPE_I16", ggml_type::GGML_TYPE_I16)
        .value("GGML_TYPE_I32", ggml_type::GGML_TYPE_I32)
        .value("GGML_TYPE_I64", ggml_type::GGML_TYPE_I64)
        .value("GGML_TYPE_F64", ggml_type::GGML_TYPE_F64)
        .value("GGML_TYPE_IQ1_M", ggml_type::GGML_TYPE_IQ1_M)
        .value("GGML_TYPE_BF16", ggml_type::GGML_TYPE_BF16)
        .value("GGML_TYPE_TQ1_0", ggml_type::GGML_TYPE_TQ1_0)
        .value("GGML_TYPE_TQ2_0", ggml_type::GGML_TYPE_TQ2_0)
        .value("GGML_TYPE_COUNT", ggml_type::GGML_TYPE_COUNT);

    py::enum_<ggml_prec>(m, "ggml_prec")
        .value("GGML_PREC_DEFAULT", ggml_prec::GGML_PREC_DEFAULT)
        .value("GGML_PREC_F32", ggml_prec::GGML_PREC_F32);

    py::enum_<ggml_ftype>(m, "ggml_ftype")
        .value("GGML_FTYPE_UNKNOWN", ggml_ftype::GGML_FTYPE_UNKNOWN)
        .value("GGML_FTYPE_ALL_F32", ggml_ftype::GGML_FTYPE_ALL_F32)
        .value("GGML_FTYPE_MOSTLY_F16", ggml_ftype::GGML_FTYPE_MOSTLY_F16)
        .value("GGML_FTYPE_MOSTLY_Q4_0", ggml_ftype::GGML_FTYPE_MOSTLY_Q4_0)
        .value("GGML_FTYPE_MOSTLY_Q4_1", ggml_ftype::GGML_FTYPE_MOSTLY_Q4_1)
        .value("GGML_FTYPE_MOSTLY_Q4_1_SOME_F16", ggml_ftype::GGML_FTYPE_MOSTLY_Q4_1_SOME_F16)
        .value("GGML_FTYPE_MOSTLY_Q8_0", ggml_ftype::GGML_FTYPE_MOSTLY_Q8_0)
        .value("GGML_FTYPE_MOSTLY_Q5_0", ggml_ftype::GGML_FTYPE_MOSTLY_Q5_0)
        .value("GGML_FTYPE_MOSTLY_Q5_1", ggml_ftype::GGML_FTYPE_MOSTLY_Q5_1)
        .value("GGML_FTYPE_MOSTLY_Q2_K", ggml_ftype::GGML_FTYPE_MOSTLY_Q2_K)
        .value("GGML_FTYPE_MOSTLY_Q3_K", ggml_ftype::GGML_FTYPE_MOSTLY_Q3_K)
        .value("GGML_FTYPE_MOSTLY_Q4_K", ggml_ftype::GGML_FTYPE_MOSTLY_Q4_K)
        .value("GGML_FTYPE_MOSTLY_Q5_K", ggml_ftype::GGML_FTYPE_MOSTLY_Q5_K)
        .value("GGML_FTYPE_MOSTLY_Q6_K", ggml_ftype::GGML_FTYPE_MOSTLY_Q6_K)
        .value("GGML_FTYPE_MOSTLY_IQ2_XXS", ggml_ftype::GGML_FTYPE_MOSTLY_IQ2_XXS)
        .value("GGML_FTYPE_MOSTLY_IQ2_XS", ggml_ftype::GGML_FTYPE_MOSTLY_IQ2_XS)
        .value("GGML_FTYPE_MOSTLY_IQ3_XXS", ggml_ftype::GGML_FTYPE_MOSTLY_IQ3_XXS)
        .value("GGML_FTYPE_MOSTLY_IQ1_S", ggml_ftype::GGML_FTYPE_MOSTLY_IQ1_S)
        .value("GGML_FTYPE_MOSTLY_IQ4_NL", ggml_ftype::GGML_FTYPE_MOSTLY_IQ4_NL)
        .value("GGML_FTYPE_MOSTLY_IQ3_S", ggml_ftype::GGML_FTYPE_MOSTLY_IQ3_S)
        .value("GGML_FTYPE_MOSTLY_IQ2_S", ggml_ftype::GGML_FTYPE_MOSTLY_IQ2_S)
        .value("GGML_FTYPE_MOSTLY_IQ4_XS", ggml_ftype::GGML_FTYPE_MOSTLY_IQ4_XS)
        .value("GGML_FTYPE_MOSTLY_IQ1_M", ggml_ftype::GGML_FTYPE_MOSTLY_IQ1_M)
        .value("GGML_FTYPE_MOSTLY_BF16", ggml_ftype::GGML_FTYPE_MOSTLY_BF16);

    py::enum_<ggml_op>(m, "ggml_op")
        .value("GGML_OP_NONE", ggml_op::GGML_OP_NONE)
        .value("GGML_OP_DUP", ggml_op::GGML_OP_DUP)
        .value("GGML_OP_ADD", ggml_op::GGML_OP_ADD)
        .value("GGML_OP_ADD1", ggml_op::GGML_OP_ADD1)
        .value("GGML_OP_ACC", ggml_op::GGML_OP_ACC)
        .value("GGML_OP_SUB", ggml_op::GGML_OP_SUB)
        .value("GGML_OP_MUL", ggml_op::GGML_OP_MUL)
        .value("GGML_OP_DIV", ggml_op::GGML_OP_DIV)
        .value("GGML_OP_SQR", ggml_op::GGML_OP_SQR)
        .value("GGML_OP_SQRT", ggml_op::GGML_OP_SQRT)
        .value("GGML_OP_LOG", ggml_op::GGML_OP_LOG)
        .value("GGML_OP_SIN", ggml_op::GGML_OP_SIN)
        .value("GGML_OP_COS", ggml_op::GGML_OP_COS)
        .value("GGML_OP_SUM", ggml_op::GGML_OP_SUM)
        .value("GGML_OP_SUM_ROWS", ggml_op::GGML_OP_SUM_ROWS)
        .value("GGML_OP_MEAN", ggml_op::GGML_OP_MEAN)
        .value("GGML_OP_ARGMAX", ggml_op::GGML_OP_ARGMAX)
        .value("GGML_OP_COUNT_EQUAL", ggml_op::GGML_OP_COUNT_EQUAL)
        .value("GGML_OP_REPEAT", ggml_op::GGML_OP_REPEAT)
        .value("GGML_OP_REPEAT_BACK", ggml_op::GGML_OP_REPEAT_BACK)
        .value("GGML_OP_CONCAT", ggml_op::GGML_OP_CONCAT)
        .value("GGML_OP_SILU_BACK", ggml_op::GGML_OP_SILU_BACK)
        .value("GGML_OP_NORM", ggml_op::GGML_OP_NORM)
        .value("GGML_OP_RMS_NORM", ggml_op::GGML_OP_RMS_NORM)
        .value("GGML_OP_RMS_NORM_BACK", ggml_op::GGML_OP_RMS_NORM_BACK)
        .value("GGML_OP_GROUP_NORM", ggml_op::GGML_OP_GROUP_NORM)
        .value("GGML_OP_L2_NORM", ggml_op::GGML_OP_L2_NORM)
        .value("GGML_OP_MUL_MAT", ggml_op::GGML_OP_MUL_MAT)
        .value("GGML_OP_MUL_MAT_ID", ggml_op::GGML_OP_MUL_MAT_ID)
        .value("GGML_OP_OUT_PROD", ggml_op::GGML_OP_OUT_PROD)
        .value("GGML_OP_SCALE", ggml_op::GGML_OP_SCALE)
        .value("GGML_OP_SET", ggml_op::GGML_OP_SET)
        .value("GGML_OP_CPY", ggml_op::GGML_OP_CPY)
        .value("GGML_OP_CONT", ggml_op::GGML_OP_CONT)
        .value("GGML_OP_RESHAPE", ggml_op::GGML_OP_RESHAPE)
        .value("GGML_OP_VIEW", ggml_op::GGML_OP_VIEW)
        .value("GGML_OP_PERMUTE", ggml_op::GGML_OP_PERMUTE)
        .value("GGML_OP_TRANSPOSE", ggml_op::GGML_OP_TRANSPOSE)
        .value("GGML_OP_GET_ROWS", ggml_op::GGML_OP_GET_ROWS)
        .value("GGML_OP_GET_ROWS_BACK", ggml_op::GGML_OP_GET_ROWS_BACK)
        .value("GGML_OP_DIAG", ggml_op::GGML_OP_DIAG)
        .value("GGML_OP_DIAG_MASK_INF", ggml_op::GGML_OP_DIAG_MASK_INF)
        .value("GGML_OP_DIAG_MASK_ZERO", ggml_op::GGML_OP_DIAG_MASK_ZERO)
        .value("GGML_OP_SOFT_MAX", ggml_op::GGML_OP_SOFT_MAX)
        .value("GGML_OP_SOFT_MAX_BACK", ggml_op::GGML_OP_SOFT_MAX_BACK)
        .value("GGML_OP_ROPE", ggml_op::GGML_OP_ROPE)
        .value("GGML_OP_ROPE_BACK", ggml_op::GGML_OP_ROPE_BACK)
        .value("GGML_OP_CLAMP", ggml_op::GGML_OP_CLAMP)
        .value("GGML_OP_CONV_TRANSPOSE_1D", ggml_op::GGML_OP_CONV_TRANSPOSE_1D)
        .value("GGML_OP_IM2COL", ggml_op::GGML_OP_IM2COL)
        .value("GGML_OP_IM2COL_BACK", ggml_op::GGML_OP_IM2COL_BACK)
        .value("GGML_OP_CONV_TRANSPOSE_2D", ggml_op::GGML_OP_CONV_TRANSPOSE_2D)
        .value("GGML_OP_POOL_1D", ggml_op::GGML_OP_POOL_1D)
        .value("GGML_OP_POOL_2D", ggml_op::GGML_OP_POOL_2D)
        .value("GGML_OP_POOL_2D_BACK", ggml_op::GGML_OP_POOL_2D_BACK)
        .value("GGML_OP_UPSCALE", ggml_op::GGML_OP_UPSCALE)
        .value("GGML_OP_PAD", ggml_op::GGML_OP_PAD)
        .value("GGML_OP_PAD_REFLECT_1D", ggml_op::GGML_OP_PAD_REFLECT_1D)
        .value("GGML_OP_ARANGE", ggml_op::GGML_OP_ARANGE)
        .value("GGML_OP_TIMESTEP_EMBEDDING", ggml_op::GGML_OP_TIMESTEP_EMBEDDING)
        .value("GGML_OP_ARGSORT", ggml_op::GGML_OP_ARGSORT)
        .value("GGML_OP_LEAKY_RELU", ggml_op::GGML_OP_LEAKY_RELU)
        .value("GGML_OP_FLASH_ATTN_EXT", ggml_op::GGML_OP_FLASH_ATTN_EXT)
        .value("GGML_OP_FLASH_ATTN_BACK", ggml_op::GGML_OP_FLASH_ATTN_BACK)
        .value("GGML_OP_SSM_CONV", ggml_op::GGML_OP_SSM_CONV)
        .value("GGML_OP_SSM_SCAN", ggml_op::GGML_OP_SSM_SCAN)
        .value("GGML_OP_WIN_PART", ggml_op::GGML_OP_WIN_PART)
        .value("GGML_OP_WIN_UNPART", ggml_op::GGML_OP_WIN_UNPART)
        .value("GGML_OP_GET_REL_POS", ggml_op::GGML_OP_GET_REL_POS)
        .value("GGML_OP_ADD_REL_POS", ggml_op::GGML_OP_ADD_REL_POS)
        .value("GGML_OP_RWKV_WKV6", ggml_op::GGML_OP_RWKV_WKV6)
        .value("GGML_OP_GATED_LINEAR_ATTN", ggml_op::GGML_OP_GATED_LINEAR_ATTN)
        .value("GGML_OP_RWKV_WKV7", ggml_op::GGML_OP_RWKV_WKV7)
        .value("GGML_OP_UNARY", ggml_op::GGML_OP_UNARY)
        .value("GGML_OP_MAP_CUSTOM1", ggml_op::GGML_OP_MAP_CUSTOM1)
        .value("GGML_OP_MAP_CUSTOM2", ggml_op::GGML_OP_MAP_CUSTOM2)
        .value("GGML_OP_MAP_CUSTOM3", ggml_op::GGML_OP_MAP_CUSTOM3)
        .value("GGML_OP_CUSTOM", ggml_op::GGML_OP_CUSTOM)
        .value("GGML_OP_CROSS_ENTROPY_LOSS", ggml_op::GGML_OP_CROSS_ENTROPY_LOSS)
        .value("GGML_OP_CROSS_ENTROPY_LOSS_BACK", ggml_op::GGML_OP_CROSS_ENTROPY_LOSS_BACK)
        .value("GGML_OP_OPT_STEP_ADAMW", ggml_op::GGML_OP_OPT_STEP_ADAMW)
        .value("GGML_OP_COUNT", ggml_op::GGML_OP_COUNT);

    py::enum_<ggml_unary_op>(m, "ggml_unary_op")
        .value("GGML_UNARY_OP_ABS", ggml_unary_op::GGML_UNARY_OP_ABS)
        .value("GGML_UNARY_OP_SGN", ggml_unary_op::GGML_UNARY_OP_SGN)
        .value("GGML_UNARY_OP_NEG", ggml_unary_op::GGML_UNARY_OP_NEG)
        .value("GGML_UNARY_OP_STEP", ggml_unary_op::GGML_UNARY_OP_STEP)
        .value("GGML_UNARY_OP_TANH", ggml_unary_op::GGML_UNARY_OP_TANH)
        .value("GGML_UNARY_OP_ELU", ggml_unary_op::GGML_UNARY_OP_ELU)
        .value("GGML_UNARY_OP_RELU", ggml_unary_op::GGML_UNARY_OP_RELU)
        .value("GGML_UNARY_OP_SIGMOID", ggml_unary_op::GGML_UNARY_OP_SIGMOID)
        .value("GGML_UNARY_OP_GELU", ggml_unary_op::GGML_UNARY_OP_GELU)
        .value("GGML_UNARY_OP_GELU_QUICK", ggml_unary_op::GGML_UNARY_OP_GELU_QUICK)
        .value("GGML_UNARY_OP_SILU", ggml_unary_op::GGML_UNARY_OP_SILU)
        .value("GGML_UNARY_OP_HARDSWISH", ggml_unary_op::GGML_UNARY_OP_HARDSWISH)
        .value("GGML_UNARY_OP_HARDSIGMOID", ggml_unary_op::GGML_UNARY_OP_HARDSIGMOID)
        .value("GGML_UNARY_OP_EXP", ggml_unary_op::GGML_UNARY_OP_EXP)
        .value("GGML_UNARY_OP_COUNT", ggml_unary_op::GGML_UNARY_OP_COUNT);

    py::enum_<ggml_object_type>(m, "ggml_object_type")
        .value("GGML_OBJECT_TYPE_TENSOR", ggml_object_type::GGML_OBJECT_TYPE_TENSOR)
        .value("GGML_OBJECT_TYPE_GRAPH", ggml_object_type::GGML_OBJECT_TYPE_GRAPH)
        .value("GGML_OBJECT_TYPE_WORK_BUFFER", ggml_object_type::GGML_OBJECT_TYPE_WORK_BUFFER);

    py::enum_<ggml_log_level>(m, "ggml_log_level")
        .value("GGML_LOG_LEVEL_NONE", ggml_log_level::GGML_LOG_LEVEL_NONE)
        .value("GGML_LOG_LEVEL_DEBUG", ggml_log_level::GGML_LOG_LEVEL_DEBUG)
        .value("GGML_LOG_LEVEL_INFO", ggml_log_level::GGML_LOG_LEVEL_INFO)
        .value("GGML_LOG_LEVEL_WARN", ggml_log_level::GGML_LOG_LEVEL_WARN)
        .value("GGML_LOG_LEVEL_ERROR", ggml_log_level::GGML_LOG_LEVEL_ERROR)
        .value("GGML_LOG_LEVEL_CONT", ggml_log_level::GGML_LOG_LEVEL_CONT);

    py::enum_<ggml_tensor_flag>(m, "ggml_tensor_flag")
        .value("GGML_TENSOR_FLAG_INPUT", ggml_tensor_flag::GGML_TENSOR_FLAG_INPUT)
        .value("GGML_TENSOR_FLAG_OUTPUT", ggml_tensor_flag::GGML_TENSOR_FLAG_OUTPUT)
        .value("GGML_TENSOR_FLAG_PARAM", ggml_tensor_flag::GGML_TENSOR_FLAG_PARAM)
        .value("GGML_TENSOR_FLAG_LOSS", ggml_tensor_flag::GGML_TENSOR_FLAG_LOSS);

    py::class_<ggml_init_params>(m, "ggml_init_params")
        .def(py::init<>())
        .def_readwrite("mem_size", &ggml_init_params::mem_size)
        .def_readwrite("mem_buffer", &ggml_init_params::mem_buffer)
        .def_readwrite("no_alloc", &ggml_init_params::no_alloc);

    py::class_<ggml_tensor>(m, "ggml_tensor")
        .def(py::init<>())
        .def_readwrite("type", &ggml_tensor::type)
        .def_readwrite("buffer", &ggml_tensor::buffer)
        .def_readwrite("ne", &ggml_tensor::ne)
        .def_readwrite("nb", &ggml_tensor::nb)
        .def_readwrite("op", &ggml_tensor::op)
        .def_readwrite("op_params", &ggml_tensor::op_params)
        .def_readwrite("flags", &ggml_tensor::flags)
        .def_readwrite("src", &ggml_tensor::src)
        .def_readwrite("view_src", &ggml_tensor::view_src)
        .def_readwrite("view_offs", &ggml_tensor::view_offs)
        .def_readwrite("data", &ggml_tensor::data)
        .def_readwrite("name", &ggml_tensor::name)
        .def_readwrite("extra", &ggml_tensor::extra)
        .def_readwrite("padding", &ggml_tensor::padding);

    py::bind_constant(m, "GGML_TENSOR_SIZE", &GGML_TENSOR_SIZE);

    m.def("ggml_guid_matches", &ggml_guid_matches);

    m.def("ggml_time_init", &ggml_time_init);

    m.def("ggml_time_ms", &ggml_time_ms);

    m.def("ggml_time_us", &ggml_time_us);

    m.def("ggml_cycles", &ggml_cycles);

    m.def("ggml_cycles_per_ms", &ggml_cycles_per_ms);

    m.def("ggml_fopen", &ggml_fopen);

    m.def("ggml_print_object", &ggml_print_object);

    m.def("ggml_print_objects", &ggml_print_objects);

    m.def("ggml_nelements", &ggml_nelements);

    m.def("ggml_nrows", &ggml_nrows);

    m.def("ggml_nbytes", &ggml_nbytes);

    m.def("ggml_nbytes_pad", &ggml_nbytes_pad);

    m.def("ggml_blck_size", &ggml_blck_size);

    m.def("ggml_type_size", &ggml_type_size);

    m.def("ggml_row_size", &ggml_row_size);

    m.def("ggml_type_sizef", &ggml_type_sizef);

    m.def("ggml_type_name", &ggml_type_name);

    m.def("ggml_op_name", &ggml_op_name);

    m.def("ggml_op_symbol", &ggml_op_symbol);

    m.def("ggml_unary_op_name", &ggml_unary_op_name);

    m.def("ggml_op_desc", &ggml_op_desc);

    m.def("ggml_element_size", &ggml_element_size);

    m.def("ggml_is_quantized", &ggml_is_quantized);

    m.def("ggml_ftype_to_ggml_type", &ggml_ftype_to_ggml_type);

    m.def("ggml_is_transposed", &ggml_is_transposed);

    m.def("ggml_is_permuted", &ggml_is_permuted);

    m.def("ggml_is_empty", &ggml_is_empty);

    m.def("ggml_is_scalar", &ggml_is_scalar);

    m.def("ggml_is_vector", &ggml_is_vector);

    m.def("ggml_is_matrix", &ggml_is_matrix);

    m.def("ggml_is_3d", &ggml_is_3d);

    m.def("ggml_n_dims", &ggml_n_dims);

    m.def("ggml_is_contiguous", &ggml_is_contiguous);

    m.def("ggml_is_contiguous_0", &ggml_is_contiguous_0);

    m.def("ggml_is_contiguous_1", &ggml_is_contiguous_1);

    m.def("ggml_is_contiguous_2", &ggml_is_contiguous_2);

    m.def("ggml_are_same_shape", &ggml_are_same_shape);

    m.def("ggml_are_same_stride", &ggml_are_same_stride);

    m.def("ggml_can_repeat", &ggml_can_repeat);

    m.def("ggml_tensor_overhead", &ggml_tensor_overhead);

    m.def("ggml_validate_row_data", &ggml_validate_row_data);

    m.def("ggml_init", &ggml_init);

    m.def("ggml_reset", &ggml_reset);

    m.def("ggml_free", &ggml_free);

    m.def("ggml_used_mem", &ggml_used_mem);

    m.def("ggml_get_no_alloc", &ggml_get_no_alloc);

    m.def("ggml_set_no_alloc", &ggml_set_no_alloc);

    m.def("ggml_get_mem_buffer", &ggml_get_mem_buffer);

    m.def("ggml_get_mem_size", &ggml_get_mem_size);

    m.def("ggml_get_max_tensor_size", &ggml_get_max_tensor_size);

    m.def("ggml_new_tensor", &ggml_new_tensor);

    m.def("ggml_new_tensor_1d", &ggml_new_tensor_1d);

    m.def("ggml_new_tensor_2d", &ggml_new_tensor_2d);

    m.def("ggml_new_tensor_3d", &ggml_new_tensor_3d);

    m.def("ggml_new_tensor_4d", &ggml_new_tensor_4d);

    m.def("ggml_new_buffer", &ggml_new_buffer);

    m.def("ggml_dup_tensor", &ggml_dup_tensor);

    m.def("ggml_view_tensor", &ggml_view_tensor);

    m.def("ggml_get_first_tensor", &ggml_get_first_tensor);

    m.def("ggml_get_next_tensor", &ggml_get_next_tensor);

    m.def("ggml_get_tensor", &ggml_get_tensor);

    m.def("ggml_unravel_index", &ggml_unravel_index);

    m.def("ggml_get_unary_op", &ggml_get_unary_op);

    m.def("ggml_get_data", &ggml_get_data);

    m.def("ggml_get_data_f32", &ggml_get_data_f32);

    m.def("ggml_get_name", &ggml_get_name);

    m.def("ggml_set_name", &ggml_set_name);

    m.def("ggml_format_name", &ggml_format_name);

    m.def("ggml_set_input", &ggml_set_input);

    m.def("ggml_set_output", &ggml_set_output);

    m.def("ggml_set_param", &ggml_set_param);

    m.def("ggml_set_loss", &ggml_set_loss);

    m.def("ggml_dup", &ggml_dup);

    m.def("ggml_dup_inplace", &ggml_dup_inplace);

    m.def("ggml_add", &ggml_add);

    m.def("ggml_add_inplace", &ggml_add_inplace);

    m.def("ggml_add_cast", &ggml_add_cast);

    m.def("ggml_add1", &ggml_add1);

    m.def("ggml_add1_inplace", &ggml_add1_inplace);

    m.def("ggml_acc", &ggml_acc);

    m.def("ggml_acc_inplace", &ggml_acc_inplace);

    m.def("ggml_sub", &ggml_sub);

    m.def("ggml_sub_inplace", &ggml_sub_inplace);

    m.def("ggml_mul", &ggml_mul);

    m.def("ggml_mul_inplace", &ggml_mul_inplace);

    m.def("ggml_div", &ggml_div);

    m.def("ggml_div_inplace", &ggml_div_inplace);

    m.def("ggml_sqr", &ggml_sqr);

    m.def("ggml_sqr_inplace", &ggml_sqr_inplace);

    m.def("ggml_sqrt", &ggml_sqrt);

    m.def("ggml_sqrt_inplace", &ggml_sqrt_inplace);

    m.def("ggml_log", &ggml_log);

    m.def("ggml_log_inplace", &ggml_log_inplace);

    m.def("ggml_sin", &ggml_sin);

    m.def("ggml_sin_inplace", &ggml_sin_inplace);

    m.def("ggml_cos", &ggml_cos);

    m.def("ggml_cos_inplace", &ggml_cos_inplace);

    m.def("ggml_sum", &ggml_sum);

    m.def("ggml_sum_rows", &ggml_sum_rows);

    m.def("ggml_mean", &ggml_mean);

    m.def("ggml_argmax", &ggml_argmax);

    m.def("ggml_count_equal", &ggml_count_equal);

    m.def("ggml_repeat", &ggml_repeat);

    m.def("ggml_repeat_back", &ggml_repeat_back);

    m.def("ggml_concat", &ggml_concat);

    m.def("ggml_abs", &ggml_abs);

    m.def("ggml_abs_inplace", &ggml_abs_inplace);

    m.def("ggml_sgn", &ggml_sgn);

    m.def("ggml_sgn_inplace", &ggml_sgn_inplace);

    m.def("ggml_neg", &ggml_neg);

    m.def("ggml_neg_inplace", &ggml_neg_inplace);

    m.def("ggml_step", &ggml_step);

    m.def("ggml_step_inplace", &ggml_step_inplace);

    m.def("ggml_tanh", &ggml_tanh);

    m.def("ggml_tanh_inplace", &ggml_tanh_inplace);

    m.def("ggml_elu", &ggml_elu);

    m.def("ggml_elu_inplace", &ggml_elu_inplace);

    m.def("ggml_relu", &ggml_relu);

    m.def("ggml_leaky_relu", &ggml_leaky_relu);

    m.def("ggml_relu_inplace", &ggml_relu_inplace);

    m.def("ggml_sigmoid", &ggml_sigmoid);

    m.def("ggml_sigmoid_inplace", &ggml_sigmoid_inplace);

    m.def("ggml_gelu", &ggml_gelu);

    m.def("ggml_gelu_inplace", &ggml_gelu_inplace);

    m.def("ggml_gelu_quick", &ggml_gelu_quick);

    m.def("ggml_gelu_quick_inplace", &ggml_gelu_quick_inplace);

    m.def("ggml_silu", &ggml_silu);

    m.def("ggml_silu_inplace", &ggml_silu_inplace);

    m.def("ggml_silu_back", &ggml_silu_back);

    m.def("ggml_hardswish", &ggml_hardswish);

    m.def("ggml_hardsigmoid", &ggml_hardsigmoid);

    m.def("ggml_exp", &ggml_exp);

    m.def("ggml_exp_inplace", &ggml_exp_inplace);

    m.def("ggml_norm", &ggml_norm);

    m.def("ggml_norm_inplace", &ggml_norm_inplace);

    m.def("ggml_rms_norm", &ggml_rms_norm);

    m.def("ggml_rms_norm_inplace", &ggml_rms_norm_inplace);

    m.def("ggml_group_norm", &ggml_group_norm);

    m.def("ggml_group_norm_inplace", &ggml_group_norm_inplace);

    m.def("ggml_l2_norm", &ggml_l2_norm);

    m.def("ggml_l2_norm_inplace", &ggml_l2_norm_inplace);

    m.def("ggml_rms_norm_back", &ggml_rms_norm_back);

    m.def("ggml_mul_mat", &ggml_mul_mat);

    m.def("ggml_mul_mat_set_prec", &ggml_mul_mat_set_prec);

    m.def("ggml_mul_mat_id", &ggml_mul_mat_id);

    m.def("ggml_out_prod", &ggml_out_prod);

    m.def("ggml_scale", &ggml_scale);

    m.def("ggml_scale_inplace", &ggml_scale_inplace);

    m.def("ggml_set", &ggml_set);

    m.def("ggml_set_inplace", &ggml_set_inplace);

    m.def("ggml_set_1d", &ggml_set_1d);

    m.def("ggml_set_1d_inplace", &ggml_set_1d_inplace);

    m.def("ggml_set_2d", &ggml_set_2d);

    m.def("ggml_set_2d_inplace", &ggml_set_2d_inplace);

    m.def("ggml_cpy", &ggml_cpy);

    m.def("ggml_cast", &ggml_cast);

    m.def("ggml_cont", &ggml_cont);

    m.def("ggml_cont_1d", &ggml_cont_1d);

    m.def("ggml_cont_2d", &ggml_cont_2d);

    m.def("ggml_cont_3d", &ggml_cont_3d);

    m.def("ggml_cont_4d", &ggml_cont_4d);

    m.def("ggml_reshape", &ggml_reshape);

    m.def("ggml_reshape_1d", &ggml_reshape_1d);

    m.def("ggml_reshape_2d", &ggml_reshape_2d);

    m.def("ggml_reshape_3d", &ggml_reshape_3d);

    m.def("ggml_reshape_4d", &ggml_reshape_4d);

    m.def("ggml_view_1d", &ggml_view_1d);

    m.def("ggml_view_2d", &ggml_view_2d);

    m.def("ggml_view_3d", &ggml_view_3d);

    m.def("ggml_view_4d", &ggml_view_4d);

    m.def("ggml_permute", &ggml_permute);

    m.def("ggml_transpose", &ggml_transpose);

    m.def("ggml_get_rows", &ggml_get_rows);

    m.def("ggml_get_rows_back", &ggml_get_rows_back);

    m.def("ggml_diag", &ggml_diag);

    m.def("ggml_diag_mask_inf", &ggml_diag_mask_inf);

    m.def("ggml_diag_mask_inf_inplace", &ggml_diag_mask_inf_inplace);

    m.def("ggml_diag_mask_zero", &ggml_diag_mask_zero);

    m.def("ggml_diag_mask_zero_inplace", &ggml_diag_mask_zero_inplace);

    m.def("ggml_soft_max", &ggml_soft_max);

    m.def("ggml_soft_max_inplace", &ggml_soft_max_inplace);

    m.def("ggml_soft_max_ext", &ggml_soft_max_ext);

    m.def("ggml_soft_max_ext_back", &ggml_soft_max_ext_back);

    m.def("ggml_soft_max_ext_back_inplace", &ggml_soft_max_ext_back_inplace);

    m.def("ggml_rope", &ggml_rope);

    m.def("ggml_rope_inplace", &ggml_rope_inplace);

    m.def("ggml_rope_ext", &ggml_rope_ext);

    m.def("ggml_rope_multi", &ggml_rope_multi);

    m.def("ggml_rope_ext_inplace", &ggml_rope_ext_inplace);

    m.def("ggml_rope_custom", &ggml_rope_custom);

    m.def("ggml_rope_custom_inplace", &ggml_rope_custom_inplace);

    m.def("ggml_rope_yarn_corr_dims", &ggml_rope_yarn_corr_dims);

    m.def("ggml_rope_ext_back", &ggml_rope_ext_back);

    m.def("ggml_rope_multi_back", &ggml_rope_multi_back);

    m.def("ggml_clamp", &ggml_clamp);

    m.def("ggml_im2col", &ggml_im2col);

    m.def("ggml_im2col_back", &ggml_im2col_back);

    m.def("ggml_conv_1d", &ggml_conv_1d);

    m.def("ggml_conv_1d_ph", &ggml_conv_1d_ph);

    m.def("ggml_conv_1d_dw", &ggml_conv_1d_dw);

    m.def("ggml_conv_1d_dw_ph", &ggml_conv_1d_dw_ph);

    m.def("ggml_conv_transpose_1d", &ggml_conv_transpose_1d);

    m.def("ggml_conv_2d", &ggml_conv_2d);

    m.def("ggml_conv_2d_sk_p0", &ggml_conv_2d_sk_p0);

    m.def("ggml_conv_2d_s1_ph", &ggml_conv_2d_s1_ph);

    m.def("ggml_conv_2d_dw", &ggml_conv_2d_dw);

    m.def("ggml_conv_transpose_2d_p0", &ggml_conv_transpose_2d_p0);

    py::enum_<ggml_op_pool>(m, "ggml_op_pool")
        .value("GGML_OP_POOL_MAX", ggml_op_pool::GGML_OP_POOL_MAX)
        .value("GGML_OP_POOL_AVG", ggml_op_pool::GGML_OP_POOL_AVG)
        .value("GGML_OP_POOL_COUNT", ggml_op_pool::GGML_OP_POOL_COUNT);

    m.def("ggml_pool_1d", &ggml_pool_1d);

    m.def("ggml_pool_2d", &ggml_pool_2d);

    m.def("ggml_pool_2d_back", &ggml_pool_2d_back);

    py::enum_<ggml_scale_mode>(m, "ggml_scale_mode")
        .value("GGML_SCALE_MODE_NEAREST", ggml_scale_mode::GGML_SCALE_MODE_NEAREST)
        .value("GGML_SCALE_MODE_BILINEAR", ggml_scale_mode::GGML_SCALE_MODE_BILINEAR);

    m.def("ggml_upscale", &ggml_upscale);

    m.def("ggml_upscale_ext", &ggml_upscale_ext);

    m.def("ggml_pad", &ggml_pad);

    m.def("ggml_pad_reflect_1d", &ggml_pad_reflect_1d);

    m.def("ggml_timestep_embedding", &ggml_timestep_embedding);

    py::enum_<ggml_sort_order>(m, "ggml_sort_order")
        .value("GGML_SORT_ORDER_ASC", ggml_sort_order::GGML_SORT_ORDER_ASC)
        .value("GGML_SORT_ORDER_DESC", ggml_sort_order::GGML_SORT_ORDER_DESC);

    m.def("ggml_argsort", &ggml_argsort);

    m.def("ggml_arange", &ggml_arange);

    m.def("ggml_top_k", &ggml_top_k);

    m.def("ggml_flash_attn_ext", &ggml_flash_attn_ext);

    m.def("ggml_flash_attn_ext_set_prec", &ggml_flash_attn_ext_set_prec);

    m.def("ggml_flash_attn_ext_get_prec", &ggml_flash_attn_ext_get_prec);

    m.def("ggml_flash_attn_back", &ggml_flash_attn_back);

    m.def("ggml_ssm_conv", &ggml_ssm_conv);

    m.def("ggml_ssm_scan", &ggml_ssm_scan);

    m.def("ggml_win_part", &ggml_win_part);

    m.def("ggml_win_unpart", &ggml_win_unpart);

    m.def("ggml_unary", &ggml_unary);

    m.def("ggml_unary_inplace", &ggml_unary_inplace);

    m.def("ggml_get_rel_pos", &ggml_get_rel_pos);

    m.def("ggml_add_rel_pos", &ggml_add_rel_pos);

    m.def("ggml_add_rel_pos_inplace", &ggml_add_rel_pos_inplace);

    m.def("ggml_rwkv_wkv6", &ggml_rwkv_wkv6);

    m.def("ggml_gated_linear_attn", &ggml_gated_linear_attn);

    m.def("ggml_rwkv_wkv7", &ggml_rwkv_wkv7);

    m.def("ggml_map_custom1", &ggml_map_custom1);

    m.def("ggml_map_custom1_inplace", &ggml_map_custom1_inplace);

    m.def("ggml_map_custom2", &ggml_map_custom2);

    m.def("ggml_map_custom2_inplace", &ggml_map_custom2_inplace);

    m.def("ggml_map_custom3", &ggml_map_custom3);

    m.def("ggml_map_custom3_inplace", &ggml_map_custom3_inplace);

    m.def("ggml_custom_4d", &ggml_custom_4d);

    m.def("ggml_custom_inplace", &ggml_custom_inplace);

    m.def("ggml_cross_entropy_loss", &ggml_cross_entropy_loss);

    m.def("ggml_cross_entropy_loss_back", &ggml_cross_entropy_loss_back);

    m.def("ggml_opt_step_adamw", &ggml_opt_step_adamw);

    m.def("ggml_build_forward_expand", &ggml_build_forward_expand);

    m.def("ggml_build_backward_expand", &ggml_build_backward_expand);

    m.def("ggml_new_graph", &ggml_new_graph);

    m.def("ggml_new_graph_custom", &ggml_new_graph_custom);

    m.def("ggml_graph_dup", &ggml_graph_dup);

    m.def("ggml_graph_cpy", &ggml_graph_cpy);

    m.def("ggml_graph_reset", &ggml_graph_reset);

    m.def("ggml_graph_clear", &ggml_graph_clear);

    m.def("ggml_graph_size", &ggml_graph_size);

    m.def("ggml_graph_node", &ggml_graph_node);

    m.def("ggml_graph_nodes", &ggml_graph_nodes);

    m.def("ggml_graph_n_nodes", &ggml_graph_n_nodes);

    m.def("ggml_graph_add_node", &ggml_graph_add_node);

    m.def("ggml_graph_overhead", &ggml_graph_overhead);

    m.def("ggml_graph_overhead_custom", &ggml_graph_overhead_custom);

    m.def("ggml_graph_get_tensor", &ggml_graph_get_tensor);

    m.def("ggml_graph_get_grad", &ggml_graph_get_grad);

    m.def("ggml_graph_get_grad_acc", &ggml_graph_get_grad_acc);

    m.def("ggml_graph_export", &ggml_graph_export);

    m.def("ggml_graph_import", &ggml_graph_import);

    m.def("ggml_graph_print", &ggml_graph_print);

    m.def("ggml_graph_dump_dot", &ggml_graph_dump_dot);

    m.def("ggml_log_set", &ggml_log_set);

    m.def("ggml_set_zero", &ggml_set_zero);

    m.def("ggml_quantize_init", &ggml_quantize_init);

    m.def("ggml_quantize_free", &ggml_quantize_free);

    m.def("ggml_quantize_requires_imatrix", &ggml_quantize_requires_imatrix);

    m.def("ggml_quantize_chunk", &ggml_quantize_chunk);

    py::class_<ggml_type_traits>(m, "ggml_type_traits")
        .def(py::init<>())
        .def_readwrite("type_name", &ggml_type_traits::type_name)
        .def_readwrite("blck_size", &ggml_type_traits::blck_size)
        .def_readwrite("blck_size_interleave", &ggml_type_traits::blck_size_interleave)
        .def_readwrite("type_size", &ggml_type_traits::type_size)
        .def_readwrite("is_quantized", &ggml_type_traits::is_quantized)
        .def_readwrite("to_float", &ggml_type_traits::to_float)
        .def_readwrite("from_float_ref", &ggml_type_traits::from_float_ref);

    m.def("ggml_get_type_traits", &ggml_get_type_traits);

    py::enum_<ggml_sched_priority>(m, "ggml_sched_priority")
        .value("GGML_SCHED_PRIO_NORMAL", ggml_sched_priority::GGML_SCHED_PRIO_NORMAL)
        .value("GGML_SCHED_PRIO_MEDIUM", ggml_sched_priority::GGML_SCHED_PRIO_MEDIUM)
        .value("GGML_SCHED_PRIO_HIGH", ggml_sched_priority::GGML_SCHED_PRIO_HIGH)
        .value("GGML_SCHED_PRIO_REALTIME", ggml_sched_priority::GGML_SCHED_PRIO_REALTIME);

    py::class_<ggml_threadpool_params>(m, "ggml_threadpool_params")
        .def(py::init<>())
        .def_readwrite("cpumask", &ggml_threadpool_params::cpumask)
        .def_readwrite("n_threads", &ggml_threadpool_params::n_threads)
        .def_readwrite("prio", &ggml_threadpool_params::prio)
        .def_readwrite("poll", &ggml_threadpool_params::poll)
        .def_readwrite("strict_cpu", &ggml_threadpool_params::strict_cpu)
        .def_readwrite("paused", &ggml_threadpool_params::paused);

    m.def("ggml_threadpool_params_default", &ggml_threadpool_params_default);

    m.def("ggml_threadpool_params_init", &ggml_threadpool_params_init);

    m.def("ggml_threadpool_params_match", &ggml_threadpool_params_match);
}