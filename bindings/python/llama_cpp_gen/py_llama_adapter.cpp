#include <pybind11/pybind11.h>

#include "llama-adapter.h"

namespace py = pybind11;

PYBIND11_MODULE(llama_adapter, m) {
    py::class_<llama_adapter_cvec>(m, "llama_adapter_cvec")
        .def(py::init<>())
        .def_readwrite("tensor_for", &llama_adapter_cvec::tensor_for)
        .def_readwrite("apply_to", &llama_adapter_cvec::apply_to)
        .def_readwrite("apply", &llama_adapter_cvec::apply)
        .def_readwrite("private", &llama_adapter_cvec::private)
        .def_readwrite("layer_start", &llama_adapter_cvec::layer_start)
        .def_readwrite("layer_end", &llama_adapter_cvec::layer_end);

    py::class_<llama_adapter_lora_weight>(m, "llama_adapter_lora_weight")
        .def(py::init<>())
        .def_readwrite("a", &llama_adapter_lora_weight::a)
        .def_readwrite("b", &llama_adapter_lora_weight::b)
        .def_readwrite("get_scale", &llama_adapter_lora_weight::get_scale)
        .def_readwrite("llama_adapter_lora_weight", &llama_adapter_lora_weight::llama_adapter_lora_weight);

    py::class_<llama_adapter_lora>(m, "llama_adapter_lora")
        .def(py::init<>())
        .def_readwrite("alpha", &llama_adapter_lora::alpha)
        .def_readwrite("llama_adapter_lora", &llama_adapter_lora::llama_adapter_lora)
        .def_readwrite("get_weight", &llama_adapter_lora::get_weight);
}