#include <pybind11/pybind11.h>

#include "llama-adapter.h"

namespace py = pybind11;

PYBIND11_MODULE(llama_adapter, m) {
    m.doc() = "Python bindings for llama adapter classes.";

    py::class_<llama_adapter_cvec>(m, "llama_adapter_cvec")
        .def(py::init<>())
        .def("tensor_for", &llama_adapter_cvec::tensor_for, py::return_value_policy::reference_internal)
        .def("apply_to", &llama_adapter_cvec::apply_to, py::return_value_policy::reference_internal)
        .def("apply", &llama_adapter_cvec::apply)
        .def_readonly("layer_start", &llama_adapter_cvec::layer_start)
        .def_readonly("layer_end", &llama_adapter_cvec::layer_end)
        .def_readonly("tensors", &llama_adapter_cvec::tensors);

    py::class_<llama_adapter_lora_weight>(m, "llama_adapter_lora_weight")
        .def(py::init<ggml_tensor*, ggml_tensor*>())
        .def_readwrite("a", &llama_adapter_lora_weight::a)
        .def_readwrite("b", &llama_adapter_lora_weight::b)
        .def("get_scale", &llama_adapter_lora_weight::get_scale);

    py::class_<llama_adapter_lora>(m, "llama_adapter_lora")
        .def(py::init<>())
        .def("get_weight", &llama_adapter_lora::get_weight, py::return_value_policy::reference_internal)
        .def_readwrite("ab_map", &llama_adapter_lora::ab_map)
        .def_readwrite("alpha", &llama_adapter_lora::alpha);

    // Define the llama_adapter_loras type (a map of LlamaAdapterLora* to float)
    py::class_<llama_adapter_loras>(m, "llama_adapter_loras")
        .def(py::init<>())
        .def(py::self += py::self)
        .def("items", [](const llama_adapter_loras& self) {
            return py::dict(self.begin(), self.end());
        })
        .def("__getitem__", [](const llama_adapter_loras& self, llama_adapter_lora* key) {
            return self.at(key);
        });
}