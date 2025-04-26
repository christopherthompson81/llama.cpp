#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "llama.h"
#include "py_llama_model.h"
#include "py_llama_context.h"
#include "py_llama_batch.h"
#include "py_llama_sampler.h"
#include "py_llama_vocab.h"
#include "py_chat_template.h"

namespace py = pybind11;

PYBIND11_MODULE(llama_cpp, m) {
    m.doc() = "Python bindings for llama.cpp";
    
    // Expose LlamaModel
    py::class_<PyLlamaModel>(m, "LlamaModel")
        .def(py::init<const std::string&>())
        .def_property_readonly("n_ctx_train", &PyLlamaModel::n_ctx_train)
        .def_property_readonly("n_embd", &PyLlamaModel::n_embd)
        .def_property_readonly("n_layer", &PyLlamaModel::n_layer)
        .def_property_readonly("n_head", &PyLlamaModel::n_head)
        .def_property_readonly("n_head_kv", &PyLlamaModel::n_head_kv)
        .def_property_readonly("vocab", &PyLlamaModel::get_vocab)
        .def_property_readonly("vocab_size", [](PyLlamaModel& model) {
            return model.get_vocab().n_tokens();
        })
        .def("meta_val", &PyLlamaModel::meta_val)
        .def("chat_template", &PyLlamaModel::chat_template)
        .def("create_batch", [](PyLlamaModel& model, int n_tokens, int embd, int n_seq_max) {
            return PyLlamaBatch(n_tokens, embd, n_seq_max);
        }, py::arg("n_tokens"), py::arg("embd") = 0, py::arg("n_seq_max") = 1)
        .def("create_context", [](PyLlamaModel& model, py::kwargs kwargs) {
            py::dict params_dict;
            for (auto item : kwargs) {
                params_dict[item.first] = item.second;
            }
            return PyLlamaContext(model, params_dict);
        })
        .def("tokenize", [](PyLlamaModel& model, const std::string& text, bool add_bos, bool special) {
            return model.get_vocab().tokenize(text, add_bos, special);
        }, py::arg("text"), py::arg("add_bos") = false, py::arg("special") = true)
        .def("detokenize", [](PyLlamaModel& model, const std::vector<llama_token>& tokens) {
            return model.get_vocab().detokenize(tokens);
        });
    
    // Expose LlamaContext
    py::class_<PyLlamaContext>(m, "LlamaContext")
        .def(py::init<PyLlamaModel&, const py::dict&>())
        .def("decode", &PyLlamaContext::decode)
        .def("get_logits", &PyLlamaContext::get_logits)
        .def("get_embeddings", &PyLlamaContext::get_embeddings)
        .def("kv_cache_clear", &PyLlamaContext::kv_cache_clear)
        .def("set_n_threads", &PyLlamaContext::set_n_threads)
        .def("get_context_ptr", &PyLlamaContext::get_context_ptr);
    
    // Expose LlamaBatch
    py::class_<PyLlamaBatch>(m, "LlamaBatch")
        .def(py::init<int32_t, int32_t, int32_t>())
        .def_property("tokens", &PyLlamaBatch::get_tokens, &PyLlamaBatch::set_tokens)
        .def_property("positions", &PyLlamaBatch::get_positions, &PyLlamaBatch::set_positions)
        .def_property("n_seq_id", &PyLlamaBatch::get_n_seq_id, &PyLlamaBatch::set_n_seq_id)
        // Expose the new seq_id property using the getter/setter
        .def_property("seq_id", &PyLlamaBatch::get_sequence_ids, &PyLlamaBatch::set_sequence_ids,
                      "Sequence IDs for each token (NumPy array, shape: [n_tokens, n_seq_max])")
        .def_property("logits", &PyLlamaBatch::get_logits, &PyLlamaBatch::set_logits);

    // Expose LlamaSampler
    py::class_<PyLlamaSampler>(m, "LlamaSampler")
        .def(py::init<>())
        .def("add_greedy", &PyLlamaSampler::add_greedy)
        .def("add_top_k", &PyLlamaSampler::add_top_k)
        .def("add_top_p", &PyLlamaSampler::add_top_p)
        .def("add_temperature", &PyLlamaSampler::add_temperature)
        .def("add_mirostat", &PyLlamaSampler::add_mirostat)
        .def("add_grammar", &PyLlamaSampler::add_grammar)
        .def("sample", &PyLlamaSampler::sample)
        .def("accept", &PyLlamaSampler::accept)
        .def("get_context_ptr", &PyLlamaSampler::get_context_ptr);
    
    // Expose LlamaVocab
    py::class_<PyLlamaVocab>(m, "LlamaVocab")
        .def("tokenize", &PyLlamaVocab::tokenize, py::arg("text"), py::arg("add_bos") = false, py::arg("special") = true)
        .def("detokenize", &PyLlamaVocab::detokenize)
        .def("token_to_piece", &PyLlamaVocab::token_to_piece)
        .def_property_readonly("bos_token", &PyLlamaVocab::bos_token)
        .def_property_readonly("eos_token", &PyLlamaVocab::eos_token)
        .def_property_readonly("n_tokens", &PyLlamaVocab::n_tokens);
    
    // Expose helper functions
    m.def("chat_apply_template", &py_chat_apply_template, 
          py::arg("model"), py::arg("messages"), py::arg("custom_template") = "");
}
