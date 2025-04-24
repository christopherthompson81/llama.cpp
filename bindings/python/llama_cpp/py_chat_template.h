#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llama.h"
#include "py_llama_model.h"

namespace py = pybind11;

// Helper function for chat templates
std::string py_chat_apply_template(
    const PyLlamaModel& model,
    const std::vector<std::map<std::string, std::string>>& messages,
    const std::string& custom_template = "");
