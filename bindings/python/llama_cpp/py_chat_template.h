#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llama.h"
#include "py_llama_model.h"

namespace py = pybind11;

// Helper function for chat templates
/**
 * @brief Apply a chat template to format messages for the model
 * @param model The model containing the default chat template
 * @param messages Vector of message maps, each with "role" and "content" keys
 * @param custom_template Optional custom template to use instead of the model's default
 * @return Formatted chat text ready for the model
 */
std::string py_chat_apply_template(
    const PyLlamaModel& model,
    const std::vector<std::map<std::string, std::string>>& messages,
    const std::string& custom_template = "");
