#include "py_chat_template.h"

std::string py_chat_apply_template(
    const PyLlamaModel& model,
    const std::vector<std::map<std::string, std::string>>& messages,
    const std::string& custom_template) {
    
    // Convert messages to C format
    std::vector<llama_chat_message> c_messages(messages.size());
    for (size_t i = 0; i < messages.size(); i++) {
        const auto& msg = messages[i];
        
        auto role_it = msg.find("role");
        auto content_it = msg.find("content");
        
        c_messages[i].role = role_it != msg.end() ? role_it->second.c_str() : "";
        c_messages[i].content = content_it != msg.end() ? content_it->second.c_str() : "";
    }
    
    // Allocate a buffer for the result
    std::vector<char> buf(65536, 0); // Start with 64KB buffer
    
    // Apply template
    int result = llama_chat_apply_template(
        custom_template.empty() ? nullptr : custom_template.c_str(),
        c_messages.data(),
        c_messages.size(),
        true,  // add_ass
        buf.data(),
        buf.size()
    );
    
    if (result < 0) {
        throw std::runtime_error("Failed to apply chat template");
    }
    
    if (result >= (int)buf.size()) {
        // Buffer was too small, resize and try again
        buf.resize(result + 1);
        result = llama_chat_apply_template(
            custom_template.empty() ? nullptr : custom_template.c_str(),
            c_messages.data(),
            c_messages.size(),
            true,  // add_ass
            buf.data(),
            buf.size()
        );
        
        if (result < 0 || result >= (int)buf.size()) {
            throw std::runtime_error("Failed to apply chat template after resizing buffer");
        }
    }
    
    return std::string(buf.data(), result);
}
