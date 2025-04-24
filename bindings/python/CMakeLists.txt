find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(llama_cpp llama_cpp/llama_cpp.cpp)
target_link_libraries(llama_cpp PRIVATE llama)
target_include_directories(llama_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Set the output directory to be within the Python package
set_target_properties(llama_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp
)
