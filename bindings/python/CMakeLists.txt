find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11 package
find_package(pybind11 CONFIG QUIET)

# If not found, fetch it
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching it...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(llama_cpp llama_cpp/llama_cpp.cpp)
target_link_libraries(llama_cpp PRIVATE llama)
target_include_directories(llama_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Set the output directory to be within the Python package
set_target_properties(llama_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp
)
