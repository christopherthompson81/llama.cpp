# Find Python interpreter and development components
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

# Create the module using pybind11's functions
add_library(llama_cpp MODULE llama_cpp/llama_cpp.cpp)
target_link_libraries(llama_cpp PRIVATE pybind11::module llama)
target_include_directories(llama_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Set Python-specific properties
set_target_properties(llama_cpp PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Set the output directory to be within the Python package
set_target_properties(llama_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp
)
