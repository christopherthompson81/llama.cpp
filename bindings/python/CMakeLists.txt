# Find Python interpreter
find_package(Python COMPONENTS Interpreter REQUIRED)

# Get Python include directories directly from the interpreter
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get Python library directory
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE PYTHON_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get Python extension suffix
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# If empty, use default
if(NOT PYTHON_MODULE_EXTENSION)
    set(PYTHON_MODULE_EXTENSION ".so")
endif()

# Set empty prefix (Python modules don't use 'lib' prefix)
set(PYTHON_MODULE_PREFIX "")

message(STATUS "Python include directory: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Python library directory: ${PYTHON_LIBRARY_DIR}")
message(STATUS "Python module extension: ${PYTHON_MODULE_EXTENSION}")

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

# Create the module using standard CMake functions
add_library(llama_cpp MODULE llama_cpp/llama_cpp.cpp)
target_link_libraries(llama_cpp PRIVATE llama)
target_include_directories(llama_cpp PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${PYTHON_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIRS}
)

# Set Python-specific properties
set_target_properties(llama_cpp PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Set the output directory to be within the Python package
set_target_properties(llama_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp
)
