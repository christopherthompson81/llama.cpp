cmake_minimum_required(VERSION 3.12)
project(quantization_analysis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Qt5 COMPONENTS Core Widgets Charts REQUIRED)
find_package(SQLite3 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/common)

# Add the executable
add_executable(quantization_analysis
    quantization_analysis.cpp
)

# Link libraries
target_link_libraries(quantization_analysis
    PRIVATE
    llama
    common
    Qt5::Core
    Qt5::Widgets
    Qt5::Charts
    SQLite::SQLite3
)

# Enable automoc for Qt
set_target_properties(quantization_analysis PROPERTIES
    AUTOMOC ON
)
