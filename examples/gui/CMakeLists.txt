set(TARGET quantization_analysis)
add_executable(${TARGET} quantization_analysis.cpp)
install(TARGETS ${TARGET} RUNTIME)

# Find required packages
#find_package(Qt6 COMPONENTS Core Widgets Charts REQUIRED)
#find_package(SQLite3 REQUIRED)

# Include directories
#include_directories(${CMAKE_SOURCE_DIR})
#include_directories(${CMAKE_SOURCE_DIR}/common)

# Add the executable
#add_executable(quantization_analysis
#    quantization_analysis.cpp
#)

#target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
# Link libraries
target_link_libraries(${TARGET}
    PRIVATE
    llama
    common
    ${CMAKE_THREAD_LIBS_INIT}
    Qt6::Core
    Qt6::Widgets
    Qt6::Charts
    SQLite::SQLite3
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable automoc for Qt
set_target_properties(quantization_analysis PROPERTIES
    AUTOMOC ON
)
