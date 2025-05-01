# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(batched)
    add_subdirectory(embedding)
    add_subdirectory(eval-callback)

    add_subdirectory(gguf-hash)
    add_subdirectory(gguf-split)
    add_subdirectory(gguf)
    add_subdirectory(gritlm)
    add_subdirectory(imatrix)
    add_subdirectory(infill)
    add_subdirectory(llama-bench)
    add_subdirectory(lookahead)
    add_subdirectory(lookup)
    add_subdirectory(main)
    add_subdirectory(parallel)
    add_subdirectory(passkey)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    add_subdirectory(retrieval)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(server)
    endif()
    add_subdirectory(save-load-state)
    add_subdirectory(run)
    add_subdirectory(simple)
    add_subdirectory(simple-chat)
    add_subdirectory(speculative)
    add_subdirectory(speculative-simple)
    add_subdirectory(tokenize)
    add_subdirectory(tts)
    add_subdirectory(gen-docs)
    if (NOT GGML_BACKEND_DL)
        # these examples use the backends directly and cannot be built with dynamic loading
        add_subdirectory(convert-llama2c-to-ggml)
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
        add_subdirectory(llava)
        if (GGML_RPC)
            add_subdirectory(rpc)
        endif()
        if (GGML_SYCL)
            add_subdirectory(sycl)
        endif()
    endif()

    find_package(Qt6 COMPONENTS Core Widgets Charts)
    find_package(SQLite3)
    if(Qt6_FOUND AND SQLite3_FOUND)
        option(LLAMA_BUILD_EXAMPLES_GUI "Build GUI examples" ON)
        if(LLAMA_BUILD_EXAMPLES_GUI)
            add_subdirectory(gui)
        endif()
    else()
        message(STATUS "Qt6 or SQLite3 not found, GUI examples will not be built")
    endif()
endif()
