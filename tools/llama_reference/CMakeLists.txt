cmake_minimum_required(VERSION 3.18)
project(guild_llama_reference LANGUAGES C CXX)

# Independent oracle build: the caller supplies a pinned, local llama.cpp checkout.
# No FetchContent or network dependency is introduced into benchmark execution.
set(LLAMA_SOURCE_DIR "" CACHE PATH "Local llama.cpp checkout")
set(LLAMA_REVISION "" CACHE STRING "Required full llama.cpp commit SHA")
if(NOT EXISTS "${LLAMA_SOURCE_DIR}/CMakeLists.txt" OR NOT LLAMA_REVISION)
  message(FATAL_ERROR "Set LLAMA_SOURCE_DIR and LLAMA_REVISION (full commit SHA)")
endif()
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD WORKING_DIRECTORY "${LLAMA_SOURCE_DIR}"
  OUTPUT_VARIABLE actual OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
  WORKING_DIRECTORY "${LLAMA_SOURCE_DIR}" OUTPUT_VARIABLE changes COMMAND_ERROR_IS_FATAL ANY)
if(NOT actual STREQUAL LLAMA_REVISION OR changes)
  message(FATAL_ERROR "llama.cpp source must be clean and match LLAMA_REVISION")
endif()
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_SOURCE_DIR}" llama)
add_executable(llama_prefill_reference ../llama_prefill_reference.cpp)
target_compile_features(llama_prefill_reference PRIVATE cxx_std_17)
target_link_libraries(llama_prefill_reference PRIVATE llama ggml)
