# Canonical independently owned CUDA arithmetic and shared precision headers.
file(READ "${CMAKE_CURRENT_LIST_DIR}/../backends.lock.json" _cuda_lock)
string(JSON _locked_cuda_revision GET "${_cuda_lock}" cuda revision)
set(GUILD_CUDA_SOURCE_DIR "" CACHE PATH "Independent CUDA development checkout; empty uses the locked revision")
set(GUILD_CUDA_BACKEND_REVISION "${_locked_cuda_revision}" CACHE STRING "Pinned CUDA source revision")
if(GUILD_CUDA_SOURCE_DIR)
  get_filename_component(_guild_cuda_source "${GUILD_CUDA_SOURCE_DIR}" ABSOLUTE)
else()
  include(FetchContent)
  FetchContent_Declare(guild_cuda_backend
    GIT_REPOSITORY https://github.com/guild-inference/guild-kernels-cuda.git
    GIT_TAG ${GUILD_CUDA_BACKEND_REVISION} GIT_SHALLOW FALSE)
  FetchContent_GetProperties(guild_cuda_backend)
  if(NOT guild_cuda_backend_POPULATED)
    FetchContent_Populate(guild_cuda_backend)
  endif()
  set(_guild_cuda_source "${guild_cuda_backend_SOURCE_DIR}")
  if(NOT EXISTS "${_guild_cuda_source}/cmake/GuildCUDADenseSources.cmake")
    message(FATAL_ERROR "Pinned CUDA sources absent; allow the initial dependency download or set GUILD_CUDA_SOURCE_DIR")
  endif()
  find_package(Git REQUIRED)
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD WORKING_DIRECTORY "${_guild_cuda_source}"
    OUTPUT_VARIABLE _cuda_actual OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _cuda_status)
  execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${_guild_cuda_source}" OUTPUT_VARIABLE _cuda_changes RESULT_VARIABLE _cuda_dirty_status)
  if(NOT _cuda_status EQUAL 0 OR NOT _cuda_dirty_status EQUAL 0 OR _cuda_changes OR
     NOT _cuda_actual STREQUAL GUILD_CUDA_BACKEND_REVISION)
    message(FATAL_ERROR "CUDA source checkout differs from its immutable pin; use a source override for development")
  endif()
endif()
include("${_guild_cuda_source}/cmake/GuildCUDADenseSources.cmake")
include("${_guild_cuda_source}/cmake/GuildCUDAAttnSources.cmake")
include("${_guild_cuda_source}/cmake/GuildCUDAMoESources.cmake")
include_directories("${_guild_cuda_source}/include")
