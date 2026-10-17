# The independent repository is the canonical owner of bundled CPU arithmetic.
set(GUILD_CPU_SOURCE_DIR "" CACHE PATH "Independent guild-kernels-cpu checkout (empty: fetch pinned revision)")
file(READ "${CMAKE_CURRENT_LIST_DIR}/../backends.lock.json" _backend_lock_json)
string(JSON _locked_cpu_revision GET "${_backend_lock_json}" cpu revision)
set(GUILD_CPU_BACKEND_REVISION "${_locked_cpu_revision}" CACHE STRING "Pinned bundled CPU source revision")
if(GUILD_CPU_SOURCE_DIR)
  get_filename_component(_guild_cpu_source "${GUILD_CPU_SOURCE_DIR}" ABSOLUTE)
else()
  include(FetchContent)
  FetchContent_Declare(guild_cpu_backend
    GIT_REPOSITORY https://github.com/guild-inference/guild-kernels-cpu.git
    GIT_TAG ${GUILD_CPU_BACKEND_REVISION} GIT_SHALLOW FALSE)
  FetchContent_GetProperties(guild_cpu_backend)
  if(NOT guild_cpu_backend_POPULATED)
    FetchContent_Populate(guild_cpu_backend)
  endif()
  set(_guild_cpu_source "${guild_cpu_backend_SOURCE_DIR}")
  if(NOT EXISTS "${_guild_cpu_source}/cmake/GuildCPUSources.cmake")
    message(FATAL_ERROR "Pinned CPU sources are absent. Allow the initial FetchContent download or set GUILD_CPU_SOURCE_DIR to a compatible independent checkout (FETCHCONTENT_FULLY_DISCONNECTED=${FETCHCONTENT_FULLY_DISCONNECTED}).")
  endif()
  find_package(Git REQUIRED)
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD WORKING_DIRECTORY "${_guild_cpu_source}"
    OUTPUT_VARIABLE _guild_cpu_actual OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _guild_cpu_git_status)
  if(NOT _guild_cpu_git_status EQUAL 0 OR NOT _guild_cpu_actual STREQUAL GUILD_CPU_BACKEND_REVISION)
    message(FATAL_ERROR "CPU source revision '${_guild_cpu_actual}' does not match pinned '${GUILD_CPU_BACKEND_REVISION}'. Update the cached dependency before an offline build.")
  endif()
  execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${_guild_cpu_source}" OUTPUT_VARIABLE _guild_cpu_changes
    RESULT_VARIABLE _guild_cpu_status)
  if(NOT _guild_cpu_status EQUAL 0 OR _guild_cpu_changes)
    message(FATAL_ERROR "Fetched CPU source has local changes; use GUILD_CPU_SOURCE_DIR for development overrides")
  endif()
endif()
include("${_guild_cpu_source}/cmake/GuildCPUSources.cmake")
