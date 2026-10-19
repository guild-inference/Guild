# cmake/QwenRunnersBackend.cmake
#
# Resolves the extracted guild-runners-qwen repository for the default bundled
# build mode (GUILD_BUILTIN_QWEN_RUNNERS=ON). When GUILD_BUILTIN_QWEN_RUNNERS=OFF,
# Guild Core compiles without bundled Qwen model runner definitions and relies
# on signed external .gkernel runner packages at runtime.

include(FetchContent)
file(READ "${CMAKE_CURRENT_LIST_DIR}/../backends.lock.json" _runner_lock)
string(JSON _locked_runner_revision GET "${_runner_lock}" qwen_runners revision)
string(JSON GUILD_QWEN_RUNNERS_GIT_REPOSITORY GET "${_runner_lock}" qwen_runners repository)
set(GUILD_QWEN_RUNNERS_SOURCE_DIR "" CACHE PATH "Independent runner development checkout; empty uses the locked revision")
set(GUILD_QWEN_RUNNERS_GIT_TAG "${_locked_runner_revision}" CACHE STRING "Pinned runner source revision")

function(guild_resolve_qwen_runners_backend)
    if(NOT GUILD_BUILTIN_QWEN_RUNNERS)
        set(GUILD_QWEN_RUNNERS_BACKEND_SOURCES "" PARENT_SCOPE)
        set(GUILD_QWEN_RUNNERS_BACKEND_INCLUDE_DIRS "" PARENT_SCOPE)
        return()
    endif()

    if(GUILD_QWEN_RUNNERS_SOURCE_DIR)
        if(NOT EXISTS "${GUILD_QWEN_RUNNERS_SOURCE_DIR}/cmake/GuildQwenRunnersSources.cmake")
            message(FATAL_ERROR
                "GUILD_QWEN_RUNNERS_SOURCE_DIR='${GUILD_QWEN_RUNNERS_SOURCE_DIR}' does not contain "
                "cmake/GuildQwenRunnersSources.cmake")
        endif()
        set(_qwen_runners_root "${GUILD_QWEN_RUNNERS_SOURCE_DIR}")
        message(STATUS "Guild Qwen runners backend: using explicit source directory ${_qwen_runners_root}")
    else()
        message(STATUS
            "Guild Qwen runners backend: fetching ${GUILD_QWEN_RUNNERS_GIT_REPOSITORY} "
            "@ ${GUILD_QWEN_RUNNERS_GIT_TAG}")
        FetchContent_Declare(
            guild_runners_qwen_ext
            GIT_REPOSITORY "${GUILD_QWEN_RUNNERS_GIT_REPOSITORY}"
            GIT_TAG        "${GUILD_QWEN_RUNNERS_GIT_TAG}"
        )
        FetchContent_Populate(guild_runners_qwen_ext)
        set(_qwen_runners_root "${guild_runners_qwen_ext_SOURCE_DIR}")
        find_package(Git REQUIRED)
        execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD WORKING_DIRECTORY "${_qwen_runners_root}"
            OUTPUT_VARIABLE _actual OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
        execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
            WORKING_DIRECTORY "${_qwen_runners_root}" OUTPUT_VARIABLE _dirty COMMAND_ERROR_IS_FATAL ANY)
        if(_dirty OR NOT _actual STREQUAL GUILD_QWEN_RUNNERS_GIT_TAG)
            message(FATAL_ERROR "Runner source checkout differs from its immutable pin; use a source override for development")
        endif()
    endif()

    set(GUILD_QWEN_RUNNERS_ROOT "${_qwen_runners_root}")
    set(GUILD_QWEN_RUNNERS_ROOT "${_qwen_runners_root}" PARENT_SCOPE)
    include("${_qwen_runners_root}/cmake/GuildQwenRunnersSources.cmake")

    set(GUILD_QWEN_RUNNERS_BACKEND_SOURCES "${GUILD_QWEN_RUNNERS_SOURCES}" PARENT_SCOPE)
    set(GUILD_QWEN_RUNNERS_BACKEND_INCLUDE_DIRS "${GUILD_QWEN_RUNNERS_INCLUDE_DIRS}" PARENT_SCOPE)
endfunction()
