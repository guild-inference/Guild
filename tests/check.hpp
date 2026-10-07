#pragma once

#include <cstdio>
#include <cstdlib>

// Unlike assert(), these checks (and their side effects) run in Release builds.
#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)
