#pragma once

#include <cstdio>
#include <iostream>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace guild::cli {

inline void enable_virtual_terminal() {
#if defined(_WIN32)
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE) {
            DWORD mode = 0;
            if (GetConsoleMode(hOut, &mode)) {
                SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
            }
        }
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    }
#endif
}

inline bool is_tty() {
#if defined(_WIN32)
    enable_virtual_terminal();
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

namespace ansi {
inline const char* reset() {
    return is_tty() ? "\033[0m" : "";
}
inline const char* bold() {
    return is_tty() ? "\033[1m" : "";
}
inline const char* dim() {
    return is_tty() ? "\033[2m" : "";
}
inline const char* cyan() {
    return is_tty() ? "\033[36m" : "";
}
inline const char* red() {
    return is_tty() ? "\033[31m" : "";
}
inline const char* green() {
    return is_tty() ? "\033[32m" : "";
}
inline const char* yellow() {
    return is_tty() ? "\033[33m" : "";
}
inline const char* blue() {
    return is_tty() ? "\033[34m" : "";
}
inline const char* magenta() {
    return is_tty() ? "\033[35m" : "";
}
inline const char* white() {
    return is_tty() ? "\033[37m" : "";
}
inline const char* gray() {
    return is_tty() ? "\033[90m" : "";
}
} // namespace ansi

} // namespace guild::cli
