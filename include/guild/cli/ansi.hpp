#pragma once

#include <iostream>
#include <string>
#include <unistd.h>

namespace guild::cli {

inline bool is_tty() {
#if defined(_WIN32)
    return true;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

namespace ansi {
    inline const char* reset()   { return is_tty() ? "\033[0m" : ""; }
    inline const char* bold()    { return is_tty() ? "\033[1m" : ""; }
    inline const char* dim()     { return is_tty() ? "\033[2m" : ""; }
    inline const char* cyan()    { return is_tty() ? "\033[36m" : ""; }
    inline const char* green()   { return is_tty() ? "\033[32m" : ""; }
    inline const char* yellow()  { return is_tty() ? "\033[33m" : ""; }
    inline const char* blue()    { return is_tty() ? "\033[34m" : ""; }
    inline const char* magenta() { return is_tty() ? "\033[35m" : ""; }
    inline const char* white()   { return is_tty() ? "\033[37m" : ""; }
    inline const char* gray()    { return is_tty() ? "\033[90m" : ""; }
}  // namespace ansi

}  // namespace guild::cli
