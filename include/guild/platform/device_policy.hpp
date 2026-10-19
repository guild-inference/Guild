// Configure CUDA module policy before the first runtime/device discovery call.
#pragma once
#include <cstdlib>

namespace guild::platform {
inline void configure_device_runtime_policy() {
    // Preserve an explicit embedding application's choice. CUDA reads this when
    // it initializes, so setting it later in model loading has no effect after
    // package validation or hardware discovery initialized the runtime first.
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    }
}
} // namespace guild::platform
