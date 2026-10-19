#pragma once

#include <mutex>

namespace guild::abi {
// Lock order: package-manager mutex -> lifecycle gate -> individual registry.
// Only discovery/publication/removal use this gate, never inference execution.
// Recursive because a package transaction commits through both registry APIs.
inline std::recursive_mutex& provider_lifecycle_mutex() {
    static std::recursive_mutex mutex;
    return mutex;
}
using ProviderTransaction = std::lock_guard<std::recursive_mutex>;
} // namespace guild::abi
