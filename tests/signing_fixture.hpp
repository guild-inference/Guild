// Public deterministic TEST VECTOR, never a release signing identity.
#pragma once
#include "guild/abi/gkernel_package.hpp"
namespace guild::abi {
inline constexpr const char* kOfficialGuildDevSeedHex =
    "4755494c445f4f4646494349414c5f4445565f534545445f323032365f563031";
inline std::string official_guild_public_key_hex() {
    crypto::Ed25519KeyPair key;
    crypto::ed25519_keypair_from_seed_hex(kOfficialGuildDevSeedHex, key);
    return key.public_key_hex;
}
inline void trust_test_signing_key(PackageSecurityPolicy& policy) {
    policy.trust_publisher_key(kOfficialGuildPublisherId, kOfficialGuildReleaseKeyId, official_guild_public_key_hex());
}
} // namespace guild::abi
