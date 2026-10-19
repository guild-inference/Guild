// include/guild/abi/ed25519.hpp - RFC 8032 Ed25519 Signatures & SHA-512 for .gkernel Verification
//
// Copyright (c) Guild Inference Contributors. MIT License.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace guild::abi::crypto {

struct Ed25519KeyPair {
    std::array<uint8_t, 32> seed{};
    std::array<uint8_t, 32> public_key{};
    std::array<uint8_t, 64> secret_key{}; // seed (32B) || public_key (32B)
    std::string seed_hex;
    std::string public_key_hex;
};

/// Compute standard FIPS 180-4 SHA-512 (64 bytes).
void sha512(const uint8_t* data, size_t len, uint8_t out_digest[64]);

/// Derive an RFC 8032 Ed25519 keypair from a 32-byte private seed.
Ed25519KeyPair ed25519_keypair_from_seed(const uint8_t seed[32]);

/// Derive an RFC 8032 Ed25519 keypair from a 64-character hex seed string.
bool ed25519_keypair_from_seed_hex(const std::string& seed_hex, Ed25519KeyPair& out_kp);

/// Generate a fresh random 32-byte seed and derive an Ed25519 keypair.
bool ed25519_generate_keypair(Ed25519KeyPair& out_kp);

/// Compute an RFC 8032 Ed25519 64-byte signature over `msg[0..msg_len)`.
void ed25519_sign(const uint8_t* msg, size_t msg_len, const uint8_t seed[32], const uint8_t public_key[32],
                  uint8_t out_sig[64]);

/// Verify an RFC 8032 Ed25519 64-byte signature over `msg[0..msg_len)` against `public_key[32]`.
/// Returns true iff the signature is cryptographically valid (including canonical S < L check).
bool ed25519_verify(const uint8_t* msg, size_t msg_len, const uint8_t sig[64], const uint8_t public_key[32]);

/// Sign a string message using a 64-char hex seed, returning a 128-char lowercase hex signature.
bool ed25519_sign_hex(const std::string& message, const std::string& seed_hex, std::string& out_sig_hex,
                      std::string* out_pubkey_hex = nullptr);

/// Verify a 128-char hex signature over `message` against a 64-char hex public key.
bool ed25519_verify_hex(const std::string& message, const std::string& sig_hex, const std::string& pubkey_hex);

/// Hex encoding/decoding utilities.
std::string bytes_to_hex(const uint8_t* data, size_t len);
bool hex_to_bytes(const std::string& hex, uint8_t* out, size_t expected_bytes);

} // namespace guild::abi::crypto
