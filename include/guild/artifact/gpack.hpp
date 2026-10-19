#pragma once

#include "guild/abi/kernel_abi.h"
#include "guild/artifact/gguf_reader.hpp"
#include "guild/model/model_descriptor.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace guild::artifact {

inline constexpr uint32_t kGPackMagic = 0x4B435047u;       // "GPCK"
inline constexpr uint32_t kGPackFooterMagic = 0x454B5047u; // "GPKE"
inline constexpr uint16_t kGPackVersionV1 = 1u;
inline constexpr uint32_t kGPackTokenizerMagic = 0x31544247u; // "GBT1"

enum GPackFlags : uint32_t {
    GUILD_GPACK_FLAG_LITTLE_ENDIAN    = 1u << 0,
    GUILD_GPACK_FLAG_HAS_DENSE_PACK   = 1u << 1,
    GUILD_GPACK_FLAG_HAS_NATIVE_DENSE = 1u << 2,
    GUILD_GPACK_FLAG_HAS_EXPERT_BLOBS = 1u << 3,
    GUILD_GPACK_FLAG_HAS_TOKENIZER_BIN = 1u << 4,
    GUILD_GPACK_FLAG_HAS_MTP          = 1u << 5,
    GUILD_GPACK_FLAG_HAS_PLE          = 1u << 6,
    GUILD_GPACK_FLAG_EXTERNAL_EXPERTS = 1u << 7,
};

enum class GPackSectionKind : uint32_t {
    ManifestJson        = 1,
    MetadataKvBin       = 2,
    TensorIndexBin      = 3,
    DenseIndexBin       = 4,
    DensePackBlob       = 5,
    NativeDenseIndexBin = 6,
    NativeDenseBlob     = 7,
    ExpertLayoutBin     = 8,
    ExpertBlobs         = 9,
    EmbedBlob           = 10,
    HeadBlob            = 11,
    PleBlob             = 12,
    TokenizerBin        = 13,
    TokenizerAssets     = 14,
    MtpBlob             = 15,
};

#pragma pack(push, 1)
struct GPackHeaderV1 {
    uint32_t magic = kGPackMagic;
    uint16_t format_version = kGPackVersionV1;
    uint16_t min_reader_version = kGPackVersionV1;
    uint32_t flags = GUILD_GPACK_FLAG_LITTLE_ENDIAN;
    uint32_t section_count = 0;
    uint64_t manifest_offset = 0;
    uint64_t manifest_size = 0;
    uint64_t section_table_offset = 0;
    uint64_t total_file_size = 0;
    uint8_t header_manifest_sha256_prefix[16] = {};
};
static_assert(sizeof(GPackHeaderV1) == 64, "GPackHeaderV1 must be 64 bytes");

struct GPackSectionEntryV1 {
    uint32_t kind = 0;
    uint32_t flags = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint64_t alignment = 64;
    uint64_t item_count = 0;
    uint8_t sha256[32] = {};
    char name[56] = {};
};
static_assert(sizeof(GPackSectionEntryV1) == 128, "GPackSectionEntryV1 must be 128 bytes");

struct GPackFooterV1 {
    uint64_t manifest_offset = 0;
    uint64_t section_table_offset = 0;
    uint64_t total_payload_bytes = 0;
    uint8_t manifest_sha256[32] = {};
    uint32_t section_count = 0;
    uint32_t magic = kGPackFooterMagic;
};
static_assert(sizeof(GPackFooterV1) == 64, "GPackFooterV1 must be 64 bytes");

struct GPackSingleTensorHeaderV1 {
    int32_t ggml_type = -1;
    int32_t reserved = 0;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    uint64_t byte_size = 0;
    uint64_t data_offset = 64; // relative to section start (64-byte aligned)
    uint8_t reserved_pad[24] = {};
};
static_assert(sizeof(GPackSingleTensorHeaderV1) == 64, "GPackSingleTensorHeaderV1 must be 64 bytes");
#pragma pack(pop)

struct GPackDenseRowV1 {
    std::string name;
    int32_t file_id = 0;
    int32_t kind = 0;
    uint64_t src_off = 0;
    uint64_t src_bytes = 0;
    uint64_t dst_off = 0;
    uint64_t dst_bytes = 0;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int32_t code_bits = 0;
    int32_t code_bias = 0;
    int32_t group_elems = 0;
    int32_t codebook = 0;
    int32_t has_offset = 0;
    uint64_t codes_bytes = 0;
    uint64_t scales_bytes = 0;
    uint64_t offset_bytes = 0;
    int32_t scales_fp16 = 0;
    int32_t act_kind = 0;
    bool raw16 = false;
    bool skip_in_native = false;
};

struct GPackNativeDenseRowV1 {
    std::string name;
    int32_t ggml_type = -1;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    uint64_t blob_offset = 0;
    uint64_t byte_size = 0;
    uint64_t arena_offset = 0;
    uint64_t aligned_size = 0;
    bool is_ple_key = false;
};

struct GPackExpertLayerRowV1 {
    int64_t layer = 0;
    int32_t gu_type = -1;
    int32_t d_type = -1;
    uint64_t layer_offset = 0;
    uint64_t blob_bytes = 0;
    uint64_t gate_off = 0;
    uint64_t up_off = 0;
    uint64_t down_off = 0;
    uint64_t gguf_gate_off = 0;
    uint64_t gguf_up_off = 0;
    uint64_t gguf_down_off = 0;
    std::string gate_shard;
    std::string up_shard;
    std::string down_shard;
};

struct GPackExecutionMetadata {
    std::string runner_id;
    std::string arch_id;
    std::string prefill_strategy = "sequential_verifier";
    uint32_t required_core_caps = 0;
    std::vector<uint32_t> required_ops;
    std::vector<std::string> required_modules;
    bool requires_ple = false;
    bool supports_mtp = false;
    std::vector<int64_t> eos_token_ids;
};

class GPackReader {
public:
    GPackReader();
    ~GPackReader();
    GPackReader(const GPackReader&) = delete;
    GPackReader& operator=(const GPackReader&) = delete;

    static bool is_gpack_file(const std::string& path);

    bool open(const std::string& path, std::string& err, bool verify_all_hashes = false);
    void close();

    bool is_open() const { return base_ != nullptr; }
    const std::string& path() const { return path_; }
    uint64_t file_size() const { return file_size_; }
    const GPackHeaderV1& header() const { return header_; }
    const GPackFooterV1& footer() const { return footer_; }
    const std::vector<GPackSectionEntryV1>& sections() const { return sections_; }
    const std::string& manifest_json() const { return manifest_json_; }
    std::string manifest_sha256_hex() const;

    const model::ModelDescriptor& descriptor() const { return desc_; }
    const GPackExecutionMetadata& execution_metadata() const { return exec_meta_; }
    const std::map<std::string, MetaValue>& metadata_kv() const { return metadata_kv_; }
    const MetaValue* get_meta(const std::string& key) const {
        auto it = metadata_kv_.find(key);
        return it != metadata_kv_.end() ? &it->second : nullptr;
    }

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* find_tensor(const std::string& name) const {
        auto it = tensor_by_name_.find(name);
        return it != tensor_by_name_.end() ? &tensors_[it->second] : nullptr;
    }
    guild_model_reader_v1_t make_abi_model_reader() const;

    const GPackSectionEntryV1* find_section(GPackSectionKind kind) const {
        const uint32_t k = static_cast<uint32_t>(kind);
        for (const auto& s : sections_) {
            if (s.kind == k) return &s;
        }
        return nullptr;
    }

    // Precomputed dense weight pack
    bool has_dense_pack() const { return (header_.flags & GUILD_GPACK_FLAG_HAS_DENSE_PACK) != 0; }
    uint64_t dense_pool_bytes() const { return dense_pool_bytes_; }
    uint64_t dense_compacted_pool_bytes() const { return dense_compacted_pool_bytes_; }
    const std::vector<GPackDenseRowV1>& dense_rows() const { return dense_rows_; }
    const std::set<std::string>& precomputed_skip_set() const { return precomputed_skip_set_; }
    const uint8_t* dense_pack_data(uint64_t* out_size = nullptr) const {
        const auto* sec = find_section(GPackSectionKind::DensePackBlob);
        if (!sec || !base_) return nullptr;
        if (out_size) *out_size = sec->size;
        return base_ + sec->offset;
    }

    // Precomputed native dense quantized projections
    bool has_native_dense() const { return (header_.flags & GUILD_GPACK_FLAG_HAS_NATIVE_DENSE) != 0; }
    uint64_t native_dense_arena_bytes() const { return native_dense_arena_bytes_; }
    const std::vector<GPackNativeDenseRowV1>& native_dense_rows() const { return native_dense_rows_; }
    const uint8_t* native_dense_data(uint64_t* out_size = nullptr) const {
        const auto* sec = find_section(GPackSectionKind::NativeDenseBlob);
        if (!sec || !base_) return nullptr;
        if (out_size) *out_size = sec->size;
        return base_ + sec->offset;
    }

    // Precomputed expert layout & contiguous expert blobs
    bool has_expert_layout() const { return !expert_layers_.empty(); }
    bool has_embedded_experts() const { return (header_.flags & GUILD_GPACK_FLAG_HAS_EXPERT_BLOBS) != 0; }
    bool has_external_experts() const { return (header_.flags & GUILD_GPACK_FLAG_EXTERNAL_EXPERTS) != 0; }
    bool expert_native_format() const { return expert_native_; }
    int64_t expert_count_per_layer() const { return expert_n_expert_; }
    uint64_t total_expert_bytes() const { return expert_total_bytes_; }
    const std::vector<GPackExpertLayerRowV1>& expert_layers() const { return expert_layers_; }
    const uint8_t* expert_blobs_data(uint64_t* out_size = nullptr, uint64_t* out_file_offset = nullptr) const {
        const auto* sec = find_section(GPackSectionKind::ExpertBlobs);
        if (!sec || !base_) return nullptr;
        if (out_size) *out_size = sec->size;
        if (out_file_offset) *out_file_offset = sec->offset;
        return base_ + sec->offset;
    }
    std::string external_expert_primary_gguf() const { return external_expert_gguf_; }

    // Pre-validated single tensors (Embedding, Output Head, PLE Table)
    bool embed_tensor(GPackSingleTensorHeaderV1& out_hdr, const uint8_t*& out_data) const {
        const auto* sec = find_section(GPackSectionKind::EmbedBlob);
        if (!sec || !base_ || sec->size < sizeof(GPackSingleTensorHeaderV1)) return false;
        std::memcpy(&out_hdr, base_ + sec->offset, sizeof(out_hdr));
        if (out_hdr.data_offset < sizeof(GPackSingleTensorHeaderV1) ||
            out_hdr.data_offset > sec->size ||
            out_hdr.byte_size > sec->size - out_hdr.data_offset) {
            return false;
        }
        out_data = base_ + sec->offset + out_hdr.data_offset;
        return true;
    }
    bool head_tensor(GPackSingleTensorHeaderV1& out_hdr, const uint8_t*& out_data) const {
        const auto* sec = find_section(GPackSectionKind::HeadBlob);
        if (!sec || !base_ || sec->size < sizeof(GPackSingleTensorHeaderV1)) return false;
        std::memcpy(&out_hdr, base_ + sec->offset, sizeof(out_hdr));
        if (out_hdr.data_offset < sizeof(GPackSingleTensorHeaderV1) ||
            out_hdr.data_offset > sec->size ||
            out_hdr.byte_size > sec->size - out_hdr.data_offset) {
            return false;
        }
        out_data = base_ + sec->offset + out_hdr.data_offset;
        return true;
    }
    bool ple_tensor(GPackSingleTensorHeaderV1& out_hdr, const uint8_t*& out_data, uint64_t* out_file_offset = nullptr) const {
        const auto* sec = find_section(GPackSectionKind::PleBlob);
        if (!sec || !base_ || sec->size < sizeof(GPackSingleTensorHeaderV1)) return false;
        std::memcpy(&out_hdr, base_ + sec->offset, sizeof(out_hdr));
        if (out_hdr.data_offset < sizeof(GPackSingleTensorHeaderV1) ||
            out_hdr.data_offset > sec->size ||
            out_hdr.byte_size > sec->size - out_hdr.data_offset) {
            return false;
        }
        out_data = base_ + sec->offset + out_hdr.data_offset;
        if (out_file_offset) *out_file_offset = sec->offset + out_hdr.data_offset;
        return true;
    }

    // Precomputed binary tokenizer (GBT1) & raw assets
    bool has_tokenizer_bin() const { return (header_.flags & GUILD_GPACK_FLAG_HAS_TOKENIZER_BIN) != 0; }
    const uint8_t* tokenizer_bin_data(uint64_t* out_size = nullptr) const {
        const auto* sec = find_section(GPackSectionKind::TokenizerBin);
        if (!sec || !base_) return nullptr;
        if (out_size) *out_size = sec->size;
        return base_ + sec->offset;
    }
    bool extract_tokenizer_assets(const std::string& dst_dir, std::string& err) const;

    // Optional MTP draft model payload
    bool has_mtp() const { return (header_.flags & GUILD_GPACK_FLAG_HAS_MTP) != 0; }
    const uint8_t* mtp_blob_data(uint64_t* out_size = nullptr) const {
        const auto* sec = find_section(GPackSectionKind::MtpBlob);
        if (!sec || !base_) return nullptr;
        if (out_size) *out_size = sec->size;
        return base_ + sec->offset;
    }

private:
    std::string path_;
    const uint8_t* base_ = nullptr;
    uint64_t file_size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif

    GPackHeaderV1 header_{};
    GPackFooterV1 footer_{};
    std::vector<GPackSectionEntryV1> sections_;
    std::string manifest_json_;

    model::ModelDescriptor desc_{};
    GPackExecutionMetadata exec_meta_{};
    std::map<std::string, MetaValue> metadata_kv_;
    std::vector<TensorInfo> tensors_;
    std::map<std::string, size_t> tensor_by_name_;

    uint64_t dense_pool_bytes_ = 0;
    uint64_t dense_compacted_pool_bytes_ = 0;
    std::vector<GPackDenseRowV1> dense_rows_;
    std::set<std::string> precomputed_skip_set_;

    uint64_t native_dense_arena_bytes_ = 0;
    std::vector<GPackNativeDenseRowV1> native_dense_rows_;

    bool expert_native_ = true;
    int64_t expert_n_expert_ = 0;
    uint64_t expert_total_bytes_ = 0;
    std::vector<GPackExpertLayerRowV1> expert_layers_;
    std::string external_expert_gguf_;
};

struct GPackBuildOptions {
    std::string primary_input_path;
    std::vector<std::string> additional_shards;
    std::string pack_dir;
    std::string tokenizer_dir;
    std::string mtp_dir;
    std::string ple_gguf;
    std::string output_path;
    std::string model_name;
    bool embed_experts = true;
    bool verify_after_write = true;
};

struct GPackBuildReport {
    std::string output_path;
    std::string manifest_sha256;
    uint64_t total_file_size = 0;
    uint64_t dense_pack_bytes = 0;
    uint64_t native_dense_bytes = 0;
    uint64_t expert_blob_bytes = 0;
    uint64_t embed_bytes = 0;
    uint64_t head_bytes = 0;
    uint64_t tokenizer_bin_bytes = 0;
    size_t tensor_count = 0;
    size_t section_count = 0;
    double build_ms = 0.0;
    double verify_ms = 0.0;
};

class GPackBuilder {
public:
    static bool build(const GPackBuildOptions& opts, GPackBuildReport& report, std::string& err);
    static bool build_from_safetensors(const GPackBuildOptions& opts, GPackBuildReport& report, std::string& err);
};

} // namespace guild::artifact
