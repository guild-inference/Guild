// The `.gpack` reader: a read-only, memory-mapped container (header, sections, payload, footer).  Two things are
// worth knowing before reading the code:
//
//  * Every section is checksummed three ways - the SHA-256 in the footer's directory, in the section entry, and in
//    the header prefix - so `open()` can detect a container that was edited, truncated or reassembled by comparing
//    the three instead of trusting any single one.  `verify_all_hashes` controls whether the payload is re-hashed
//    on open: production paths leave it off (a full re-hash on every start would dominate load time and the pack
//    was verified when it was written), while tools and tests that handle a file from elsewhere turn it on.
//  * The mapping *is* the model.  `base_` points into the file for the life of the reader: tensors, the expert blob
//    and the tokenizer bytes are all views, never copies, which is why `close()` must not run while a session is
//    executing (the kernel lease in the runtime is what guarantees that ordering).

#include "guild/artifact/gpack.hpp"
#include "guild/artifact/gguf_split.hpp"
#include "guild/model/archetype.hpp"
#include "guild/model/runner.hpp"
#include "guild/models/sha256.hpp"
#include "guild/runtime/tokenizer.hpp"
#include "guild/server/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace guild::artifact {

namespace fs = std::filesystem;
using namespace guild::server::json;
using Clock = std::chrono::steady_clock;

namespace {

uint64_t align_up(uint64_t v, uint64_t a) {
    if (a <= 1)
        return v;
    return (v + a - 1) / a * a;
}

bool is_power_of_two(uint64_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

std::string bytes_to_hex(const uint8_t* bytes, size_t n) {
    static const char* lut = "0123456789abcdef";
    std::string out(n * 2, '0');
    for (size_t i = 0; i < n; ++i) {
        out[2 * i] = lut[(bytes[i] >> 4) & 0xF];
        out[2 * i + 1] = lut[bytes[i] & 0xF];
    }
    return out;
}

bool hex_to_bytes(const std::string& hex, uint8_t* out, size_t n) {
    if (hex.size() != n * 2)
        return false;
    auto val = [](char c) -> int {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < n; ++i) {
        int hi = val(hex[2 * i]);
        int lo = val(hex[2 * i + 1]);
        if (hi < 0 || lo < 0)
            return false;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

void sha256_bytes(const void* data, size_t len, uint8_t out[32]) {
    models::Sha256 hasher;
    if (len > 0 && data != nullptr) {
        hasher.update(data, len);
    }
    const std::string hex = hasher.finalize_hex();
    hex_to_bytes(hex, out, 32);
}

class BinaryWriter {
public:
    std::vector<uint8_t> buf;
    void u8(uint8_t v) {
        buf.push_back(v);
    }
    void i32(int32_t v) {
        append(&v, sizeof(v));
    }
    void u32(uint32_t v) {
        append(&v, sizeof(v));
    }
    void i64(int64_t v) {
        append(&v, sizeof(v));
    }
    void u64(uint64_t v) {
        append(&v, sizeof(v));
    }
    void f64(double v) {
        append(&v, sizeof(v));
    }
    void str(const std::string& s) {
        u32(static_cast<uint32_t>(s.size()));
        if (!s.empty())
            append(s.data(), s.size());
    }
    void append(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        buf.insert(buf.end(), b, b + n);
    }
};

class BinaryReader {
public:
    const uint8_t* p = nullptr;
    size_t n = 0;
    size_t at = 0;

    BinaryReader(const uint8_t* ptr, size_t size) : p(ptr), n(size), at(0) {
    }

    bool read_bytes(void* dst, size_t len) {
        if (at > n || len > n - at)
            return false;
        if (len > 0)
            std::memcpy(dst, p + at, len);
        at += len;
        return true;
    }
    bool u8(uint8_t& v) {
        return read_bytes(&v, sizeof(v));
    }
    bool i32(int32_t& v) {
        return read_bytes(&v, sizeof(v));
    }
    bool u32(uint32_t& v) {
        return read_bytes(&v, sizeof(v));
    }
    bool i64(int64_t& v) {
        return read_bytes(&v, sizeof(v));
    }
    bool u64(uint64_t& v) {
        return read_bytes(&v, sizeof(v));
    }
    bool f64(double& v) {
        return read_bytes(&v, sizeof(v));
    }
    bool str(std::string& s, uint32_t max_len = 16u << 20) {
        uint32_t len = 0;
        if (!u32(len) || len > max_len || at > n || len > n - at)
            return false;
        s.assign(reinterpret_cast<const char*>(p + at), len);
        at += len;
        return true;
    }
};

bool eligible_native_dense(const guild::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0)
        return false;
    if (name == "blk.1.ple_key.weight") {
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    }
    constexpr const char* suffixes[] = {
        ".attn_qkv.weight",     ".attn_gate.weight",      ".ssm_out.weight",       ".attn_q.weight",
        ".attn_k.weight",       ".attn_v.weight",         ".attn_output.weight",   ".ffn_gate_shexp.weight",
        ".ffn_up_shexp.weight", ".ffn_down_shexp.weight", ".nextn.eh_proj.weight",
    };
    for (const char* suffix : suffixes) {
        const size_t n = std::strlen(suffix);
        if (name.size() > n && name.compare(name.size() - n, n, suffix) == 0)
            return true;
    }
    return false;
}

bool supported_native_mmvq_type(int type, int64_t ne0, int64_t ne1) {
    if (ne0 <= 0 || ne1 <= 0)
        return false;
    switch (type) {
    case 2:  // Q4_0
    case 6:  // Q5_0
    case 7:  // Q5_1
    case 8:  // Q8_0
    case 11: // Q3_K
    case 12: // Q4_K
    case 13: // Q5_K
    case 14: // Q6_K
    case 16: // IQ2_XXS
    case 17: // IQ2_XS
    case 18: // IQ3_XXS
    case 19: // IQ1_S
    case 20: // IQ4_NL
    case 21: // IQ3_S
    case 22: // IQ2_S
    case 23: // IQ4_XS
    case 29: // IQ1_M
    case 42: // Q2_0
        return true;
    default: return false;
    }
}

bool parse_pack_index_file(const std::string& index_path, uint64_t& out_pool_bytes,
                           std::vector<GPackDenseRowV1>& out_rows, std::string& err) {
    std::ifstream in(index_path);
    if (!in.is_open()) {
        err = "cannot open pack index: " + index_path;
        return false;
    }
    out_rows.clear();
    out_pool_bytes = 0;
    bool have_header = false;
    size_t want_n = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        if (line[0] == '#') {
            unsigned long long a = 0, pb = 0, tn = 0;
            if (std::sscanf(line.c_str(), "# align %llu pool %llu tensors %llu", &a, &pb, &tn) == 3) {
                if (a != 64) {
                    err = "pack index alignment must be 64";
                    return false;
                }
                out_pool_bytes = pb;
                want_n = static_cast<size_t>(tn);
                have_header = true;
            }
            continue;
        }
        char name_buf[256] = {};
        int file_id = 0, kind = 0, cb = 0, bias = 0, grp = 0, cbk = 0, hoff = 0, sfp16 = 0, ak = 0, raw16 = 0;
        unsigned long long so = 0, sb = 0, doff = 0, db = 0, cbytes = 0, sbytes = 0, obytes = 0;
        long long n0 = 0, n1 = 0;
        int n =
            std::sscanf(line.c_str(), "%255s %d %d %llu %llu %llu %llu %lld %lld %d %d %d %d %d %llu %llu %llu %d %d",
                        name_buf, &file_id, &kind, &so, &sb, &doff, &db, &n0, &n1, &cb, &bias, &grp, &cbk, &hoff,
                        &cbytes, &sbytes, &obytes, &sfp16, &ak);
        if (n < 19) {
            err = "malformed row in pack index: " + line;
            return false;
        }
        if (kind == 4 || kind == 5) {
            raw16 = 1;
            kind = (kind == 4) ? 1 : 3; // WeightKind::Bf16InF32 (1) or WeightKind::F16InF32 (3)
        }
        GPackDenseRowV1 row;
        row.name = name_buf;
        row.file_id = file_id;
        row.kind = kind;
        row.src_off = so;
        row.src_bytes = sb;
        row.dst_off = doff;
        row.dst_bytes = db;
        row.ne0 = n0;
        row.ne1 = n1;
        row.code_bits = cb;
        row.code_bias = bias;
        row.group_elems = grp;
        row.codebook = cbk;
        row.has_offset = hoff;
        row.codes_bytes = cbytes;
        row.scales_bytes = sbytes;
        row.offset_bytes = obytes;
        row.scales_fp16 = sfp16;
        row.act_kind = ak;
        row.raw16 = (raw16 != 0);
        out_rows.push_back(std::move(row));
    }
    if (!have_header || out_rows.size() != want_n) {
        err = "pack index row count mismatch in " + index_path;
        return false;
    }
    return true;
}

} // namespace

GPackReader::GPackReader() = default;

GPackReader::~GPackReader() {
    close();
}

bool GPackReader::is_gpack_file(const std::string& path) {
    if (path.empty())
        return false;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        return false;
    uint32_t magic = 0;
    if (!f.read(reinterpret_cast<char*>(&magic), sizeof(magic)))
        return false;
    return magic == kGPackMagic;
}

void GPackReader::close() {
    if (base_ != nullptr) {
#if defined(_WIN32)
        UnmapViewOfFile(base_);
        if (mapping_handle_)
            CloseHandle(static_cast<HANDLE>(mapping_handle_));
        if (file_handle_)
            CloseHandle(static_cast<HANDLE>(file_handle_));
        mapping_handle_ = nullptr;
        file_handle_ = nullptr;
#else
        munmap(const_cast<uint8_t*>(base_), static_cast<size_t>(file_size_));
        if (fd_ >= 0)
            ::close(fd_);
        fd_ = -1;
#endif
        base_ = nullptr;
    }
    file_size_ = 0;
    path_.clear();
    sections_.clear();
    manifest_json_.clear();
    metadata_kv_.clear();
    tensors_.clear();
    tensor_by_name_.clear();
    dense_rows_.clear();
    precomputed_skip_set_.clear();
    native_dense_rows_.clear();
    expert_layers_.clear();
}

bool GPackReader::open(const std::string& path, std::string& err, bool verify_all_hashes) {
    close();
    err.clear();

    if (path.empty()) {
        err = "GPACK open: empty path";
        return false;
    }

#if defined(_WIN32)
    HANDLE hf = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                            nullptr);
    if (hf == INVALID_HANDLE_VALUE) {
        err = "GPACK open: cannot open " + path;
        return false;
    }
    LARGE_INTEGER li{};
    if (!GetFileSizeEx(hf, &li) || li.QuadPart < static_cast<LONGLONG>(sizeof(GPackHeaderV1) + sizeof(GPackFooterV1))) {
        CloseHandle(hf);
        err = "GPACK open: file truncated or smaller than minimum GPACK container size: " + path;
        return false;
    }
    HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hm) {
        CloseHandle(hf);
        err = "GPACK open: CreateFileMapping failed: " + path;
        return false;
    }
    const void* view = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        CloseHandle(hm);
        CloseHandle(hf);
        err = "GPACK open: MapViewOfFile failed: " + path;
        return false;
    }
    file_handle_ = hf;
    mapping_handle_ = hm;
    base_ = static_cast<const uint8_t*>(view);
    file_size_ = static_cast<uint64_t>(li.QuadPart);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "GPACK open: cannot open " + path;
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(GPackHeaderV1) + sizeof(GPackFooterV1))) {
        ::close(fd);
        err = "GPACK open: file truncated or smaller than minimum GPACK container size: " + path;
        return false;
    }
    void* view = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) {
        ::close(fd);
        err = "GPACK open: mmap failed: " + path;
        return false;
    }
    fd_ = fd;
    base_ = static_cast<const uint8_t*>(view);
    file_size_ = static_cast<uint64_t>(st.st_size);
#endif
    path_ = path;

    std::memcpy(&header_, base_, sizeof(GPackHeaderV1));
    std::memcpy(&footer_, base_ + (file_size_ - sizeof(GPackFooterV1)), sizeof(GPackFooterV1));

    if (header_.magic != kGPackMagic) {
        err = "GPACK open: invalid header magic (expected GPCK)";
        close();
        return false;
    }
    if (footer_.magic != kGPackFooterMagic) {
        err = "GPACK open: invalid or missing footer magic (expected GPKE; file may be truncated)";
        close();
        return false;
    }
    if (header_.format_version != kGPackVersionV1 || header_.min_reader_version > kGPackVersionV1 ||
        header_.min_reader_version == 0) {
        err = "GPACK open: unsupported GPACK format version " + std::to_string(header_.format_version) +
              " (min_reader_version " + std::to_string(header_.min_reader_version) + ")";
        close();
        return false;
    }
    if ((header_.flags & GUILD_GPACK_FLAG_LITTLE_ENDIAN) == 0) {
        err = "GPACK open: container endianness flag mismatch";
        close();
        return false;
    }
    if (header_.total_file_size != file_size_) {
        err = "GPACK open: total_file_size mismatch (header=" + std::to_string(header_.total_file_size) +
              ", actual=" + std::to_string(file_size_) + ")";
        close();
        return false;
    }
    if (header_.manifest_offset != footer_.manifest_offset ||
        header_.section_table_offset != footer_.section_table_offset ||
        header_.section_count != footer_.section_count) {
        err = "GPACK open: header and footer directory pointers disagree";
        close();
        return false;
    }
    if (header_.section_count == 0 || header_.section_count > 256) {
        err = "GPACK open: invalid section_count " + std::to_string(header_.section_count);
        close();
        return false;
    }

    const uint64_t payload_end = file_size_ - sizeof(GPackFooterV1);
    const uint64_t table_bytes = static_cast<uint64_t>(header_.section_count) * sizeof(GPackSectionEntryV1);
    if (header_.section_table_offset < sizeof(GPackHeaderV1) || (header_.section_table_offset % 64) != 0 ||
        header_.section_table_offset > payload_end || table_bytes > payload_end - header_.section_table_offset) {
        err = "GPACK open: section table out of bounds or unaligned";
        close();
        return false;
    }

    sections_.resize(header_.section_count);
    std::memcpy(sections_.data(), base_ + header_.section_table_offset, static_cast<size_t>(table_bytes));

    struct Span {
        uint64_t begin;
        uint64_t end;
    };
    std::vector<Span> spans;
    spans.push_back({header_.section_table_offset, header_.section_table_offset + table_bytes});

    const GPackSectionEntryV1* manifest_sec = nullptr;
    for (size_t i = 0; i < sections_.size(); ++i) {
        const auto& s = sections_[i];
        if (std::memchr(s.name, '\0', sizeof(s.name)) == nullptr) {
            err = "GPACK open: section name is not null-terminated";
            close();
            return false;
        }
        if (!is_power_of_two(s.alignment) || (s.offset % s.alignment) != 0) {
            err = "GPACK open: section '" + std::string(s.name) + "' violates alignment " + std::to_string(s.alignment);
            close();
            return false;
        }
        if (s.offset < sizeof(GPackHeaderV1) || s.offset > payload_end || s.size > payload_end - s.offset) {
            err = "GPACK open: section '" + std::string(s.name) + "' exceeds file bounds";
            close();
            return false;
        }
        spans.push_back({s.offset, s.offset + s.size});
        if (static_cast<GPackSectionKind>(s.kind) == GPackSectionKind::ManifestJson) {
            if (manifest_sec != nullptr) {
                err = "GPACK open: duplicate ManifestJson section";
                close();
                return false;
            }
            manifest_sec = &s;
        }
        if (verify_all_hashes) {
            uint8_t actual_sha[32] = {};
            sha256_bytes(base_ + s.offset, static_cast<size_t>(s.size), actual_sha);
            if (std::memcmp(actual_sha, s.sha256, 32) != 0) {
                err = "GPACK open: SHA-256 integrity failure in section '" + std::string(s.name) + "'";
                close();
                return false;
            }
        }
    }

    std::sort(spans.begin(), spans.end(),
              [](const Span& a, const Span& b) { return a.begin != b.begin ? a.begin < b.begin : a.end < b.end; });
    for (size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].begin < spans[i - 1].end) {
            err = "GPACK open: overlapping section spans detected";
            close();
            return false;
        }
    }

    if (!manifest_sec || manifest_sec->offset != header_.manifest_offset ||
        manifest_sec->size != header_.manifest_size || manifest_sec->size == 0) {
        err = "GPACK open: missing or mismatched ManifestJson section";
        close();
        return false;
    }

    // Always verify manifest SHA-256 against header prefix, section entry, and footer
    uint8_t manifest_sha[32] = {};
    sha256_bytes(base_ + manifest_sec->offset, static_cast<size_t>(manifest_sec->size), manifest_sha);
    if (std::memcmp(manifest_sha, manifest_sec->sha256, 32) != 0 ||
        std::memcmp(manifest_sha, footer_.manifest_sha256, 32) != 0 ||
        std::memcmp(manifest_sha, header_.header_manifest_sha256_prefix, 16) != 0) {
        err = "GPACK open: manifest SHA-256 verification failed";
        close();
        return false;
    }

    manifest_json_.assign(reinterpret_cast<const char*>(base_ + manifest_sec->offset),
                          static_cast<size_t>(manifest_sec->size));

    JsonValue mroot_mut;
    std::string jerr;
    if (!JsonValue::parse(manifest_json_, mroot_mut, jerr) || !mroot_mut.is_object()) {
        err = "GPACK open: malformed manifest JSON: " + jerr;
        close();
        return false;
    }
    const JsonValue& mroot = mroot_mut;
    if (mroot["schema"].as_string() != "guild.gpack.v1" || mroot["format_version"].as_int(0) != 1) {
        err = "GPACK open: unsupported manifest schema or format_version";
        close();
        return false;
    }
    if (!mroot.contains("model") || !mroot["model"].is_object() || !mroot.contains("execution") ||
        !mroot["execution"].is_object()) {
        err = "GPACK open: manifest missing required 'model' or 'execution' object";
        close();
        return false;
    }

    const auto& mobj = mroot["model"];
    desc_ = model::ModelDescriptor{};
    desc_.name = mobj["name"].as_string();
    desc_.arch_name = mobj["architecture"].as_string();
    desc_.archetype = model::archetype_from_string(mobj["archetype"].as_string());
    if (desc_.archetype == model::ModelArchetype::Unknown) {
        desc_.archetype = model::archetype_from_string(desc_.arch_name);
    }
    desc_.file_path = path_;
    desc_.attn.context_length = mobj["context_length"].as_int(0);
    desc_.attn.vocab_size = mobj["vocab_size"].as_int(0);
    desc_.attn.n_embd = mobj["n_embd"].as_int(0);
    desc_.attn.n_layers = mobj["n_layers"].as_int(0);
    desc_.attn.n_heads = mobj["n_heads"].as_int(0);
    desc_.attn.n_kv_heads = mobj["n_kv_heads"].as_int(0);
    desc_.attn.head_dim = mobj["head_dim"].as_int(0);
    desc_.attn.full_attn_interval = mobj["full_attn_interval"].as_int(0);
    desc_.attn.pattern =
        desc_.attn.full_attn_interval > 0 ? model::AttentionPattern::HybridGDN : model::AttentionPattern::Standard;
    if (mobj.contains("rope") && mobj["rope"].is_object()) {
        const auto& robj = mobj["rope"];
        desc_.rope.freq_base = robj["freq_base"].as_double(10000.0);
        desc_.rope.factor = robj["factor"].as_double(1.0);
        desc_.rope.orig_ctx = static_cast<double>(robj["orig_ctx"].as_int(0));
        desc_.rope.type = robj["type"].as_string("none");
    }
    if (mobj.contains("moe") && mobj["moe"].is_object()) {
        const auto& moe = mobj["moe"];
        desc_.moe.n_routed_experts = moe["n_routed_experts"].as_int(0);
        desc_.moe.k_active_experts = moe["k_active_experts"].as_int(0);
        desc_.moe.expert_dim_ff = moe["expert_dim_ff"].as_int(0);
        desc_.moe.n_shared_experts = moe["n_shared_experts"].as_int(0);
        desc_.moe.shared_dim_ff = moe["shared_dim_ff"].as_int(0);
        desc_.moe.expert_blob_bytes = moe["expert_blob_bytes"].as_int(0);
    }

    // Populate runner_model descriptor so configure_geometry & build_execution_plan work without re-querying GGUF
    guild_runner_model_v1_t rm{};
    rm.struct_size = sizeof(rm);
    rm.version = GUILD_RUNNER_MODEL_VERSION_V1;
    std::snprintf(rm.arch_id, sizeof(rm.arch_id), "%s", desc_.arch_name.c_str());
    std::snprintf(rm.name, sizeof(rm.name), "%s", desc_.name.c_str());
    rm.n_embd = desc_.attn.n_embd;
    rm.n_layers = desc_.attn.n_layers;
    rm.n_heads = desc_.attn.n_heads;
    rm.n_kv_heads = desc_.attn.n_kv_heads;
    rm.head_dim = desc_.attn.head_dim;
    rm.context_length = desc_.attn.context_length;
    rm.vocab_size = desc_.attn.vocab_size;
    rm.full_attention_interval = desc_.attn.full_attn_interval;
    rm.n_experts = desc_.moe.n_routed_experts;
    rm.active_experts = desc_.moe.k_active_experts;
    rm.expert_ff = desc_.moe.expert_dim_ff;
    rm.shared_experts = desc_.moe.n_shared_experts;
    rm.shared_ff = desc_.moe.shared_dim_ff;
    rm.ssm_state_size = mobj["ssm_state_size"].as_int(128);
    rm.ssm_k_heads = mobj["ssm_k_heads"].as_int(16);
    rm.ssm_v_heads = mobj["ssm_v_heads"].as_int(32);
    rm.ssm_d_conv = mobj["ssm_d_conv"].as_int(4);
    rm.ssm_conv_channels = mobj["ssm_conv_channels"].as_int(8192);
    rm.ssm_value_dim = mobj["ssm_value_dim"].as_int(4096);
    rm.index_heads = mobj["idx_q_heads"].as_int(0);
    rm.index_dim = mobj["idx_key_dim"].as_int(0);
    rm.residual_streams = mobj["residual_streams"].as_int(1);
    rm.residual_rank = mobj["residual_rank"].as_int(0);
    rm.gdn_gate_silu = mobj["gdn_gate_silu"].as_bool(true) ? 1u : 0u;
    rm.indexed_attention = (rm.index_heads > 0) ? 1u : 0u;
    rm.rope_base = desc_.rope.freq_base;
    rm.rope_factor = desc_.rope.factor;
    rm.rope_original_context = desc_.rope.orig_ctx;
    std::snprintf(rm.embedding_tensor, sizeof(rm.embedding_tensor), "%s", "token_embd.weight");
    std::snprintf(rm.output_tensor, sizeof(rm.output_tensor), "%s", "output.weight");
    std::snprintf(rm.output_norm_tensor, sizeof(rm.output_norm_tensor), "%s", "output_norm.weight");
    std::snprintf(rm.ple_table_tensor, sizeof(rm.ple_table_tensor), "%s", "per_layer_token_embd.weight");
    std::snprintf(rm.ple_key_tensor, sizeof(rm.ple_key_tensor), "%s", "blk.1.ple_key.weight");
    std::snprintf(rm.layer_prefix, sizeof(rm.layer_prefix), "%s", "blk.");

    const auto& eobj = mroot["execution"];
    exec_meta_ = GPackExecutionMetadata{};
    exec_meta_.runner_id = eobj["runner_id"].as_string();
    exec_meta_.arch_id = eobj["arch_id"].as_string();
    exec_meta_.prefill_strategy = eobj["prefill_strategy"].as_string("sequential_verifier");
    exec_meta_.required_core_caps = static_cast<uint32_t>(eobj["required_core_caps"].as_int(0));
    exec_meta_.requires_ple = eobj["requires_ple"].as_bool(false);
    exec_meta_.supports_mtp = eobj["supports_mtp"].as_bool(false);
    if (eobj.contains("required_ops") && eobj["required_ops"].is_array()) {
        for (const auto& op : eobj["required_ops"].arr_val) {
            exec_meta_.required_ops.push_back(static_cast<uint32_t>(op.as_int(0)));
        }
    }
    if (eobj.contains("required_modules") && eobj["required_modules"].is_array()) {
        for (const auto& mod : eobj["required_modules"].arr_val) {
            exec_meta_.required_modules.push_back(mod.as_string());
        }
    }
    if (eobj.contains("eos_token_ids") && eobj["eos_token_ids"].is_array()) {
        for (const auto& id : eobj["eos_token_ids"].arr_val) {
            exec_meta_.eos_token_ids.push_back(id.as_int(0));
        }
    }

    rm.requires_ple = exec_meta_.requires_ple ? 1u : 0u;
    rm.supports_mtp = exec_meta_.supports_mtp ? 1u : 0u;
    if (exec_meta_.prefill_strategy == "layer_wise_batched") {
        rm.prefill_strategy = GUILD_RUNNER_PREFILL_LAYER_BATCHED;
    } else if (exec_meta_.prefill_strategy == "qwen4exp_chunked") {
        rm.prefill_strategy = GUILD_RUNNER_PREFILL_CHUNKED;
    } else {
        rm.prefill_strategy = GUILD_RUNNER_PREFILL_SEQUENTIAL;
    }
    desc_.runner_model = rm;

    // Parse MetadataKvBin
    if (const auto* sec = find_section(GPackSectionKind::MetadataKvBin)) {
        BinaryReader br(base_ + sec->offset, static_cast<size_t>(sec->size));
        uint32_t count = 0;
        if (!br.u32(count) || count != sec->item_count) {
            err = "GPACK open: corrupted MetadataKvBin section";
            close();
            return false;
        }
        for (uint32_t i = 0; i < count; ++i) {
            std::string key;
            uint8_t kind = 0;
            if (!br.str(key) || !br.u8(kind)) {
                err = "GPACK open: truncated MetadataKvBin entry";
                close();
                return false;
            }
            MetaValue mv{};
            if (kind == 1) { // int64
                int64_t iv = 0;
                if (!br.i64(iv)) {
                    close();
                    err = "GPACK open: bad int64 meta";
                    return false;
                }
                mv.type = MetaType::I64;
                mv.u = static_cast<uint64_t>(iv);
            } else if (kind == 2) { // float64
                mv.type = MetaType::F64;
                if (!br.f64(mv.f)) {
                    close();
                    err = "GPACK open: bad float64 meta";
                    return false;
                }
            } else if (kind == 3) { // string
                mv.type = MetaType::STRING;
                if (!br.str(mv.s)) {
                    close();
                    err = "GPACK open: bad string meta";
                    return false;
                }
            } else if (kind == 4) { // bool
                uint8_t b = 0;
                if (!br.u8(b)) {
                    close();
                    err = "GPACK open: bad bool meta";
                    return false;
                }
                mv.type = MetaType::BOOL;
                mv.u = (b != 0) ? 1u : 0u;
            } else {
                err = "GPACK open: unknown metadata value kind";
                close();
                return false;
            }
            metadata_kv_.emplace(std::move(key), std::move(mv));
        }
    }

    // Parse TensorIndexBin
    if (const auto* sec = find_section(GPackSectionKind::TensorIndexBin)) {
        BinaryReader br(base_ + sec->offset, static_cast<size_t>(sec->size));
        uint32_t count = 0;
        if (!br.u32(count) || count != sec->item_count) {
            err = "GPACK open: corrupted TensorIndexBin section";
            close();
            return false;
        }
        tensors_.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            TensorInfo ti{};
            uint32_t n_dims = 0;
            int32_t type = 0;
            uint64_t n_bytes = 0;
            if (!br.str(ti.name) || !br.i32(type) || !br.u32(n_dims) || n_dims == 0 || n_dims > 4) {
                err = "GPACK open: invalid TensorIndexBin entry";
                close();
                return false;
            }
            ti.type = static_cast<uint32_t>(type);
            ti.shape.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                uint64_t dim = 0;
                if (!br.u64(dim) || dim == 0) {
                    err = "GPACK open: invalid tensor dimension in TensorIndexBin";
                    close();
                    return false;
                }
                ti.shape[d] = dim;
            }
            if (!br.u64(ti.offset) || !br.u64(n_bytes)) {
                err = "GPACK open: truncated TensorIndexBin offsets";
                close();
                return false;
            }
            (void) n_bytes;
            if (!tensor_by_name_.emplace(ti.name, i).second) {
                err = "GPACK open: duplicate tensor name in TensorIndexBin: " + ti.name;
                close();
                return false;
            }
            tensors_[i] = std::move(ti);
        }
    }

    // Parse DenseIndexBin
    if (const auto* sec = find_section(GPackSectionKind::DenseIndexBin)) {
        BinaryReader br(base_ + sec->offset, static_cast<size_t>(sec->size));
        uint32_t count = 0;
        if (!br.u64(dense_pool_bytes_) || !br.u64(dense_compacted_pool_bytes_) || !br.u32(count) ||
            count != sec->item_count) {
            err = "GPACK open: corrupted DenseIndexBin header";
            close();
            return false;
        }
        dense_rows_.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& r = dense_rows_[i];
            uint8_t flags = 0;
            if (!br.str(r.name) || !br.i32(r.file_id) || !br.i32(r.kind) || !br.u64(r.src_off) ||
                !br.u64(r.src_bytes) || !br.u64(r.dst_off) || !br.u64(r.dst_bytes) || !br.i64(r.ne0) ||
                !br.i64(r.ne1) || !br.i32(r.code_bits) || !br.i32(r.code_bias) || !br.i32(r.group_elems) ||
                !br.i32(r.codebook) || !br.i32(r.has_offset) || !br.u64(r.codes_bytes) || !br.u64(r.scales_bytes) ||
                !br.u64(r.offset_bytes) || !br.i32(r.scales_fp16) || !br.i32(r.act_kind) || !br.u8(flags)) {
                err = "GPACK open: truncated DenseIndexBin row";
                close();
                return false;
            }
            r.raw16 = (flags & 1u) != 0;
            r.skip_in_native = (flags & 2u) != 0;
            if (r.skip_in_native)
                precomputed_skip_set_.insert(r.name);
        }
    }

    // Parse NativeDenseIndexBin
    if (const auto* sec = find_section(GPackSectionKind::NativeDenseIndexBin)) {
        BinaryReader br(base_ + sec->offset, static_cast<size_t>(sec->size));
        uint32_t count = 0;
        if (!br.u64(native_dense_arena_bytes_) || !br.u32(count) || count != sec->item_count) {
            err = "GPACK open: corrupted NativeDenseIndexBin header";
            close();
            return false;
        }
        native_dense_rows_.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& r = native_dense_rows_[i];
            uint8_t ple_flag = 0;
            if (!br.str(r.name) || !br.i32(r.ggml_type) || !br.i64(r.ne0) || !br.i64(r.ne1) || !br.u64(r.blob_offset) ||
                !br.u64(r.byte_size) || !br.u64(r.arena_offset) || !br.u64(r.aligned_size) || !br.u8(ple_flag)) {
                err = "GPACK open: truncated NativeDenseIndexBin row";
                close();
                return false;
            }
            r.is_ple_key = (ple_flag != 0);
        }
    }

    // Parse ExpertLayoutBin
    if (const auto* sec = find_section(GPackSectionKind::ExpertLayoutBin)) {
        BinaryReader br(base_ + sec->offset, static_cast<size_t>(sec->size));
        uint8_t nat_flag = 0;
        uint32_t count = 0;
        if (!br.u8(nat_flag) || !br.i64(expert_n_expert_) || !br.u64(expert_total_bytes_) ||
            !br.str(external_expert_gguf_) || !br.u32(count) || count != sec->item_count) {
            err = "GPACK open: corrupted ExpertLayoutBin header";
            close();
            return false;
        }
        expert_native_ = (nat_flag != 0);
        expert_layers_.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& r = expert_layers_[i];
            if (!br.i64(r.layer) || !br.i32(r.gu_type) || !br.i32(r.d_type) || !br.u64(r.layer_offset) ||
                !br.u64(r.blob_bytes) || !br.u64(r.gate_off) || !br.u64(r.up_off) || !br.u64(r.down_off) ||
                !br.u64(r.gguf_gate_off) || !br.u64(r.gguf_up_off) || !br.u64(r.gguf_down_off) ||
                !br.str(r.gate_shard) || !br.str(r.up_shard) || !br.str(r.down_shard)) {
                err = "GPACK open: truncated ExpertLayoutBin row";
                close();
                return false;
            }
        }
        if (has_embedded_experts()) {
            const auto* exp_sec = find_section(GPackSectionKind::ExpertBlobs);
            if (!exp_sec || exp_sec->size != expert_total_bytes_) {
                err = "GPACK open: ExpertBlobs section size mismatch against ExpertLayoutBin total_expert_bytes";
                close();
                return false;
            }
        }
    }

    return true;
}

std::string GPackReader::manifest_sha256_hex() const {
    return bytes_to_hex(footer_.manifest_sha256, 32);
}

bool GPackReader::extract_tokenizer_assets(const std::string& dst_dir, std::string& err) const {
    const auto* sec = find_section(GPackSectionKind::TokenizerAssets);
    if (!sec || !base_) {
        err = "GPACK has no TokenizerAssets section";
        return false;
    }
    std::error_code ec;
    fs::create_directories(dst_dir, ec);
    if (ec) {
        err = "cannot create tokenizer output directory: " + dst_dir;
        return false;
    }
    BinaryReader br(base_ + sec->offset, static_cast<size_t>(sec->size));
    uint32_t count = 0;
    if (!br.u32(count) || count != sec->item_count) {
        err = "corrupted TokenizerAssets section";
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        std::string filename;
        uint64_t fbytes = 0;
        if (!br.str(filename, 256) || !br.u64(fbytes) || br.at > br.n || fbytes > br.n - br.at) {
            err = "truncated TokenizerAssets file entry";
            return false;
        }
        if (filename.empty() || filename.find('/') != std::string::npos || filename.find('\\') != std::string::npos ||
            filename.find("..") != std::string::npos) {
            err = "unsafe filename in TokenizerAssets: " + filename;
            return false;
        }
        fs::path out_p = fs::path(dst_dir) / filename;
        std::ofstream out(out_p, std::ios::binary);
        if (!out.is_open() ||
            !out.write(reinterpret_cast<const char*>(br.p + br.at), static_cast<std::streamsize>(fbytes))) {
            err = "failed to write extracted tokenizer file: " + out_p.string();
            return false;
        }
        br.at += static_cast<size_t>(fbytes);
    }
    return true;
}

guild_model_reader_v1_t GPackReader::make_abi_model_reader() const {
    guild_model_reader_v1_t reader{};
    reader.struct_size = sizeof(reader);
    reader.user = const_cast<GPackReader*>(this);
    reader.get_i64 = [](void* u, const char* key, int64_t* out) -> guild_status_t {
        if (!key || !out)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* self = static_cast<const GPackReader*>(u);
        const auto* v = self->get_meta(key);
        if (!v || !v->is_num())
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        *out = static_cast<int64_t>(v->u);
        return GUILD_STATUS_OK;
    };
    reader.get_f64 = [](void* u, const char* key, double* out) -> guild_status_t {
        if (!key || !out)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* self = static_cast<const GPackReader*>(u);
        const auto* v = self->get_meta(key);
        if (!v || !v->is_num())
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        *out = v->num();
        return GUILD_STATUS_OK;
    };
    reader.get_string = [](void* u, const char* key, char* out, size_t cap) -> guild_status_t {
        if (!key || !out || cap == 0)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* self = static_cast<const GPackReader*>(u);
        const auto* v = self->get_meta(key);
        if (!v || v->type != MetaType::STRING || v->s.size() + 1 > cap)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        std::memcpy(out, v->s.c_str(), v->s.size() + 1);
        return GUILD_STATUS_OK;
    };
    reader.get_tensor = [](void* u, const char* name, guild_tensor_desc_t* out) -> guild_status_t {
        if (!name || !out)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        const auto* self = static_cast<const GPackReader*>(u);
        const auto* t = self->find_tensor(name);
        if (!t)
            return GUILD_STATUS_ERR_INVALID_ARGUMENT;
        *out = guild_tensor_desc_t{};
        out->struct_size = sizeof(*out);
        out->dtype = static_cast<guild_dtype_t>(t->type);
        out->n_dims = static_cast<int32_t>(std::min<size_t>(t->shape.size(), GUILD_MAX_TENSOR_DIMS));
        for (int32_t i = 0; i < out->n_dims; ++i)
            out->dims[i] = static_cast<int64_t>(t->shape[static_cast<size_t>(i)]);
        out->buffer.struct_size = sizeof(out->buffer);
        out->buffer.byte_size = tensor_payload_bytes(*t);
        return GUILD_STATUS_OK;
    };
    return reader;
}

bool GPackBuilder::build(const GPackBuildOptions& opts, GPackBuildReport& report, std::string& err) try {
    const auto build_t0 = Clock::now();
    report = GPackBuildReport{};
    err.clear();

    if (opts.primary_input_path.empty()) {
        err = "GPackBuilder: primary_input_path is required";
        return false;
    }
    if (opts.output_path.empty()) {
        err = "GPackBuilder: output_path is required";
        return false;
    }

    const fs::path in_path(opts.primary_input_path);
    if (!fs::exists(in_path)) {
        err = "GPackBuilder: input path does not exist: " + opts.primary_input_path;
        return false;
    }
    if (in_path.extension() == ".safetensors" || in_path.filename() == "model.safetensors.index.json" ||
        (fs::is_directory(in_path) &&
         (fs::exists(in_path / "model.safetensors") || fs::exists(in_path / "model.safetensors.index.json")))) {
        return build_from_safetensors(opts, report, err);
    }

    std::vector<std::string> shards = guild::gguf_split_paths(opts.primary_input_path);
    if (shards.size() == 1 && !opts.additional_shards.empty()) {
        shards.insert(shards.end(), opts.additional_shards.begin(), opts.additional_shards.end());
    }

    const guild::GgufModel gguf_model(shards);
    std::shared_ptr<const model::IModelRunner> runner;
    model::ModelDescriptor desc;
    if (model::ModelRunnerRegistry::instance().resolve_for_gguf(gguf_model.meta(), &runner, err) != GUILD_STATUS_OK ||
        !runner || !runner->validate_and_describe(gguf_model, desc, err)) {
        if (err.empty())
            err = "GPackBuilder: model failed runner validation";
        return false;
    }
    if (!opts.model_name.empty())
        desc.name = opts.model_name;

    core::ModelGeometry geom;
    int64_t k_active = 0;
    if (!runner->configure_geometry(desc, geom, k_active, err)) {
        return false;
    }
    memory::ExecutionPlan dummy_plan;
    dummy_plan.context_length = desc.attn.context_length;
    dummy_plan.kv_format = memory::KvPrecision::FP16;
    dummy_plan.kv_mode = memory::KvMode::HostOnly;
    model::RunnerExecutionPlan runner_exec;
    if (!runner->build_execution_plan(desc, geom, dummy_plan, runner_exec, err)) {
        return false;
    }

    // 1. Serialize MetadataKvBin (excluding huge tokenizer arrays so metadata stays fast & compact)
    BinaryWriter meta_bw;
    uint32_t meta_count = 0;
    for (const auto& [k, v] : gguf_model.meta().metadata()) {
        if (v.type != MetaType::ARRAY) {
            ++meta_count;
        }
    }
    meta_bw.u32(meta_count);
    for (const auto& [k, v] : gguf_model.meta().metadata()) {
        if (v.type == MetaType::ARRAY)
            continue;
        if (v.type == MetaType::STRING) {
            meta_bw.str(k);
            meta_bw.u8(3);
            meta_bw.str(v.s);
        } else if (v.type == MetaType::BOOL) {
            meta_bw.str(k);
            meta_bw.u8(4);
            meta_bw.u8(v.u != 0 ? 1 : 0);
        } else if (v.type == MetaType::F32 || v.type == MetaType::F64) {
            meta_bw.str(k);
            meta_bw.u8(2);
            meta_bw.f64(v.f);
        } else {
            meta_bw.str(k);
            meta_bw.u8(1);
            meta_bw.i64(static_cast<int64_t>(v.u));
        }
    }

    // 2. Collect and deterministically sort all tensors across shards
    struct ShardTensorRef {
        TensorInfo info;
        size_t shard_idx = 0;
        uint64_t n_bytes = 0;
    };
    std::vector<ShardTensorRef> all_tensors;
    for (size_t s = 0; s < gguf_model.size(); ++s) {
        const auto& sf = gguf_model.shard(s);
        for (const auto& t : sf.tensors()) {
            const uint64_t n_bytes = tensor_payload_bytes(t);
            if (!gguf_model.in_bounds(t, s)) {
                err = "GPackBuilder: tensor '" + t.name + "' exceeds shard bounds";
                return false;
            }
            all_tensors.push_back({t, s, n_bytes});
        }
    }
    std::sort(all_tensors.begin(), all_tensors.end(),
              [](const ShardTensorRef& a, const ShardTensorRef& b) { return a.info.name < b.info.name; });

    BinaryWriter tidx_bw;
    tidx_bw.u32(static_cast<uint32_t>(all_tensors.size()));
    for (const auto& st : all_tensors) {
        const auto& t = st.info;
        tidx_bw.str(t.name);
        tidx_bw.i32(static_cast<int32_t>(t.type));
        tidx_bw.u32(static_cast<uint32_t>(t.shape.size()));
        for (uint64_t d : t.shape)
            tidx_bw.u64(d);
        tidx_bw.u64(t.offset);
        tidx_bw.u64(st.n_bytes);
    }
    report.tensor_count = all_tensors.size();

    // 3. Determine native dense projections & dense pack rows
    const bool has_pack_dir = !opts.pack_dir.empty() && fs::exists(fs::path(opts.pack_dir) / "index.txt");
    uint64_t dense_pool_bytes = 0;
    std::vector<GPackDenseRowV1> dense_rows;
    std::vector<uint8_t> synthesized_dense_bin;
    bool native_pack = true;

    if (has_pack_dir) {
        if (!parse_pack_index_file((fs::path(opts.pack_dir) / "index.txt").string(), dense_pool_bytes, dense_rows,
                                   err)) {
            return false;
        }
        native_pack = fs::exists(fs::path(opts.pack_dir) / "native_experts.txt");
    } else {
        // Standalone GGUF import: build dense index and dense.bin directly from non-expert GGUF tensors
        uint64_t cur_src = 0, cur_dst = 0;
        for (const auto& st : all_tensors) {
            const auto& t = st.info;
            const uint64_t t_bytes = st.n_bytes;
            if (t.name.find("_exps.weight") != std::string::npos || t.name == "per_layer_token_embd.weight") {
                continue;
            }
            GPackDenseRowV1 row{};
            row.name = t.name;
            row.file_id = 0;
            row.ne0 = t.shape.size() >= 1 ? static_cast<int64_t>(t.shape[0]) : 1;
            row.ne1 = t.shape.size() >= 2 ? static_cast<int64_t>(t.shape[1]) : 0;
            row.group_elems = 1;
            if (t.type == 0) { // F32
                if (t.name.find("ffn_gate_inp") != std::string::npos || t.name.find("ssm_alpha") != std::string::npos ||
                    t.name.find("ssm_beta") != std::string::npos || t.name.find("hc_") != std::string::npos) {
                    // Convert F32 router/gate/hc weights to BF16 raw16 if not norm
                    if (t.name.find("norm") == std::string::npos) {
                        row.kind = 1; // WeightKind::Bf16InF32
                        row.raw16 = true;
                        row.src_off = cur_src;
                        row.src_bytes = t_bytes / 2;
                        row.dst_off = cur_dst;
                        row.dst_bytes = t_bytes / 2;
                        const size_t n_elem = static_cast<size_t>(t_bytes / 4);
                        const float* fp = reinterpret_cast<const float*>(gguf_model.shard(st.shard_idx).tensor_data(t));
                        const size_t old_sz = synthesized_dense_bin.size();
                        synthesized_dense_bin.resize(old_sz + n_elem * 2);
                        uint16_t* bp = reinterpret_cast<uint16_t*>(synthesized_dense_bin.data() + old_sz);
                        for (size_t i = 0; i < n_elem; ++i) {
                            uint32_t bits = 0;
                            std::memcpy(&bits, &fp[i], 4);
                            uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
                            bp[i] = static_cast<uint16_t>((bits + rounding) >> 16);
                        }
                        const uint64_t padded = align_up(row.src_bytes, 64);
                        synthesized_dense_bin.resize(old_sz + static_cast<size_t>(padded), 0);
                        cur_src += padded;
                        cur_dst += padded;
                        dense_rows.push_back(std::move(row));
                        continue;
                    }
                }
                row.kind = 2; // F32
                row.src_off = cur_src;
                row.src_bytes = t_bytes;
                row.dst_off = cur_dst;
                row.dst_bytes = t_bytes;
                const uint8_t* sp = gguf_model.shard(st.shard_idx).tensor_data(t);
                const size_t old_sz = synthesized_dense_bin.size();
                const uint64_t padded = align_up(t_bytes, 64);
                synthesized_dense_bin.resize(old_sz + static_cast<size_t>(padded), 0);
                std::memcpy(synthesized_dense_bin.data() + old_sz, sp, static_cast<size_t>(t_bytes));
                cur_src += padded;
                cur_dst += padded;
            } else if (t.type == 30 || t.type == 1) { // BF16 or F16
                row.kind = (t.type == 30) ? 1 : 3;
                row.raw16 = true;
                row.src_off = cur_src;
                row.src_bytes = t_bytes;
                row.dst_off = cur_dst;
                row.dst_bytes = t_bytes;
                const uint8_t* sp = gguf_model.shard(st.shard_idx).tensor_data(t);
                const size_t old_sz = synthesized_dense_bin.size();
                const uint64_t padded = align_up(t_bytes, 64);
                synthesized_dense_bin.resize(old_sz + static_cast<size_t>(padded), 0);
                std::memcpy(synthesized_dense_bin.data() + old_sz, sp, static_cast<size_t>(t_bytes));
                cur_src += padded;
                cur_dst += padded;
            } else if (supported_native_mmvq_type(static_cast<int>(t.type), row.ne0, row.ne1)) {
                // Served natively from NativeDense / NativeHead / NativeEmbed
                row.kind = 0;
                row.src_off = cur_src;
                row.src_bytes = 0;
                row.dst_off = cur_dst;
                row.dst_bytes = 0;
                row.code_bits = 8;
                row.group_elems = 32;
            } else {
                err = "GPackBuilder: unsupported standalone tensor quantization type " + std::to_string(t.type) +
                      " for tensor '" + t.name + "'";
                return false;
            }
            dense_rows.push_back(std::move(row));
        }
        dense_pool_bytes = cur_dst;
    }

    // Compute native dense rows and skip set
    std::set<std::string> skip_set;
    std::vector<GPackNativeDenseRowV1> native_dense_rows;
    std::vector<std::pair<size_t, const TensorInfo*>> native_dense_sources;
    uint64_t native_blob_off = 0;
    uint64_t native_arena_off = 0;

    bool keep_ple_key_in_pack = false;
    for (const auto& dr : dense_rows) {
        if (dr.name == "blk.1.ple_key.weight" && dr.code_bits == 0 && dr.src_bytes > 0) {
            keep_ple_key_in_pack = true;
            break;
        }
    }

    for (const auto& st : all_tensors) {
        const auto& t = st.info;
        if (!eligible_native_dense(t, !keep_ple_key_in_pack))
            continue;
        if (t.shape.size() != 2)
            continue;
        const int64_t ne0 = static_cast<int64_t>(t.shape[0]);
        const int64_t ne1 = static_cast<int64_t>(t.shape[1]);
        if (!supported_native_mmvq_type(static_cast<int>(t.type), ne0, ne1))
            continue;
        skip_set.insert(t.name);

        GPackNativeDenseRowV1 nrow{};
        nrow.name = t.name;
        nrow.ggml_type = static_cast<int32_t>(t.type);
        nrow.ne0 = ne0;
        nrow.ne1 = ne1;
        nrow.blob_offset = native_blob_off;
        nrow.byte_size = st.n_bytes;
        nrow.arena_offset = native_arena_off;
        nrow.aligned_size = align_up(st.n_bytes, 256);
        nrow.is_ple_key = (t.name == "blk.1.ple_key.weight");
        native_blob_off += align_up(st.n_bytes, 64);
        native_arena_off += nrow.aligned_size;
        native_dense_rows.push_back(std::move(nrow));
        native_dense_sources.push_back({st.shard_idx, &st.info});
    }

    if (!runner->configure_weight_skip_set(native_pack, true, opts.pack_dir, skip_set, err)) {
        return false;
    }

    uint64_t compacted_pool_bytes = 0;
    for (auto& dr : dense_rows) {
        dr.skip_in_native = (skip_set.count(dr.name) != 0);
        if (!dr.skip_in_native) {
            compacted_pool_bytes += align_up(dr.dst_bytes, 64);
        }
    }

    BinaryWriter didx_bw;
    didx_bw.u64(dense_pool_bytes);
    didx_bw.u64(compacted_pool_bytes);
    didx_bw.u32(static_cast<uint32_t>(dense_rows.size()));
    for (const auto& r : dense_rows) {
        uint8_t flags = (r.raw16 ? 1u : 0u) | (r.skip_in_native ? 2u : 0u);
        didx_bw.str(r.name);
        didx_bw.i32(r.file_id);
        didx_bw.i32(r.kind);
        didx_bw.u64(r.src_off);
        didx_bw.u64(r.src_bytes);
        didx_bw.u64(r.dst_off);
        didx_bw.u64(r.dst_bytes);
        didx_bw.i64(r.ne0);
        didx_bw.i64(r.ne1);
        didx_bw.i32(r.code_bits);
        didx_bw.i32(r.code_bias);
        didx_bw.i32(r.group_elems);
        didx_bw.i32(r.codebook);
        didx_bw.i32(r.has_offset);
        didx_bw.u64(r.codes_bytes);
        didx_bw.u64(r.scales_bytes);
        didx_bw.u64(r.offset_bytes);
        didx_bw.i32(r.scales_fp16);
        didx_bw.i32(r.act_kind);
        didx_bw.u8(flags);
    }

    BinaryWriter ndidx_bw;
    ndidx_bw.u64(native_arena_off);
    ndidx_bw.u32(static_cast<uint32_t>(native_dense_rows.size()));
    for (const auto& r : native_dense_rows) {
        ndidx_bw.str(r.name);
        ndidx_bw.i32(r.ggml_type);
        ndidx_bw.i64(r.ne0);
        ndidx_bw.i64(r.ne1);
        ndidx_bw.u64(r.blob_offset);
        ndidx_bw.u64(r.byte_size);
        ndidx_bw.u64(r.arena_offset);
        ndidx_bw.u64(r.aligned_size);
        ndidx_bw.u8(r.is_ple_key ? 1 : 0);
    }

    // 4. Expert Layout & Expert Shard Slicing
    std::vector<GPackExpertLayerRowV1> expert_layers;
    uint64_t expert_total_bytes = 0;
    const int64_t n_layers = geom.n_layers;
    const int64_t n_expert = geom.n_expert;

    struct ExpertLayerSourcePtrs {
        const uint8_t* gate_ptr = nullptr;
        const uint8_t* up_ptr = nullptr;
        const uint8_t* down_ptr = nullptr;
        uint64_t gate_bytes = 0;
        uint64_t up_bytes = 0;
        uint64_t down_bytes = 0;
    };
    std::vector<ExpertLayerSourcePtrs> expert_src_ptrs(static_cast<size_t>(n_layers));

    for (int64_t l = 0; l < n_layers; ++l) {
        const std::string pfx = "blk." + std::to_string(l) + ".ffn_";
        size_t s_gate = 0, s_up = 0, s_down = 0;
        const auto* t_gate = gguf_model.find(pfx + "gate_exps.weight", &s_gate);
        const auto* t_up = gguf_model.find(pfx + "up_exps.weight", &s_up);
        const auto* t_down = gguf_model.find(pfx + "down_exps.weight", &s_down);
        if (!t_gate || !t_up || !t_down) {
            err = "GPackBuilder: missing routed expert tensor at layer " + std::to_string(l);
            return false;
        }
        if (t_gate->shape.size() != 3 || t_up->shape.size() != 3 || t_down->shape.size() != 3 ||
            static_cast<int64_t>(t_gate->shape[2]) != n_expert || static_cast<int64_t>(t_up->shape[2]) != n_expert ||
            static_cast<int64_t>(t_down->shape[2]) != n_expert) {
            err = "GPackBuilder: invalid routed expert tensor shape at layer " + std::to_string(l);
            return false;
        }
        if (t_gate->type != t_up->type) {
            err = "GPackBuilder: gate/up expert type mismatch at layer " + std::to_string(l);
            return false;
        }
        const uint64_t gate_tot = tensor_payload_bytes(*t_gate);
        const uint64_t up_tot = tensor_payload_bytes(*t_up);
        const uint64_t down_tot = tensor_payload_bytes(*t_down);
        if ((gate_tot % static_cast<uint64_t>(n_expert)) != 0 || (up_tot % static_cast<uint64_t>(n_expert)) != 0 ||
            (down_tot % static_cast<uint64_t>(n_expert)) != 0) {
            err = "GPackBuilder: expert tensor byte size not divisible by n_expert at layer " + std::to_string(l);
            return false;
        }
        const uint64_t g_per = gate_tot / static_cast<uint64_t>(n_expert);
        const uint64_t u_per = up_tot / static_cast<uint64_t>(n_expert);
        const uint64_t d_per = down_tot / static_cast<uint64_t>(n_expert);
        const uint64_t blob_bytes = g_per + u_per + d_per;

        GPackExpertLayerRowV1 erow{};
        erow.layer = l;
        erow.gu_type = static_cast<int32_t>(t_gate->type);
        erow.d_type = static_cast<int32_t>(t_down->type);
        erow.layer_offset = expert_total_bytes;
        erow.blob_bytes = blob_bytes;
        erow.gate_off = 0;
        erow.up_off = g_per;
        erow.down_off = g_per + u_per;
        erow.gguf_gate_off = gguf_model.shard(s_gate).data_start() + t_gate->offset;
        erow.gguf_up_off = gguf_model.shard(s_up).data_start() + t_up->offset;
        erow.gguf_down_off = gguf_model.shard(s_down).data_start() + t_down->offset;
        erow.gate_shard = fs::path(shards[s_gate]).filename().string();
        erow.up_shard = fs::path(shards[s_up]).filename().string();
        erow.down_shard = fs::path(shards[s_down]).filename().string();

        expert_src_ptrs[static_cast<size_t>(l)] = {gguf_model.shard(s_gate).tensor_data(*t_gate),
                                                   gguf_model.shard(s_up).tensor_data(*t_up),
                                                   gguf_model.shard(s_down).tensor_data(*t_down),
                                                   g_per,
                                                   u_per,
                                                   d_per};
        expert_total_bytes += blob_bytes * static_cast<uint64_t>(n_expert);
        expert_layers.push_back(std::move(erow));
    }

    BinaryWriter elay_bw;
    elay_bw.u8(1); // native expert format
    elay_bw.i64(n_expert);
    elay_bw.u64(expert_total_bytes);
    elay_bw.str(opts.embed_experts ? "" : fs::absolute(shards.front()).string());
    elay_bw.u32(static_cast<uint32_t>(expert_layers.size()));
    for (const auto& r : expert_layers) {
        elay_bw.i64(r.layer);
        elay_bw.i32(r.gu_type);
        elay_bw.i32(r.d_type);
        elay_bw.u64(r.layer_offset);
        elay_bw.u64(r.blob_bytes);
        elay_bw.u64(r.gate_off);
        elay_bw.u64(r.up_off);
        elay_bw.u64(r.down_off);
        elay_bw.u64(r.gguf_gate_off);
        elay_bw.u64(r.gguf_up_off);
        elay_bw.u64(r.gguf_down_off);
        elay_bw.str(r.gate_shard);
        elay_bw.str(r.up_shard);
        elay_bw.str(r.down_shard);
    }

    // 5. Find token_embd.weight, output.weight, and optional per_layer_token_embd.weight
    size_t s_emb = 0, s_head = 0, s_ple = 0;
    const auto* t_emb = gguf_model.find(runner->tensor_map().embedding_tensor, &s_emb);
    const auto* t_head = gguf_model.find(runner->tensor_map().output_tensor, &s_head);
    const auto* t_ple = !runner->tensor_map().ple_table_tensor.empty()
                            ? gguf_model.find(runner->tensor_map().ple_table_tensor, &s_ple)
                            : nullptr;
    if (!t_emb || !t_head) {
        err = "GPackBuilder: missing token_embd.weight or output.weight";
        return false;
    }

    // 6. Tokenizer (GBT1 binary + raw assets)
    std::string tok_dir = opts.tokenizer_dir;
    if (tok_dir.empty() && !opts.pack_dir.empty() && fs::is_directory(fs::path(opts.pack_dir) / "tokenizer")) {
        tok_dir = (fs::path(opts.pack_dir) / "tokenizer").string();
    }
    std::vector<uint8_t> tok_bin_buf;
    BinaryWriter tok_assets_bw;
    uint32_t tok_asset_count = 0;
    std::vector<int64_t> eos_ids = {248046, 248044};

    if (!tok_dir.empty() && fs::is_directory(tok_dir)) {
        runtime::Tokenizer tok;
        if (!tok.load(tok_dir, err)) {
            err = "GPackBuilder: tokenizer load failed: " + err;
            return false;
        }
        if (!tok.serialize_binary(tok_bin_buf, err)) {
            err = "GPackBuilder: tokenizer binary serialization failed: " + err;
            return false;
        }
        eos_ids.clear();
        for (int32_t id : tok.eos_token_ids())
            eos_ids.push_back(id);

        const std::vector<std::string> asset_names = {"tokenizer.json", "vocab.json", "merges.txt", "token_type.json",
                                                      "chat_template.jinja"};
        std::vector<std::pair<std::string, std::vector<uint8_t>>> asset_files;
        for (const auto& aname : asset_names) {
            fs::path ap = fs::path(tok_dir) / aname;
            if (fs::is_regular_file(ap)) {
                std::ifstream af(ap, std::ios::binary);
                std::vector<uint8_t> data((std::istreambuf_iterator<char>(af)), std::istreambuf_iterator<char>());
                asset_files.push_back({aname, std::move(data)});
            }
        }
        tok_asset_count = static_cast<uint32_t>(asset_files.size());
        tok_assets_bw.u32(tok_asset_count);
        for (const auto& [aname, data] : asset_files) {
            tok_assets_bw.str(aname);
            tok_assets_bw.u64(data.size());
            if (!data.empty())
                tok_assets_bw.append(data.data(), data.size());
        }
    }

    // 7. Optional MTP draft layer archive
    BinaryWriter mtp_bw;
    uint32_t mtp_file_count = 0;
    if (!opts.mtp_dir.empty() && fs::is_directory(opts.mtp_dir)) {
        const std::vector<std::string> mtp_files = {"dense.txt", "dense.bin", "draft_vocab.bin", "experts.bin"};
        std::vector<std::pair<std::string, fs::path>> found_mtp;
        for (const auto& mf : mtp_files) {
            fs::path mp = fs::path(opts.mtp_dir) / mf;
            if (fs::is_regular_file(mp))
                found_mtp.push_back({mf, mp});
        }
        if (!found_mtp.empty()) {
            mtp_file_count = static_cast<uint32_t>(found_mtp.size());
            mtp_bw.u32(mtp_file_count);
            for (const auto& [mf, mp] : found_mtp) {
                std::ifstream in(mp, std::ios::binary);
                std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                mtp_bw.str(mf);
                mtp_bw.u64(data.size());
                if (!data.empty())
                    mtp_bw.append(data.data(), data.size());
            }
        }
    }

    // 8. Compute dense.bin size
    uint64_t dense_bin_size = 0;
    fs::path dense_bin_path;
    if (has_pack_dir) {
        dense_bin_path = fs::path(opts.pack_dir) / "dense.bin";
        if (!fs::is_regular_file(dense_bin_path)) {
            err = "GPackBuilder: missing dense.bin in " + opts.pack_dir;
            return false;
        }
        dense_bin_size = fs::file_size(dense_bin_path);
    } else {
        dense_bin_size = synthesized_dense_bin.size();
    }

    // 9. Plan section layout & offsets deterministically
    // First build all non-manifest sections to know their exact offsets, sizes, and SHA-256 digests!
    fs::path out_file(opts.output_path);
    std::error_code ec;
    if (out_file.has_parent_path()) {
        fs::create_directories(out_file.parent_path(), ec);
    }
    static std::atomic<uint64_t> tmp_counter{0};
    const fs::path tmp_path = out_file.string() + ".tmp." +
                              std::to_string(
#if defined(_WIN32)
                                  GetCurrentProcessId()
#else
                                  ::getpid()
#endif
                                      ) +
                              "." + std::to_string(++tmp_counter);

    struct TempGuard {
        fs::path p;
        bool committed = false;
        ~TempGuard() {
            if (!committed && !p.empty()) {
                std::error_code e;
                fs::remove(p, e);
            }
        }
    } tmp_guard{tmp_path};

    FILE* fp = std::fopen(tmp_path.string().c_str(), "wb+");
    if (!fp) {
        err = "GPackBuilder: cannot create temporary output file: " + tmp_path.string();
        return false;
    }
    struct FileGuard {
        FILE* f = nullptr;
        ~FileGuard() {
            if (f)
                std::fclose(f);
        }
    } fguard{fp};

    auto write_exact = [&](const void* buf, size_t n) -> bool { return n == 0 || std::fwrite(buf, 1, n, fp) == n; };
    auto pad_to = [&](uint64_t cur, uint64_t target) -> bool {
        if (target <= cur)
            return true;
        std::vector<uint8_t> zeros(static_cast<size_t>(target - cur), 0);
        return write_exact(zeros.data(), zeros.size());
    };

    // Reserve 64 bytes for GPackHeaderV1 at offset 0
    uint64_t cursor = sizeof(GPackHeaderV1);
    GPackHeaderV1 dummy_hdr{};
    if (!write_exact(&dummy_hdr, sizeof(dummy_hdr))) {
        err = "GPackBuilder: failed to write header placeholder";
        return false;
    }

    std::vector<GPackSectionEntryV1> sections;
    auto add_memory_section = [&](GPackSectionKind kind, const char* name, uint64_t align, uint64_t items,
                                  const void* data, uint64_t size) -> bool {
        const uint64_t aligned_off = align_up(cursor, align);
        if (!pad_to(cursor, aligned_off)) {
            err = "GPackBuilder: failed writing section alignment padding";
            return false;
        }
        cursor = aligned_off;
        if (!write_exact(data, static_cast<size_t>(size))) {
            err = std::string("GPackBuilder: failed writing section ") + name;
            return false;
        }
        GPackSectionEntryV1 sec{};
        sec.kind = static_cast<uint32_t>(kind);
        sec.offset = cursor;
        sec.size = size;
        sec.alignment = align;
        sec.item_count = items;
        sha256_bytes(data, static_cast<size_t>(size), sec.sha256);
        std::snprintf(sec.name, sizeof(sec.name), "%s", name);
        sections.push_back(sec);
        cursor += size;
        return true;
    };

    if (!add_memory_section(GPackSectionKind::MetadataKvBin, "metadata_kv", 64, meta_count, meta_bw.buf.data(),
                            meta_bw.buf.size()))
        return false;
    if (!add_memory_section(GPackSectionKind::TensorIndexBin, "tensor_index", 64, all_tensors.size(),
                            tidx_bw.buf.data(), tidx_bw.buf.size()))
        return false;
    if (!add_memory_section(GPackSectionKind::DenseIndexBin, "dense_index", 64, dense_rows.size(), didx_bw.buf.data(),
                            didx_bw.buf.size()))
        return false;

    // Write DensePackBlob (4096-byte aligned)
    {
        const uint64_t align = 4096;
        const uint64_t aligned_off = align_up(cursor, align);
        if (!pad_to(cursor, aligned_off)) {
            err = "GPackBuilder: pad failed";
            return false;
        }
        cursor = aligned_off;
        models::Sha256 hasher;
        if (has_pack_dir) {
            std::ifstream dbin(dense_bin_path, std::ios::binary);
            if (!dbin.is_open()) {
                err = "GPackBuilder: cannot open dense.bin";
                return false;
            }
            std::vector<uint8_t> chunk(8u << 20);
            uint64_t rem = dense_bin_size;
            while (rem > 0) {
                const size_t n = static_cast<size_t>(std::min<uint64_t>(rem, chunk.size()));
                if (!dbin.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(n)) ||
                    !write_exact(chunk.data(), n)) {
                    err = "GPackBuilder: failed streaming dense.bin";
                    return false;
                }
                hasher.update(chunk.data(), n);
                rem -= n;
            }
        } else {
            if (!write_exact(synthesized_dense_bin.data(), synthesized_dense_bin.size())) {
                err = "GPackBuilder: failed writing synthesized dense pack";
                return false;
            }
            if (!synthesized_dense_bin.empty()) {
                hasher.update(synthesized_dense_bin.data(), synthesized_dense_bin.size());
            }
        }
        GPackSectionEntryV1 sec{};
        sec.kind = static_cast<uint32_t>(GPackSectionKind::DensePackBlob);
        sec.offset = cursor;
        sec.size = dense_bin_size;
        sec.alignment = align;
        sec.item_count = dense_rows.size();
        hex_to_bytes(hasher.finalize_hex(), sec.sha256, 32);
        std::snprintf(sec.name, sizeof(sec.name), "%s", "dense_pack");
        sections.push_back(sec);
        cursor += dense_bin_size;
        report.dense_pack_bytes = dense_bin_size;
    }

    // Write NativeDenseIndexBin + NativeDenseBlob
    if (!native_dense_rows.empty()) {
        if (!add_memory_section(GPackSectionKind::NativeDenseIndexBin, "native_dense_index", 64,
                                native_dense_rows.size(), ndidx_bw.buf.data(), ndidx_bw.buf.size()))
            return false;

        const uint64_t align = 4096;
        const uint64_t aligned_off = align_up(cursor, align);
        if (!pad_to(cursor, aligned_off)) {
            err = "GPackBuilder: pad failed";
            return false;
        }
        cursor = aligned_off;
        models::Sha256 hasher;
        uint64_t rel_off = 0;
        std::vector<uint8_t> zero64(64, 0);
        for (size_t i = 0; i < native_dense_rows.size(); ++i) {
            const auto& nrow = native_dense_rows[i];
            const auto [s_idx, t_ptr] = native_dense_sources[i];
            const uint8_t* src_ptr = gguf_model.shard(s_idx).tensor_data(*t_ptr);
            if (!write_exact(src_ptr, static_cast<size_t>(nrow.byte_size))) {
                err = "GPackBuilder: failed writing native dense tensor " + nrow.name;
                return false;
            }
            hasher.update(src_ptr, static_cast<size_t>(nrow.byte_size));
            rel_off += nrow.byte_size;
            const uint64_t next_rel = align_up(rel_off, 64);
            const size_t pad_n = static_cast<size_t>(next_rel - rel_off);
            if (pad_n > 0) {
                if (!write_exact(zero64.data(), pad_n)) {
                    err = "GPackBuilder: failed padding native dense tensor";
                    return false;
                }
                hasher.update(zero64.data(), pad_n);
                rel_off = next_rel;
            }
        }
        GPackSectionEntryV1 sec{};
        sec.kind = static_cast<uint32_t>(GPackSectionKind::NativeDenseBlob);
        sec.offset = cursor;
        sec.size = rel_off;
        sec.alignment = align;
        sec.item_count = native_dense_rows.size();
        hex_to_bytes(hasher.finalize_hex(), sec.sha256, 32);
        std::snprintf(sec.name, sizeof(sec.name), "%s", "native_dense_blob");
        sections.push_back(sec);
        cursor += rel_off;
        report.native_dense_bytes = rel_off;
    }

    // Helper for single-tensor sections (EmbedBlob, HeadBlob, PleBlob)
    auto add_single_tensor_section = [&](GPackSectionKind kind, const char* name, size_t s_idx, const TensorInfo& t,
                                         uint64_t& out_bytes) -> bool {
        const uint64_t align = 4096;
        const uint64_t aligned_off = align_up(cursor, align);
        if (!pad_to(cursor, aligned_off)) {
            err = "GPackBuilder: pad failed";
            return false;
        }
        cursor = aligned_off;

        const uint64_t t_bytes = tensor_payload_bytes(t);
        GPackSingleTensorHeaderV1 thdr{};
        thdr.ggml_type = static_cast<int32_t>(t.type);
        thdr.ne0 = t.shape.size() >= 1 ? static_cast<int64_t>(t.shape[0]) : 1;
        thdr.ne1 = t.shape.size() >= 2 ? static_cast<int64_t>(t.shape[1]) : 1;
        thdr.byte_size = t_bytes;
        thdr.data_offset = sizeof(GPackSingleTensorHeaderV1);

        const uint8_t* tdata = gguf_model.shard(s_idx).tensor_data(t);
        models::Sha256 hasher;
        if (!write_exact(&thdr, sizeof(thdr)) || !write_exact(tdata, static_cast<size_t>(t_bytes))) {
            err = std::string("GPackBuilder: failed writing tensor section ") + name;
            return false;
        }
        hasher.update(&thdr, sizeof(thdr));
        hasher.update(tdata, static_cast<size_t>(t_bytes));

        const uint64_t sec_size = sizeof(thdr) + t_bytes;
        GPackSectionEntryV1 sec{};
        sec.kind = static_cast<uint32_t>(kind);
        sec.offset = cursor;
        sec.size = sec_size;
        sec.alignment = align;
        sec.item_count = 1;
        hex_to_bytes(hasher.finalize_hex(), sec.sha256, 32);
        std::snprintf(sec.name, sizeof(sec.name), "%s", name);
        sections.push_back(sec);
        cursor += sec_size;
        out_bytes = t_bytes;
        return true;
    };

    if (!add_single_tensor_section(GPackSectionKind::EmbedBlob, "token_embd", s_emb, *t_emb, report.embed_bytes)) {
        return false;
    }
    if (!add_single_tensor_section(GPackSectionKind::HeadBlob, "output_head", s_head, *t_head, report.head_bytes)) {
        return false;
    }
    uint64_t ple_bytes = 0;
    if (t_ple != nullptr) {
        if (!add_single_tensor_section(GPackSectionKind::PleBlob, "ple_table", s_ple, *t_ple, ple_bytes)) {
            return false;
        }
    }

    // Write ExpertLayoutBin and optional contiguous ExpertBlobs
    if (!add_memory_section(GPackSectionKind::ExpertLayoutBin, "expert_layout", 64, expert_layers.size(),
                            elay_bw.buf.data(), elay_bw.buf.size()))
        return false;

    if (opts.embed_experts) {
        const uint64_t align = 4096;
        const uint64_t aligned_off = align_up(cursor, align);
        if (!pad_to(cursor, aligned_off)) {
            err = "GPackBuilder: pad failed";
            return false;
        }
        cursor = aligned_off;
        models::Sha256 hasher;
        for (int64_t l = 0; l < n_layers; ++l) {
            const auto& sp = expert_src_ptrs[static_cast<size_t>(l)];
            for (int64_t e = 0; e < n_expert; ++e) {
                const uint8_t* g_slice = sp.gate_ptr + static_cast<uint64_t>(e) * sp.gate_bytes;
                const uint8_t* u_slice = sp.up_ptr + static_cast<uint64_t>(e) * sp.up_bytes;
                const uint8_t* d_slice = sp.down_ptr + static_cast<uint64_t>(e) * sp.down_bytes;
                if (!write_exact(g_slice, static_cast<size_t>(sp.gate_bytes)) ||
                    !write_exact(u_slice, static_cast<size_t>(sp.up_bytes)) ||
                    !write_exact(d_slice, static_cast<size_t>(sp.down_bytes))) {
                    err = "GPackBuilder: failed writing expert blob at layer " + std::to_string(l);
                    return false;
                }
                hasher.update(g_slice, static_cast<size_t>(sp.gate_bytes));
                hasher.update(u_slice, static_cast<size_t>(sp.up_bytes));
                hasher.update(d_slice, static_cast<size_t>(sp.down_bytes));
            }
        }
        GPackSectionEntryV1 sec{};
        sec.kind = static_cast<uint32_t>(GPackSectionKind::ExpertBlobs);
        sec.offset = cursor;
        sec.size = expert_total_bytes;
        sec.alignment = align;
        sec.item_count = static_cast<uint64_t>(n_layers * n_expert);
        hex_to_bytes(hasher.finalize_hex(), sec.sha256, 32);
        std::snprintf(sec.name, sizeof(sec.name), "%s", "expert_blobs");
        sections.push_back(sec);
        cursor += expert_total_bytes;
        report.expert_blob_bytes = expert_total_bytes;
    }

    // Write TokenizerBin & TokenizerAssets
    if (!tok_bin_buf.empty()) {
        if (!add_memory_section(GPackSectionKind::TokenizerBin, "tokenizer_bin", 64, 1, tok_bin_buf.data(),
                                tok_bin_buf.size()))
            return false;
        report.tokenizer_bin_bytes = tok_bin_buf.size();
    }
    if (tok_asset_count > 0) {
        if (!add_memory_section(GPackSectionKind::TokenizerAssets, "tokenizer_assets", 64, tok_asset_count,
                                tok_assets_bw.buf.data(), tok_assets_bw.buf.size()))
            return false;
    }

    // Write MtpBlob
    if (mtp_file_count > 0) {
        if (!add_memory_section(GPackSectionKind::MtpBlob, "mtp_draft", 4096, mtp_file_count, mtp_bw.buf.data(),
                                mtp_bw.buf.size()))
            return false;
    }

    // 10. Build canonical ManifestJson (including all section records)
    JsonValue mroot;
    mroot["schema"] = "guild.gpack.v1";
    mroot["format_version"] = static_cast<int64_t>(1);
    mroot["min_reader_version"] = static_cast<int64_t>(1);

    JsonValue mobj;
    mobj["name"] = desc.name.empty() ? fs::path(opts.primary_input_path).stem().string() : desc.name;
    mobj["architecture"] = desc.arch_name;
    mobj["archetype"] = model::archetype_to_string(desc.archetype);
    mobj["quantization"] = "native";
    mobj["context_length"] = desc.attn.context_length;
    mobj["vocab_size"] = desc.attn.vocab_size;
    mobj["n_embd"] = geom.n_embd;
    mobj["n_layers"] = geom.n_layers;
    mobj["n_heads"] = geom.n_head;
    mobj["n_kv_heads"] = geom.n_head_kv;
    mobj["head_dim"] = geom.head_dim;
    mobj["full_attn_interval"] = geom.qsa_interval;
    mobj["ssm_state_size"] = geom.ssm_state_size;
    mobj["ssm_k_heads"] = geom.ssm_k_heads;
    mobj["ssm_v_heads"] = geom.ssm_v_heads;
    mobj["ssm_d_conv"] = geom.ssm_d_conv;
    mobj["ssm_conv_channels"] = geom.ssm_conv_channels;
    mobj["ssm_value_dim"] = geom.ssm_value_dim;
    mobj["idx_q_heads"] = geom.idx_q_heads;
    mobj["idx_key_dim"] = geom.idx_key_dim;
    mobj["residual_streams"] = geom.hc;
    mobj["residual_rank"] = geom.hc_lr;
    mobj["gdn_gate_silu"] = geom.gdn_gate_silu;

    JsonValue robj;
    robj["freq_base"] = static_cast<double>(desc.rope.freq_base);
    robj["factor"] = static_cast<double>(desc.rope.factor);
    robj["orig_ctx"] = static_cast<int64_t>(desc.rope.orig_ctx);
    robj["type"] = desc.rope.type;
    mobj["rope"] = std::move(robj);

    JsonValue moe;
    moe["n_routed_experts"] = geom.n_expert;
    moe["k_active_experts"] = k_active;
    moe["expert_dim_ff"] = geom.n_ff;
    moe["n_shared_experts"] = desc.moe.n_shared_experts > 0 ? desc.moe.n_shared_experts : 1;
    moe["shared_dim_ff"] = desc.moe.shared_dim_ff > 0 ? desc.moe.shared_dim_ff : geom.n_ff;
    moe["expert_blob_bytes"] = static_cast<int64_t>(expert_layers.empty() ? 0 : expert_layers[0].blob_bytes);
    mobj["moe"] = std::move(moe);
    mroot["model"] = std::move(mobj);

    JsonValue eobj;
    eobj["runner_id"] = runner->runner_id();
    eobj["arch_id"] = runner->arch_id();
    if (runner_exec.prefill_strategy == model::PrefillStrategy::LayerWiseBatched) {
        eobj["prefill_strategy"] = "layer_wise_batched";
    } else if (runner_exec.prefill_strategy == model::PrefillStrategy::DedicatedChunkedGemm) {
        eobj["prefill_strategy"] = "qwen4exp_chunked";
    } else {
        eobj["prefill_strategy"] = "sequential_verifier";
    }
    eobj["required_core_caps"] = static_cast<int64_t>(runner_exec.required_core_caps);
    JsonValue ops_arr(std::vector<JsonValue>{});
    for (auto op : runner_exec.required_ops) {
        ops_arr.arr_val.push_back(JsonValue(static_cast<int64_t>(op)));
    }
    eobj["required_ops"] = std::move(ops_arr);
    eobj["required_modules"] = JsonValue(std::vector<JsonValue>{});
    eobj["requires_ple"] = (runner_exec.requires_ple || t_ple != nullptr);
    eobj["supports_mtp"] = (runner_exec.supports_mtp || mtp_file_count > 0);
    JsonValue eos_arr(std::vector<JsonValue>{});
    for (int64_t id : eos_ids)
        eos_arr.arr_val.push_back(JsonValue(id));
    eobj["eos_token_ids"] = std::move(eos_arr);
    mroot["execution"] = std::move(eobj);

    JsonValue prov;
    prov["source_format"] = has_pack_dir ? "gguf+pack" : "gguf";
    prov["embedded_experts"] = opts.embed_experts;
    JsonValue shards_arr(std::vector<JsonValue>{});
    for (size_t s = 0; s < shards.size(); ++s) {
        JsonValue so;
        so["name"] = fs::path(shards[s]).filename().string();
        so["size_bytes"] = static_cast<int64_t>(gguf_model.shard(s).file_size());
        shards_arr.arr_val.push_back(std::move(so));
    }
    prov["source_shards"] = std::move(shards_arr);
    mroot["provenance"] = std::move(prov);

    JsonValue secs_arr(std::vector<JsonValue>{});
    for (const auto& s : sections) {
        JsonValue so;
        so["name"] = std::string(s.name);
        so["kind"] = static_cast<int64_t>(s.kind);
        so["offset"] = static_cast<int64_t>(s.offset);
        so["size"] = static_cast<int64_t>(s.size);
        so["alignment"] = static_cast<int64_t>(s.alignment);
        so["item_count"] = static_cast<int64_t>(s.item_count);
        so["sha256"] = bytes_to_hex(s.sha256, 32);
        secs_arr.arr_val.push_back(std::move(so));
    }
    mroot["sections"] = std::move(secs_arr);

    const std::string manifest_str = mroot.dump_pretty();
    const uint64_t manifest_offset = align_up(cursor, 64);
    if (!add_memory_section(GPackSectionKind::ManifestJson, "manifest", 64, 1, manifest_str.data(),
                            manifest_str.size()))
        return false;
    const auto& manifest_sec = sections.back();
    if (manifest_sec.offset != manifest_offset) {
        err = "GPackBuilder: internal manifest offset mismatch";
        return false;
    }

    // 11. Write Section Table (64-byte aligned) and Footer
    const uint64_t section_table_offset = align_up(cursor, 64);
    if (!pad_to(cursor, section_table_offset)) {
        err = "GPackBuilder: pad failed";
        return false;
    }
    cursor = section_table_offset;
    const uint64_t table_bytes = sections.size() * sizeof(GPackSectionEntryV1);
    if (!write_exact(sections.data(), static_cast<size_t>(table_bytes))) {
        err = "GPackBuilder: failed writing section table";
        return false;
    }
    cursor += table_bytes;

    const uint64_t footer_offset = align_up(cursor, 64);
    if (!pad_to(cursor, footer_offset)) {
        err = "GPackBuilder: pad failed";
        return false;
    }
    cursor = footer_offset;

    GPackFooterV1 footer{};
    footer.manifest_offset = manifest_sec.offset;
    footer.section_table_offset = section_table_offset;
    footer.total_payload_bytes = cursor;
    std::memcpy(footer.manifest_sha256, manifest_sec.sha256, 32);
    footer.section_count = static_cast<uint32_t>(sections.size());
    footer.magic = kGPackFooterMagic;
    if (!write_exact(&footer, sizeof(footer))) {
        err = "GPackBuilder: failed writing footer";
        return false;
    }
    cursor += sizeof(footer);

    // 12. Write finalized GPackHeaderV1 at offset 0
    GPackHeaderV1 header{};
    header.magic = kGPackMagic;
    header.format_version = kGPackVersionV1;
    header.min_reader_version = kGPackVersionV1;
    header.flags = GUILD_GPACK_FLAG_LITTLE_ENDIAN | GUILD_GPACK_FLAG_HAS_DENSE_PACK;
    if (!native_dense_rows.empty())
        header.flags |= GUILD_GPACK_FLAG_HAS_NATIVE_DENSE;
    if (opts.embed_experts)
        header.flags |= GUILD_GPACK_FLAG_HAS_EXPERT_BLOBS;
    else
        header.flags |= GUILD_GPACK_FLAG_EXTERNAL_EXPERTS;
    if (!tok_bin_buf.empty())
        header.flags |= GUILD_GPACK_FLAG_HAS_TOKENIZER_BIN;
    if (mtp_file_count > 0)
        header.flags |= GUILD_GPACK_FLAG_HAS_MTP;
    if (t_ple != nullptr)
        header.flags |= GUILD_GPACK_FLAG_HAS_PLE;
    header.section_count = static_cast<uint32_t>(sections.size());
    header.manifest_offset = manifest_sec.offset;
    header.manifest_size = manifest_sec.size;
    header.section_table_offset = section_table_offset;
    header.total_file_size = cursor;
    std::memcpy(header.header_manifest_sha256_prefix, manifest_sec.sha256, 16);

    if (std::fseek(fp, 0, SEEK_SET) != 0 || !write_exact(&header, sizeof(header)) || std::fflush(fp) != 0) {
        err = "GPackBuilder: failed finalizing header";
        return false;
    }
#if defined(_WIN32)
    _commit(_fileno(fp));
#else
    ::fsync(::fileno(fp));
#endif
    std::fclose(fp);
    fguard.f = nullptr;

    report.build_ms = std::chrono::duration<double, std::milli>(Clock::now() - build_t0).count();

    if (opts.verify_after_write) {
        const auto ver_t0 = Clock::now();
        GPackReader verifier;
        if (!verifier.open(tmp_path.string(), err, false)) {
            err = "GPackBuilder post-write verification failed: " + err;
            return false;
        }
        verifier.close();
        report.verify_ms = std::chrono::duration<double, std::milli>(Clock::now() - ver_t0).count();
    }

    // Atomic rename to output_path
    fs::rename(tmp_path, out_file, ec);
    if (ec) {
        err = "GPackBuilder: atomic rename failed: " + ec.message();
        return false;
    }
    tmp_guard.committed = true;

#if !defined(_WIN32)
    if (out_file.has_parent_path()) {
        int dir_fd = ::open(out_file.parent_path().c_str(), O_RDONLY);
        if (dir_fd >= 0) {
            (void) ::fsync(dir_fd);
            ::close(dir_fd);
        }
    }
#endif

    report.output_path = out_file.string();
    report.manifest_sha256 = bytes_to_hex(manifest_sec.sha256, 32);
    report.total_file_size = cursor;
    report.section_count = sections.size();
    return true;
} catch (const std::exception& e) {
    err = std::string("GPackBuilder: ") + e.what();
    return false;
}

bool GPackBuilder::build_from_safetensors(const GPackBuildOptions& opts, GPackBuildReport& report,
                                          std::string& err) try {
    const auto build_t0 = Clock::now();
    report = GPackBuildReport{};
    err.clear();

    const fs::path in_path(opts.primary_input_path);
    fs::path base_dir = fs::is_directory(in_path) ? in_path : in_path.parent_path();
    std::vector<fs::path> st_files;

    if (fs::is_directory(in_path)) {
        if (fs::exists(in_path / "model.safetensors.index.json")) {
            std::ifstream idx_f(in_path / "model.safetensors.index.json");
            std::string idx_str((std::istreambuf_iterator<char>(idx_f)), std::istreambuf_iterator<char>());
            JsonValue idx_root;
            std::string jerr;
            if (!JsonValue::parse(idx_str, idx_root, jerr) || !idx_root.is_object() ||
                !idx_root.contains("weight_map") || !idx_root["weight_map"].is_object()) {
                err = "Safetensors import: malformed model.safetensors.index.json";
                return false;
            }
            std::set<std::string> shard_names;
            for (const auto& [tname, sfile] : idx_root["weight_map"].obj_val) {
                shard_names.insert(sfile.as_string());
            }
            for (const auto& sn : shard_names) {
                fs::path sp = in_path / sn;
                if (!fs::is_regular_file(sp)) {
                    err = "Safetensors import: missing shard " + sp.string();
                    return false;
                }
                st_files.push_back(sp);
            }
        } else if (fs::is_regular_file(in_path / "model.safetensors")) {
            st_files.push_back(in_path / "model.safetensors");
        } else {
            err = "Safetensors import: no model.safetensors found in " + in_path.string();
            return false;
        }
    } else if (in_path.filename() == "model.safetensors.index.json") {
        std::ifstream idx_f(in_path);
        std::string idx_str((std::istreambuf_iterator<char>(idx_f)), std::istreambuf_iterator<char>());
        JsonValue idx_root;
        std::string jerr;
        if (!JsonValue::parse(idx_str, idx_root, jerr) || !idx_root.is_object() || !idx_root.contains("weight_map") ||
            !idx_root["weight_map"].is_object()) {
            err = "Safetensors import: malformed model.safetensors.index.json";
            return false;
        }
        std::set<std::string> shard_names;
        for (const auto& [tname, sfile] : idx_root["weight_map"].obj_val) {
            shard_names.insert(sfile.as_string());
        }
        for (const auto& sn : shard_names) {
            fs::path sp = base_dir / sn;
            if (!fs::is_regular_file(sp)) {
                err = "Safetensors import: missing shard " + sp.string();
                return false;
            }
            st_files.push_back(sp);
        }
    } else {
        st_files.push_back(in_path);
        for (const auto& add : opts.additional_shards)
            st_files.push_back(add);
    }

    if (st_files.empty()) {
        err = "Safetensors import: no safetensors files found";
        return false;
    }

    // Parse config.json in base_dir for model hyperparameters
    fs::path cfg_path = base_dir / "config.json";
    if (!fs::is_regular_file(cfg_path)) {
        err = "Safetensors import: missing config.json alongside safetensors in " + base_dir.string();
        return false;
    }
    std::ifstream cf(cfg_path);
    std::string cfg_str((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
    JsonValue cfg_mut;
    std::string jerr;
    if (!JsonValue::parse(cfg_str, cfg_mut, jerr) || !cfg_mut.is_object()) {
        err = "Safetensors import: malformed config.json: " + jerr;
        return false;
    }
    const JsonValue& cfg = cfg_mut;

    const std::string arch_name = cfg["architecture"].as_string(cfg["model_type"].as_string("qwen3_5_moe"));
    const int64_t n_embd = cfg["hidden_size"].as_int(cfg["n_embd"].as_int(0));
    const int64_t n_layers = cfg["num_hidden_layers"].as_int(cfg["n_layers"].as_int(0));
    const int64_t n_heads = cfg["num_attention_heads"].as_int(cfg["n_heads"].as_int(0));
    const int64_t n_kv_heads = cfg["num_key_value_heads"].as_int(cfg["n_kv_heads"].as_int(n_heads));
    const int64_t head_dim = cfg["head_dim"].as_int(n_heads > 0 ? n_embd / n_heads : 0);
    const int64_t vocab_size = cfg["vocab_size"].as_int(0);
    const int64_t context_length = cfg["max_position_embeddings"].as_int(cfg["context_length"].as_int(32768));
    const int64_t n_expert = cfg["num_experts"].as_int(cfg["n_routed_experts"].as_int(0));
    const int64_t k_active = cfg["num_experts_per_tok"].as_int(cfg["k_active_experts"].as_int(0));
    const int64_t expert_ff = cfg["moe_intermediate_size"].as_int(cfg["expert_dim_ff"].as_int(0));
    const int64_t full_attn_interval = cfg["full_attention_interval"].as_int(4);

    if (n_embd <= 0 || n_layers <= 0 || n_heads <= 0 || vocab_size <= 0 || n_expert <= 0 || expert_ff <= 0) {
        err = "Safetensors import: incomplete or unsupported model dimensions in config.json";
        return false;
    }

    struct SafetensorSlice {
        std::string name;
        uint32_t ggml_type = 0;
        std::vector<uint64_t> ne;
        std::vector<uint8_t> data;
    };
    std::map<std::string, SafetensorSlice> parsed_tensors;

    for (const auto& st_path : st_files) {
        std::ifstream sf(st_path, std::ios::binary | std::ios::ate);
        if (!sf.is_open()) {
            err = "Safetensors import: cannot open " + st_path.string();
            return false;
        }
        const uint64_t fsize = static_cast<uint64_t>(sf.tellg());
        if (fsize < 8) {
            err = "Safetensors import: truncated header in " + st_path.string();
            return false;
        }
        sf.seekg(0);
        uint64_t hdr_len = 0;
        sf.read(reinterpret_cast<char*>(&hdr_len), 8);
        if (hdr_len == 0 || hdr_len > (64ull << 20) || 8 + hdr_len > fsize) {
            err = "Safetensors import: invalid header length in " + st_path.string();
            return false;
        }
        std::string hdr_json(static_cast<size_t>(hdr_len), '\0');
        sf.read(hdr_json.data(), static_cast<std::streamsize>(hdr_len));
        JsonValue hroot;
        if (!JsonValue::parse(hdr_json, hroot, jerr) || !hroot.is_object()) {
            err = "Safetensors import: malformed JSON header in " + st_path.string();
            return false;
        }
        const uint64_t data_base = 8 + hdr_len;
        for (const auto& [tname, tval] : hroot.obj_val) {
            if (tname == "__metadata__")
                continue;
            if (!tval.is_object() || !tval.contains("dtype") || !tval.contains("shape") ||
                !tval.contains("data_offsets")) {
                err = "Safetensors import: invalid tensor descriptor for '" + tname + "'";
                return false;
            }
            const std::string dtype = tval["dtype"].as_string();
            uint32_t gtype = 0;
            uint64_t elem_bytes = 0;
            if (dtype == "F32") {
                gtype = 0;
                elem_bytes = 4;
            } else if (dtype == "F16") {
                gtype = 1;
                elem_bytes = 2;
            } else if (dtype == "BF16") {
                gtype = 30;
                elem_bytes = 2;
            } else {
                err = "Safetensors import: unsupported safetensors dtype '" + dtype + "' for tensor '" + tname + "'";
                return false;
            }

            if (!tval["shape"].is_array() || tval["shape"].arr_val.empty() || tval["shape"].arr_val.size() > 4) {
                err = "Safetensors import: unsupported rank for tensor '" + tname + "'";
                return false;
            }
            // Safetensors shape is row-major [d0, d1, ...]; GGML ne is column-major [ne0=last_dim, ne1=...]
            std::vector<uint64_t> ne;
            uint64_t total_elems = 1;
            const auto& sh = tval["shape"].arr_val;
            for (size_t i = sh.size(); i > 0; --i) {
                int64_t dim = sh[i - 1].as_int(0);
                if (dim <= 0) {
                    err = "Safetensors import: non-positive dimension in tensor '" + tname + "'";
                    return false;
                }
                ne.push_back(static_cast<uint64_t>(dim));
                total_elems *= static_cast<uint64_t>(dim);
            }
            if (!tval["data_offsets"].is_array() || tval["data_offsets"].arr_val.size() != 2) {
                err = "Safetensors import: invalid data_offsets for '" + tname + "'";
                return false;
            }
            const int64_t off0_i = tval["data_offsets"].arr_val[0].as_int(-1);
            const int64_t off1_i = tval["data_offsets"].arr_val[1].as_int(-1);
            if (off0_i < 0 || off1_i < off0_i) {
                err = "Safetensors import: negative or inverted data_offsets for '" + tname + "'";
                return false;
            }
            const uint64_t off0 = static_cast<uint64_t>(off0_i);
            const uint64_t off1 = static_cast<uint64_t>(off1_i);
            const uint64_t byte_len = off1 - off0;
            if (byte_len != total_elems * elem_bytes || off1 > fsize - data_base) {
                err = "Safetensors import: tensor '" + tname + "' data_offsets out of bounds or size mismatch";
                return false;
            }
            if (parsed_tensors.count(tname)) {
                err = "Safetensors import: duplicate tensor '" + tname + "' across shards";
                return false;
            }
            SafetensorSlice slice;
            slice.name = tname;
            slice.ggml_type = gtype;
            slice.ne = std::move(ne);
            slice.data.resize(static_cast<size_t>(byte_len));
            sf.seekg(static_cast<std::streamoff>(data_base + off0));
            if (byte_len > 0 &&
                !sf.read(reinterpret_cast<char*>(slice.data.data()), static_cast<std::streamsize>(byte_len))) {
                err = "Safetensors import: failed reading tensor '" + tname + "'";
                return false;
            }
            parsed_tensors.emplace(tname, std::move(slice));
        }
    }

    if (!parsed_tensors.count("token_embd.weight") || !parsed_tensors.count("output.weight")) {
        err = "Safetensors import: missing required canonical tensors token_embd.weight or output.weight";
        return false;
    }

    // Build MetadataKvBin & TensorIndexBin & DenseIndexBin & ExpertLayoutBin
    BinaryWriter meta_bw;
    meta_bw.u32(7);
    auto write_kv_str = [&](const std::string& k, const std::string& v) {
        meta_bw.str(k);
        meta_bw.u8(3);
        meta_bw.str(v);
    };
    auto write_kv_i64 = [&](const std::string& k, int64_t v) {
        meta_bw.str(k);
        meta_bw.u8(1);
        meta_bw.i64(v);
    };
    write_kv_str("general.architecture", arch_name);
    write_kv_str("general.name", opts.model_name.empty() ? "safetensors-model" : opts.model_name);
    write_kv_i64(arch_name + ".embedding_length", n_embd);
    write_kv_i64(arch_name + ".block_count", n_layers);
    write_kv_i64(arch_name + ".attention.head_count", n_heads);
    write_kv_i64(arch_name + ".attention.head_count_kv", n_kv_heads);
    write_kv_i64(arch_name + ".expert_count", n_expert);

    BinaryWriter tidx_bw;
    tidx_bw.u32(static_cast<uint32_t>(parsed_tensors.size()));
    uint64_t synthetic_off = 0;
    for (const auto& [tname, sl] : parsed_tensors) {
        tidx_bw.str(tname);
        tidx_bw.i32(static_cast<int32_t>(sl.ggml_type));
        tidx_bw.u32(static_cast<uint32_t>(sl.ne.size()));
        for (uint64_t d : sl.ne)
            tidx_bw.u64(d);
        tidx_bw.u64(synthetic_off);
        tidx_bw.u64(sl.data.size());
        synthetic_off += align_up(sl.data.size(), 64);
    }

    std::vector<GPackDenseRowV1> dense_rows;
    std::vector<uint8_t> dense_blob;
    uint64_t cur_off = 0;
    for (const auto& [tname, sl] : parsed_tensors) {
        if (tname.find("_exps.weight") != std::string::npos)
            continue;
        GPackDenseRowV1 r{};
        r.name = tname;
        r.kind = (sl.ggml_type == 0) ? 2 : 0;
        r.raw16 = (sl.ggml_type == 30);
        r.ne0 = static_cast<int64_t>(sl.ne[0]);
        r.ne1 = sl.ne.size() >= 2 ? static_cast<int64_t>(sl.ne[1]) : 0;
        r.group_elems = 1;
        r.src_off = cur_off;
        r.src_bytes = sl.data.size();
        r.dst_off = cur_off;
        r.dst_bytes = sl.data.size();
        r.skip_in_native = (tname == "token_embd.weight" || tname == "output.weight");
        const uint64_t padded = align_up(sl.data.size(), 64);
        const size_t old_sz = dense_blob.size();
        dense_blob.resize(old_sz + static_cast<size_t>(padded), 0);
        if (!sl.data.empty())
            std::memcpy(dense_blob.data() + old_sz, sl.data.data(), sl.data.size());
        cur_off += padded;
        dense_rows.push_back(std::move(r));
    }

    BinaryWriter didx_bw;
    didx_bw.u64(cur_off);
    didx_bw.u64(cur_off);
    didx_bw.u32(static_cast<uint32_t>(dense_rows.size()));
    for (const auto& r : dense_rows) {
        uint8_t flags = (r.raw16 ? 1u : 0u) | (r.skip_in_native ? 2u : 0u);
        didx_bw.str(r.name);
        didx_bw.i32(r.file_id);
        didx_bw.i32(r.kind);
        didx_bw.u64(r.src_off);
        didx_bw.u64(r.src_bytes);
        didx_bw.u64(r.dst_off);
        didx_bw.u64(r.dst_bytes);
        didx_bw.i64(r.ne0);
        didx_bw.i64(r.ne1);
        didx_bw.i32(r.code_bits);
        didx_bw.i32(r.code_bias);
        didx_bw.i32(r.group_elems);
        didx_bw.i32(r.codebook);
        didx_bw.i32(r.has_offset);
        didx_bw.u64(r.codes_bytes);
        didx_bw.u64(r.scales_bytes);
        didx_bw.u64(r.offset_bytes);
        didx_bw.i32(r.scales_fp16);
        didx_bw.i32(r.act_kind);
        didx_bw.u8(flags);
    }

    std::vector<GPackExpertLayerRowV1> exp_layers;
    std::vector<uint8_t> exp_blob;
    uint64_t exp_total = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        const std::string pfx = "blk." + std::to_string(l) + ".ffn_";
        auto it_g = parsed_tensors.find(pfx + "gate_exps.weight");
        auto it_u = parsed_tensors.find(pfx + "up_exps.weight");
        auto it_d = parsed_tensors.find(pfx + "down_exps.weight");
        if (it_g == parsed_tensors.end() || it_u == parsed_tensors.end() || it_d == parsed_tensors.end()) {
            err = "Safetensors import: missing expert tensors for layer " + std::to_string(l);
            return false;
        }
        const uint64_t g_per = it_g->second.data.size() / static_cast<uint64_t>(n_expert);
        const uint64_t u_per = it_u->second.data.size() / static_cast<uint64_t>(n_expert);
        const uint64_t d_per = it_d->second.data.size() / static_cast<uint64_t>(n_expert);
        const uint64_t bbytes = g_per + u_per + d_per;

        GPackExpertLayerRowV1 er{};
        er.layer = l;
        er.gu_type = static_cast<int32_t>(it_g->second.ggml_type);
        er.d_type = static_cast<int32_t>(it_d->second.ggml_type);
        er.layer_offset = exp_total;
        er.blob_bytes = bbytes;
        er.gate_off = 0;
        er.up_off = g_per;
        er.down_off = g_per + u_per;
        exp_layers.push_back(er);

        for (int64_t e = 0; e < n_expert; ++e) {
            const uint8_t* gp = it_g->second.data.data() + static_cast<size_t>(e * g_per);
            const uint8_t* up = it_u->second.data.data() + static_cast<size_t>(e * u_per);
            const uint8_t* dp = it_d->second.data.data() + static_cast<size_t>(e * d_per);
            exp_blob.insert(exp_blob.end(), gp, gp + g_per);
            exp_blob.insert(exp_blob.end(), up, up + u_per);
            exp_blob.insert(exp_blob.end(), dp, dp + d_per);
        }
        exp_total += bbytes * static_cast<uint64_t>(n_expert);
    }

    BinaryWriter elay_bw;
    elay_bw.u8(1);
    elay_bw.i64(n_expert);
    elay_bw.u64(exp_total);
    elay_bw.str("");
    elay_bw.u32(static_cast<uint32_t>(exp_layers.size()));
    for (const auto& r : exp_layers) {
        elay_bw.i64(r.layer);
        elay_bw.i32(r.gu_type);
        elay_bw.i32(r.d_type);
        elay_bw.u64(r.layer_offset);
        elay_bw.u64(r.blob_bytes);
        elay_bw.u64(r.gate_off);
        elay_bw.u64(r.up_off);
        elay_bw.u64(r.down_off);
        elay_bw.u64(0);
        elay_bw.u64(0);
        elay_bw.u64(0);
        elay_bw.str("");
        elay_bw.str("");
        elay_bw.str("");
    }

    // Write container atomically
    fs::path out_file(opts.output_path);
    std::error_code ec;
    if (out_file.has_parent_path())
        fs::create_directories(out_file.parent_path(), ec);
    const fs::path tmp_path = out_file.string() + ".tmp.safetensors";
    FILE* fp = std::fopen(tmp_path.string().c_str(), "wb+");
    if (!fp) {
        err = "Safetensors import: cannot open temp file";
        return false;
    }

    auto write_exact = [&](const void* buf, size_t n) -> bool { return n == 0 || std::fwrite(buf, 1, n, fp) == n; };
    auto pad_to = [&](uint64_t cur, uint64_t target) -> bool {
        if (target <= cur)
            return true;
        std::vector<uint8_t> zeros(static_cast<size_t>(target - cur), 0);
        return write_exact(zeros.data(), zeros.size());
    };

    uint64_t cursor = sizeof(GPackHeaderV1);
    GPackHeaderV1 dummy_hdr{};
    write_exact(&dummy_hdr, sizeof(dummy_hdr));

    std::vector<GPackSectionEntryV1> sections;
    auto add_mem = [&](GPackSectionKind kind, const char* name, uint64_t align, uint64_t items, const void* data,
                       uint64_t size) -> bool {
        const uint64_t aligned_off = align_up(cursor, align);
        if (!pad_to(cursor, aligned_off))
            return false;
        cursor = aligned_off;
        if (!write_exact(data, static_cast<size_t>(size)))
            return false;
        GPackSectionEntryV1 sec{};
        sec.kind = static_cast<uint32_t>(kind);
        sec.offset = cursor;
        sec.size = size;
        sec.alignment = align;
        sec.item_count = items;
        sha256_bytes(data, static_cast<size_t>(size), sec.sha256);
        std::snprintf(sec.name, sizeof(sec.name), "%s", name);
        sections.push_back(sec);
        cursor += size;
        return true;
    };

    add_mem(GPackSectionKind::MetadataKvBin, "metadata_kv", 64, 7, meta_bw.buf.data(), meta_bw.buf.size());
    add_mem(GPackSectionKind::TensorIndexBin, "tensor_index", 64, parsed_tensors.size(), tidx_bw.buf.data(),
            tidx_bw.buf.size());
    add_mem(GPackSectionKind::DenseIndexBin, "dense_index", 64, dense_rows.size(), didx_bw.buf.data(),
            didx_bw.buf.size());
    add_mem(GPackSectionKind::DensePackBlob, "dense_pack", 4096, dense_rows.size(), dense_blob.data(),
            dense_blob.size());

    auto write_single = [&](GPackSectionKind kind, const char* name, const SafetensorSlice& sl) {
        std::vector<uint8_t> buf(sizeof(GPackSingleTensorHeaderV1) + sl.data.size(), 0);
        GPackSingleTensorHeaderV1 th{};
        th.ggml_type = static_cast<int32_t>(sl.ggml_type);
        th.ne0 = static_cast<int64_t>(sl.ne[0]);
        th.ne1 = sl.ne.size() >= 2 ? static_cast<int64_t>(sl.ne[1]) : 1;
        th.byte_size = sl.data.size();
        th.data_offset = sizeof(th);
        std::memcpy(buf.data(), &th, sizeof(th));
        if (!sl.data.empty())
            std::memcpy(buf.data() + sizeof(th), sl.data.data(), sl.data.size());
        add_mem(kind, name, 4096, 1, buf.data(), buf.size());
    };
    write_single(GPackSectionKind::EmbedBlob, "token_embd", parsed_tensors.at("token_embd.weight"));
    write_single(GPackSectionKind::HeadBlob, "output_head", parsed_tensors.at("output.weight"));
    add_mem(GPackSectionKind::ExpertLayoutBin, "expert_layout", 64, exp_layers.size(), elay_bw.buf.data(),
            elay_bw.buf.size());
    add_mem(GPackSectionKind::ExpertBlobs, "expert_blobs", 4096, static_cast<uint64_t>(n_layers * n_expert),
            exp_blob.data(), exp_blob.size());

    JsonValue mroot;
    mroot["schema"] = "guild.gpack.v1";
    mroot["format_version"] = static_cast<int64_t>(1);
    mroot["min_reader_version"] = static_cast<int64_t>(1);
    JsonValue mobj;
    mobj["name"] = opts.model_name.empty() ? "safetensors-model" : opts.model_name;
    mobj["architecture"] = arch_name;
    mobj["archetype"] = "Qwen3_5_MoE";
    mobj["quantization"] = "safetensors";
    mobj["context_length"] = context_length;
    mobj["vocab_size"] = vocab_size;
    mobj["n_embd"] = n_embd;
    mobj["n_layers"] = n_layers;
    mobj["n_heads"] = n_heads;
    mobj["n_kv_heads"] = n_kv_heads;
    mobj["head_dim"] = head_dim;
    mobj["full_attn_interval"] = full_attn_interval;
    JsonValue robj;
    robj["freq_base"] = 10000.0;
    robj["factor"] = 1.0;
    robj["orig_ctx"] = context_length;
    robj["type"] = "none";
    mobj["rope"] = std::move(robj);
    JsonValue moe;
    moe["n_routed_experts"] = n_expert;
    moe["k_active_experts"] = k_active > 0 ? k_active : 8;
    moe["expert_dim_ff"] = expert_ff;
    moe["n_shared_experts"] = static_cast<int64_t>(1);
    moe["shared_dim_ff"] = expert_ff;
    moe["expert_blob_bytes"] = static_cast<int64_t>(exp_layers.empty() ? 0 : exp_layers[0].blob_bytes);
    mobj["moe"] = std::move(moe);
    mroot["model"] = std::move(mobj);

    JsonValue eobj;
    eobj["runner_id"] = "guild.runner.qwen3_5_moe.v1";
    eobj["arch_id"] = arch_name;
    eobj["prefill_strategy"] = "layer_wise_batched";
    eobj["required_core_caps"] = static_cast<int64_t>(0);
    eobj["required_ops"] = JsonValue(std::vector<JsonValue>{});
    eobj["required_modules"] = JsonValue(std::vector<JsonValue>{});
    eobj["requires_ple"] = false;
    eobj["supports_mtp"] = false;
    eobj["eos_token_ids"] = JsonValue(std::vector<JsonValue>{JsonValue(static_cast<int64_t>(248046))});
    mroot["execution"] = std::move(eobj);

    JsonValue prov;
    prov["source_format"] = "safetensors";
    prov["embedded_experts"] = true;
    JsonValue sarr(std::vector<JsonValue>{});
    for (const auto& sp : st_files) {
        JsonValue so;
        so["name"] = sp.filename().string();
        so["size_bytes"] = static_cast<int64_t>(fs::file_size(sp));
        sarr.arr_val.push_back(std::move(so));
    }
    prov["source_shards"] = std::move(sarr);
    mroot["provenance"] = std::move(prov);

    JsonValue secs_arr(std::vector<JsonValue>{});
    for (const auto& s : sections) {
        JsonValue so;
        so["name"] = std::string(s.name);
        so["kind"] = static_cast<int64_t>(s.kind);
        so["offset"] = static_cast<int64_t>(s.offset);
        so["size"] = static_cast<int64_t>(s.size);
        so["alignment"] = static_cast<int64_t>(s.alignment);
        so["item_count"] = static_cast<int64_t>(s.item_count);
        so["sha256"] = bytes_to_hex(s.sha256, 32);
        secs_arr.arr_val.push_back(std::move(so));
    }
    mroot["sections"] = std::move(secs_arr);

    const std::string manifest_str = mroot.dump_pretty();
    add_mem(GPackSectionKind::ManifestJson, "manifest", 64, 1, manifest_str.data(), manifest_str.size());
    const auto& manifest_sec = sections.back();

    const uint64_t section_table_offset = align_up(cursor, 64);
    pad_to(cursor, section_table_offset);
    cursor = section_table_offset;
    write_exact(sections.data(), sections.size() * sizeof(GPackSectionEntryV1));
    cursor += sections.size() * sizeof(GPackSectionEntryV1);

    const uint64_t footer_offset = align_up(cursor, 64);
    pad_to(cursor, footer_offset);
    cursor = footer_offset;

    GPackFooterV1 footer{};
    footer.manifest_offset = manifest_sec.offset;
    footer.section_table_offset = section_table_offset;
    footer.total_payload_bytes = cursor;
    std::memcpy(footer.manifest_sha256, manifest_sec.sha256, 32);
    footer.section_count = static_cast<uint32_t>(sections.size());
    footer.magic = kGPackFooterMagic;
    write_exact(&footer, sizeof(footer));
    cursor += sizeof(footer);

    GPackHeaderV1 header{};
    header.magic = kGPackMagic;
    header.format_version = kGPackVersionV1;
    header.min_reader_version = kGPackVersionV1;
    header.flags = GUILD_GPACK_FLAG_LITTLE_ENDIAN | GUILD_GPACK_FLAG_HAS_DENSE_PACK | GUILD_GPACK_FLAG_HAS_EXPERT_BLOBS;
    header.section_count = static_cast<uint32_t>(sections.size());
    header.manifest_offset = manifest_sec.offset;
    header.manifest_size = manifest_sec.size;
    header.section_table_offset = section_table_offset;
    header.total_file_size = cursor;
    std::memcpy(header.header_manifest_sha256_prefix, manifest_sec.sha256, 16);

    std::fseek(fp, 0, SEEK_SET);
    write_exact(&header, sizeof(header));
    std::fflush(fp);
    std::fclose(fp);

    fs::rename(tmp_path, out_file, ec);
    if (ec) {
        fs::remove(tmp_path, ec);
        err = "Safetensors import: rename failed";
        return false;
    }

    report.output_path = out_file.string();
    report.manifest_sha256 = bytes_to_hex(manifest_sec.sha256, 32);
    report.total_file_size = cursor;
    report.tensor_count = parsed_tensors.size();
    report.section_count = sections.size();
    report.build_ms = std::chrono::duration<double, std::milli>(Clock::now() - build_t0).count();
    return true;
} catch (const std::exception& e) {
    err = std::string("Safetensors import: ") + e.what();
    return false;
}

} // namespace guild::artifact
