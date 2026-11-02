#include "guild/core/native_dense.hpp"
#include <cstdlib>
#include "guild/core/weights.hpp"
#include "guild/artifact/gguf_reader.hpp"
#include "guild/artifact/gpack.hpp"
#include "guild/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace guild::core {
namespace {
int g_layer_lb = -1, g_layer_le = -1; // set_layer_range; -1: every layer
bool in_range(const std::string& name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0)
        return true;
    const int l = std::atoi(name.c_str() + 4);
    return l >= g_layer_lb && l < g_layer_le;
}
bool eligible(const guild::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0)
        return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight",       ".attn_gate.weight",      ".ssm_out.weight",
                                     ".attn_q.weight",         ".attn_k.weight",         ".attn_v.weight",
                                     ".attn_output.weight",    ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight",
                                     ".ffn_down_shexp.weight", ".nextn.eh_proj.weight"};
    for (const char* suffix : suffixes)
        if (name.ends_with(suffix))
            return true;
    return false;
}
// --dense q6_k (NativeDense::set_q6k) or GUILD_DENSE=q6_k: the Q8_0 matrices of 2048 rows or more whose rows are whole
// 256-value blocks go to the GPU as Q6_K, 6.56 instead of 8.5 bits a weight: each row dequantized exactly, then
// quantized as llama-quantize would without an importance matrix.  Another rounding of the weights, so the output
// changes.  Smaller matrices (the shared expert's 640 rows, k and v's 512) stay Q8_0: their products wait on latency,
// not bytes (Qwen3.8's shared expert gate 2560 x 640: 11.4 us either way, 12.2 -> 13.9 us at two columns), and the PLE
// key keeps its type (its kernel reads only Q8_0 among these).
bool g_q6k = false; // NativeDense::set_q6k
bool dense_q6k() {
    static const bool env = [] {
        const char* e = std::getenv("GUILD_DENSE");
        return e && std::string(e) == "q6_k";
    }();
    return g_q6k || env;
}
std::vector<uint8_t> q8_0_to_q6k(const uint8_t* q8, int64_t n_in, int64_t n_out) {
    const size_t q6_row = ggml_row_size(GGML_TYPE_Q6_K, n_in), q8_row = (size_t) (n_in / 32) * 34;
    std::vector<uint8_t> out(q6_row * (size_t) n_out);
    constexpr int64_t kRows = 64;
    std::atomic<int64_t> next{0};
    auto work = [&] {
        std::vector<float> buf((size_t) kRows * n_in);
        for (;;) {
            const int64_t r0 = next.fetch_add(kRows);
            if (r0 >= n_out)
                return;
            const int64_t nr = std::min(kRows, n_out - r0);
            for (int64_t r = 0; r < nr; ++r) {
                const uint8_t* q = q8 + (size_t) (r0 + r) * q8_row;
                float* f = buf.data() + (size_t) r * n_in;
                for (int64_t b = 0; b < n_in / 32; ++b) {
                    ggml_fp16_t h;
                    std::memcpy(&h, q + b * 34, 2);
                    const float d = ggml_fp16_to_fp32(h);
                    for (int k = 0; k < 32; ++k)
                        f[b * 32 + k] = d * (float) (int8_t) q[b * 34 + 2 + k];
                }
            }
            ggml_quantize_chunk(GGML_TYPE_Q6_K, buf.data(), out.data() + (size_t) r0 * q6_row, 0, nr, n_in, nullptr);
        }
    };
    const unsigned n = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned i = 1; i < n; ++i)
        th.emplace_back(work);
    work();
    for (auto& t : th)
        t.join();
    return out;
}
bool g_hc_q8 = false; // NativeDense::set_hc_q8
// a layer's hyper-connection down / up matrix in the GGUF's Q8_0, its shape the read's ([320][10240] or [10240][320])
bool hc_q8_tensor(const guild::TensorInfo& tensor) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0 || tensor.type != 8 || tensor.shape.size() != 2)
        return false;
    const bool down = name.ends_with(".hc_attn_down.weight") || name.ends_with(".hc_ffn_down.weight");
    const bool up = name.ends_with(".hc_attn_up.weight") || name.ends_with(".hc_ffn_up.weight");
    return (down && tensor.shape[0] == 10240 && tensor.shape[1] == 320) ||
           (up && tensor.shape[0] == 320 && tensor.shape[1] == 10240);
}
// Q8_0 blocks (f16 d, 32 int8) of `rows` rows of `cols` values -> the read's two planes: the int8 values row-major,
// then the scales row-major, one f16 a 32 values.  Bytes moved, not changed.
std::vector<uint8_t> hc_q8_planes(const uint8_t* q8, int64_t cols, int64_t rows) {
    const int64_t nb = cols / 32;
    std::vector<uint8_t> out((size_t) (rows * cols + rows * nb * 2));
    uint8_t* qs = out.data();
    uint8_t* sc = out.data() + (size_t) (rows * cols);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t b = 0; b < nb; ++b) {
            const uint8_t* blk = q8 + (size_t) (r * nb + b) * 34;
            std::memcpy(sc + (size_t) (r * nb + b) * 2, blk, 2);
            std::memcpy(qs + (size_t) (r * cols + b * 32), blk + 2, 32);
        }
    return out;
}
struct DeviceFree {
    void operator()(void* p) const {
        if (p)
            cudaFree(p);
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
};
struct HcPending {
    WeightRef* ref;
    uint64_t bytes;
    DevicePtr data;
};
} // namespace

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key, std::set<std::string>& out,
                               std::string& err) {
    try {
        for (const auto& path : shards) {
            guild::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if ((eligible(tensor, include_ple_key) && guild::kernels::native_mmvq_supported(tensor.type) &&
                     tensor.shape.size() == 2) ||
                    (hc_q8() && hc_q8_tensor(tensor)))
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_q6k(bool on) {
    g_q6k = on;
}

void NativeDense::set_hc_q8(bool on) {
    g_hc_q8 = on;
}

bool NativeDense::hc_q8() {
    static const bool env = [] {
        const char* e = std::getenv("GUILD_HC");
        return e && std::string(e) == "q8_0";
    }();
    return g_hc_q8 || env;
}

void NativeDense::set_layer_range(int lb, int le) {
    g_layer_lb = lb;
    g_layer_le = le;
}
bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key))
        return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err))
        return false;
    if (code_bits == 0)
        skip.erase(key);
    return true;
}

NativeDense::~NativeDense() {
    if (scratch_)
        cudaFree(scratch_);
    for (void* p : weights_)
        cudaFree(p);
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key, int64_t layer_lo, int64_t layer_hi) {
    auto outside = [&](const std::string& name) { // a blk.<l>. tensor of another stage's layers
        if (layer_hi < 0 || name.rfind("blk.", 0) != 0)
            return false;
        const long l = std::strtol(name.c_str() + 4, nullptr, 10);
        return l < layer_lo || l >= layer_hi;
    };
    if (scratch_ || !weights_.empty()) {
        err = "native dense: already loaded";
        return false;
    }
    if (shards.empty()) {
        err = "native dense: at least one GGUF shard is required";
        return false;
    }
    try {
        std::vector<Pending> pending;
        std::vector<HcPending> hc_pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            guild::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture")) {
                err = guild::check_architecture(gguf);
                if (!err.empty())
                    return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors || count->u != split_count ||
                       number->u == 0 || number->u >= split_count || tensors->u != split_tensors) {
                err =
                    "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number";
                return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors())
                offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap";
                return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !guild::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name;
                    return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name;
                        return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name;
                    return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name;
                    return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name;
                    return false;
                }
            }
            for (const auto& tensor : gguf.tensors()) {
                if (hc_q8() && hc_q8_tensor(tensor) && !outside(tensor.name) && in_range(tensor.name)) {
                    auto found = table.table_.find(tensor.name);
                    if (found == table.table_.end() || found->second.hc_q8 ||
                        tensor.shape[0] != (uint64_t) found->second.ne0 ||
                        tensor.shape[1] != (uint64_t) found->second.ne1) {
                        err = "native dense: hyper-connection matrix " + tensor.name +
                              " is absent from the canonical table, loaded twice, or of another shape";
                        return false;
                    }
                    const auto planes = hc_q8_planes(static_cast<const uint8_t*>(gguf.tensor_data(tensor)),
                                                     (int64_t) tensor.shape[0], (int64_t) tensor.shape[1]);
                    void* allocation = nullptr;
                    auto status = cudaMalloc(&allocation, planes.size());
                    DevicePtr data(allocation);
                    if (status == cudaSuccess)
                        status = cudaMemcpy(data.get(), planes.data(), planes.size(), cudaMemcpyHostToDevice);
                    if (status != cudaSuccess) {
                        err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status);
                        return false;
                    }
                    total += planes.size();
                    hc_pending.push_back(HcPending{&found->second, planes.size(), std::move(data)});
                    continue;
                }
                if (!eligible(tensor, include_ple_key) || outside(tensor.name))
                    continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos)
                    continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name;
                    return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name;
                    return false;
                }
                auto& ref = found->second;
                if (ref.native_data) {
                    err = "native dense: override already attached";
                    return false;
                }
                if (!guild::kernels::native_mmvq_supported(tensor.type))
                    continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized())
                    continue;
                if (!ref.quantized() || tensor.shape.size() != 2 || ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 ||
                    ref.ne1 > INT_MAX || tensor.shape[0] != (uint64_t) ref.ne0 ||
                    tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name;
                    return false;
                }
                int type = (int) tensor.type;
                const void* src = gguf.tensor_data(tensor);
                std::vector<uint8_t> q6;
                if (dense_q6k() && type == 8 && ref.ne0 % 256 == 0 && ref.ne1 >= 2048 &&
                    tensor.name.find("ple") == std::string::npos) {
                    q6 = q8_0_to_q6k(static_cast<const uint8_t*>(src), ref.ne0, ref.ne1);
                    src = q6.data();
                    type = 14;
                    ++requantized_;
                }
                const auto bytes = guild::kernels::native_mmvq_weight_bytes(type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, bytes);
                DevicePtr data(allocation);
                if (status == cudaSuccess)
                    status = cudaMemcpy(data.get(), src, bytes, cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status);
                    return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                pending.push_back(Pending{&ref, type, bytes, std::move(data)});
            }
        }
        if (pending.empty()) {
            err = "native dense: no supported GDN/QSA matrices in supplied shards";
            return false;
        }
        void* allocation = nullptr;
        const auto status = cudaMalloc(&allocation, guild::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation);
        if (status != cudaSuccess) {
            err = std::string("native dense scratch: ") + cudaGetErrorString(status);
            return false;
        }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size() + hc_pending.size());
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            weights_.push_back(item.data.release());
        }
        for (auto& item : hc_pending) {
            item.ref->hc_q8 = static_cast<const int8_t*>(item.data.get());
            hc_q8_bytes_ += item.bytes;
            ++hc_q8_count_;
            weights_.push_back(item.data.release());
        }
        scratch_ = scratch.release();
        bytes_ = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

bool NativeDense::load_gpack(const guild::artifact::GPackReader& gpack, WeightTable& table, std::string& err) {
    if (scratch_ || !weights_.empty()) {
        err = "native dense: already loaded";
        return false;
    }
    if (!gpack.has_native_dense() || gpack.native_dense_rows().empty()) {
        err = "native dense: GPACK has no NativeDenseIndexBin section";
        return false;
    }
    uint64_t blob_size = 0;
    const uint8_t* blob_ptr = gpack.native_dense_data(&blob_size);
    if (!blob_ptr) {
        err = "native dense: GPACK has no NativeDenseBlob section";
        return false;
    }

    const uint64_t arena_bytes = gpack.native_dense_arena_bytes();
    void* arena_raw = nullptr;
    if (cudaMalloc(&arena_raw, arena_bytes) != cudaSuccess) {
        err = "native dense GPACK arena allocation failed";
        return false;
    }
    DevicePtr arena(arena_raw);
    uint8_t* arena_base = static_cast<uint8_t*>(arena.get());

    constexpr uint64_t kStageBytes = 16ull << 20;
    void* stage_buf = nullptr;
    if (cudaHostAlloc(&stage_buf, kStageBytes, cudaHostAllocDefault) != cudaSuccess) {
        stage_buf = nullptr;
    }

    int max_in = 0;
    uint64_t total = 0;
    struct PendingSlice {
        WeightRef* ref;
        int type;
        void* dev_ptr;
    };
    std::vector<PendingSlice> slices;
    slices.reserve(gpack.native_dense_rows().size());

    for (const auto& r : gpack.native_dense_rows()) {
        if (r.blob_offset > blob_size || r.byte_size > blob_size - r.blob_offset || r.arena_offset > arena_bytes ||
            r.byte_size > arena_bytes - r.arena_offset) {
            if (stage_buf)
                cudaFreeHost(stage_buf);
            err = "native dense: GPACK row out of bounds: " + r.name;
            return false;
        }
        auto found = table.table_.find(r.name);
        if (found == table.table_.end()) {
            if (stage_buf)
                cudaFreeHost(stage_buf);
            err = "native dense: tensor absent from canonical table: " + r.name;
            return false;
        }
        auto& ref = found->second;
        if (!guild::kernels::native_mmvq_supported(r.ggml_type))
            continue;
        if (r.name == "blk.1.ple_key.weight" && !ref.quantized())
            continue;
        if (!ref.quantized() || ref.ne0 != r.ne0 || ref.ne1 != r.ne1) {
            if (stage_buf)
                cudaFreeHost(stage_buf);
            err = "native dense: incompatible GPACK matrix " + r.name;
            return false;
        }
        const uint64_t expected_bytes = guild::kernels::native_mmvq_weight_bytes(r.ggml_type, (int) r.ne0, (int) r.ne1);
        if (expected_bytes != r.byte_size) {
            if (stage_buf)
                cudaFreeHost(stage_buf);
            err = "native dense: GPACK byte size mismatch for " + r.name;
            return false;
        }

        uint8_t* dst = arena_base + r.arena_offset;
        const uint8_t* src = blob_ptr + r.blob_offset;
        if (stage_buf) {
            uint64_t done = 0;
            while (done < r.byte_size) {
                const uint64_t chunk = std::min(kStageBytes, r.byte_size - done);
                std::memcpy(stage_buf, src + done, static_cast<size_t>(chunk));
                const auto st = cudaMemcpy(dst + done, stage_buf, static_cast<size_t>(chunk), cudaMemcpyHostToDevice);
                if (st != cudaSuccess) {
                    cudaFreeHost(stage_buf);
                    err = "native dense GPACK upload failed for " + r.name;
                    return false;
                }
                done += chunk;
            }
        } else {
            const auto st = cudaMemcpy(dst, src, static_cast<size_t>(r.byte_size), cudaMemcpyHostToDevice);
            if (st != cudaSuccess) {
                err = "native dense GPACK upload failed for " + r.name;
                return false;
            }
        }
        max_in = (std::max)(max_in, (int) r.ne0);
        total += r.byte_size;
        slices.push_back(PendingSlice{&ref, r.ggml_type, dst});
    }
    if (stage_buf)
        cudaFreeHost(stage_buf);
    if (slices.empty()) {
        err = "native dense: no supported matrices in GPACK";
        return false;
    }

    void* scratch_raw = nullptr;
    const auto status = cudaMalloc(&scratch_raw, guild::kernels::native_q8_1_bytes(max_in));
    DevicePtr scratch(scratch_raw);
    if (status != cudaSuccess) {
        err = std::string("native dense scratch: ") + cudaGetErrorString(status);
        return false;
    }

    for (auto& sl : slices) {
        sl.ref->native_data = sl.dev_ptr;
        sl.ref->native_type = sl.type;
        sl.ref->native_q8_1 = scratch.get();
    }
    weights_.push_back(arena.release());
    scratch_ = scratch.release();
    bytes_ = total;
    return true;
}
} // namespace guild::core
