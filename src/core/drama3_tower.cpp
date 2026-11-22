// src/core/drama3_tower.cpp - Drama 3's codebook tower (include/guild/core/drama3_tower.hpp).
#include "guild/core/drama3_tower.hpp"

#include "guild/core/dense_quant.hpp"
#include "guild/core/drama3_tower_kernels.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"
#include "guild/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#if defined(__x86_64__) && defined(__GNUC__) && !defined(_MSC_VER)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace guild::core {

namespace {

constexpr int kQ8_0 = 8; // ggml type id

struct Entry {
    std::string kind;
    int64_t rows = 0, cols = 0;
    uint64_t offset = 0, bytes = 0;
};

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// GUILD_DRAMA3_FUSE: unset or 1, "ksaw"; 0, the separate kernels, the reference sampler and the file's layout of
// before; letters pick parts (k: the fused small kernels, s: the fast sampler, a: the staged attention, w: [w1|w3] as
// one matvec, f: a and w in step_finish's layers too).  Without f, a and w run in step_begin's layers only, the
// critical path (the backbone waits for them); in every layer they shortened the tower step but lengthened the
// backbone step beside step_finish about as much.  Each part gives the same bits as the path it replaces.
bool fuse_part(char part) {
    const char* fz = std::getenv("GUILD_DRAMA3_FUSE");
    std::string f = fz ? fz : "";
    if (f.find_first_not_of("0123456789") == std::string::npos) // unset or a number
        f = f != "0" ? "ksaw" : "";
    return f.find(part) != std::string::npos;
}

bool cuda_ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess)
        return true;
    err = std::string("drama3 tower: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

} // namespace

Drama3Tower::~Drama3Tower() {
    for (Layer& L : layers_) {
        cudaFree(L.k_cache);
        cudaFree(L.v_cache);
    }
    for (void* p : {(void*) h_, (void*) e_, (void*) cat_, (void*) x_, (void*) xn_, (void*) qkv_, (void*) q_,
                    (void*) att_, (void*) proj_, (void*) ff1_, (void*) logits_, (void*) part_, xq_, dense_})
        cudaFree(p);
    for (void* p : {(void*) bh_, (void*) be_, (void*) bcat_, (void*) bx_, (void*) bxn_, (void*) bqkv_, (void*) bq_,
                    (void*) batt_, (void*) bproj_, (void*) bff1_, (void*) bff3_, bxq_, (void*) bidx_})
        cudaFree(p);
    if (host_logits_)
        cudaFreeHost(host_logits_);
    if (stream_)
        cudaStreamDestroy((cudaStream_t) stream_);
    if (prio_stream_)
        cudaStreamDestroy((cudaStream_t) prio_stream_);
}

const uint16_t* Drama3Tower::code1_rows() const {
    return codebook_embeddings_ + (size_t) g_.codebook_size * g_.dim;
}

bool Drama3Tower::load(const std::string& dir, int64_t max_frames, std::string& err) {
    std::map<std::string, std::string> meta;
    {
        std::ifstream in(dir + "/tower.txt");
        if (!in) {
            err = "drama3 tower: cannot read " + dir + "/tower.txt (tools/drama3_tower_rt.py writes it)";
            return false;
        }
        std::string k, v;
        while (in >> k >> v)
            meta[k] = v;
    }
    auto num = [&](const char* k) -> double {
        auto it = meta.find(k);
        if (it == meta.end())
            throw std::runtime_error(std::string("tower.txt lacks ") + k);
        return std::stod(it->second);
    };
    try {
        if (meta["arch"] != "drama3-tower")
            throw std::runtime_error("tower.txt: arch is not drama3-tower");
        g_.dim = (int) num("dim");
        g_.n_head = (int) num("n_head");
        g_.n_kv_head = (int) num("n_kv_head");
        g_.head_dim = (int) num("head_dim");
        g_.ffn = (int) num("ffn");
        g_.rope_dim = (int) num("rope_dim");
        g_.rope_base = (float) num("rope_base");
        g_.norm_eps = (float) num("norm_eps");
        g_.backbone_eps = (float) num("backbone_eps");
        g_.n_shared = (int) num("n_shared");
        g_.n_codebooks = (int) num("n_codebooks");
        g_.codebook_size = (int) num("codebook_size");
        g_.semantic_start = (int32_t) num("semantic_start");
        g_.semantic_end = (int32_t) num("semantic_end");
        g_.eos = (int32_t) num("eos");
    } catch (const std::exception& e) {
        err = std::string("drama3 tower: ") + e.what();
        return false;
    }
    if (g_.n_codebooks < 2 || g_.n_codebooks > 10 || g_.head_dim > 256 || g_.n_head % g_.n_kv_head || g_.dim % 256 ||
        g_.ffn % 256) {
        err = "drama3 tower: unsupported geometry in tower.txt";
        return false;
    }

    std::map<std::string, Entry> index;
    {
        std::ifstream in(dir + "/dense.txt");
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string name;
            Entry e;
            if (ls >> name >> e.kind >> e.rows >> e.cols >> e.offset >> e.bytes)
                index[name] = e;
        }
    }
    std::ifstream bin(dir + "/dense.bin", std::ios::binary | std::ios::ate);
    if (!bin || index.empty()) {
        err = "drama3 tower: cannot read " + dir + "/dense.bin or dense.txt";
        return false;
    }
    const uint64_t file_size = (uint64_t) bin.tellg();
    std::vector<char> host(file_size);
    bin.seekg(0);
    bin.read(host.data(), (std::streamsize) file_size);
    if (!bin) {
        err = "drama3 tower: short read of dense.bin";
        return false;
    }
    // GUILD_DRAMA3_TOWER=q6_k|q5_k|q4_k, or kind=type,... (kinds eh_proj, wqkv, wo, w1, w2, w3, head): the Q8_0
    // matrices re-quantized at load (dense_quant's q8_0_requant), each row's bytes fewer to read a frame.  It changes
    // the codes' logits, so it is opt-in; unset, the matrices are the file's.
    std::map<std::string, int> retyped;
    if (const char* e = std::getenv("GUILD_DRAMA3_TOWER"); e && *e) {
        const std::string spec = e;
        const bool one = spec.find('=') == std::string::npos;
        std::vector<std::pair<uint64_t, std::string>> order; // the file's order, so offsets grow as they did
        for (const auto& [name, en] : index)
            order.push_back({en.offset, name});
        std::sort(order.begin(), order.end());
        std::vector<char> out;
        out.reserve(host.size());
        for (const auto& [offset, name] : order) {
            Entry& en = index[name];
            const std::string kind = name.rfind("head.", 0) == 0 ? "head" : name.substr(name.rfind('.') + 1);
            const int type = en.kind != "q8_0" || en.cols % 256 ? 0
                             : one                              ? dense_map_type("x=" + spec, "x")
                                                                : dense_map_type(spec, "blk.0." + kind + ".weight");
            if (offset + en.bytes > file_size) {
                err = "drama3 tower: dense.txt: " + name + " runs past dense.bin";
                return false;
            }
            const size_t at = (out.size() + 255) & ~(size_t) 255; // the device copy's rows stay 256-byte aligned
            out.resize(at);
            if (type == 12 || type == 13 || type == 14) {
                const std::vector<uint8_t> rq =
                    q8_0_requant((const uint8_t*) host.data() + offset, en.cols, en.rows, type, nullptr);
                out.insert(out.end(), rq.begin(), rq.end());
                en.bytes = rq.size();
                en.kind = "q8_0"; // found as a matrix; `retyped` says which type it is now
                retyped[name] = type;
            } else {
                out.insert(out.end(), host.begin() + (std::ptrdiff_t) offset,
                           host.begin() + (std::ptrdiff_t)(offset + en.bytes));
            }
            en.offset = at;
        }
        std::fprintf(stderr, "guild drama3: tower matrices re-quantized (%s): %zu of them, %.0f -> %.0f MiB\n", e,
                     retyped.size(), file_size / 1048576.0, out.size() / 1048576.0);
        host.swap(out);
    }
    // [w1|w3] (fuse_part w): the file holds each layer's w1, w2 and w3 in that order; with w3 moved to w2's place and
    // w2 to w3's (the same bytes, the same VRAM), w1 and w3 are one matrix of 2 ffn rows, and the step computes both
    // in one MMVQ call, each row summed alone as before (the native Q8_0 MMVQ's rows are independent:
    // drama3_fuse_test).  Only the file's own Q8_0 matrices (not GUILD_DRAMA3_TOWER's).
    if (fuse_part('w') && retyped.empty()) {
        const int n_layers = g_.n_shared + g_.n_codebooks - 1;
        for (int l = 0; l < n_layers; ++l) {
            const std::string p = l < g_.n_shared ? "shared." + std::to_string(l) + "."
                                                  : "residual." + std::to_string(l - g_.n_shared) + ".";
            const auto i1 = index.find(p + "w1"), i2 = index.find(p + "w2"), i3 = index.find(p + "w3");
            if (i1 == index.end() || i2 == index.end() || i3 == index.end())
                continue;
            Entry &e1 = i1->second, &e2 = i2->second, &e3 = i3->second;
            if (e1.kind != "q8_0" || e2.kind != "q8_0" || e3.kind != "q8_0" || e1.bytes != e3.bytes ||
                e2.bytes != e3.bytes || e1.offset + e1.bytes != e2.offset || e2.offset + e2.bytes != e3.offset ||
                e3.offset + e3.bytes > host.size())
                continue;
            std::swap_ranges(host.begin() + (std::ptrdiff_t) e2.offset,
                             host.begin() + (std::ptrdiff_t)(e2.offset + e2.bytes),
                             host.begin() + (std::ptrdiff_t) e3.offset);
            std::swap(e2.offset, e3.offset);
        }
    }
    const uint64_t size = host.size();
    if (!cuda_ok(cudaMalloc(&dense_, size), "dense weights", err) ||
        !cuda_ok(cudaMemcpy(dense_, host.data(), size, cudaMemcpyHostToDevice), "dense upload", err))
        return false;
    vram_ = size;

    auto find = [&](const std::string& name, const char* kind, int64_t rows, int64_t cols) -> const void* {
        auto it = index.find(name);
        if (it == index.end() || it->second.kind != kind || (rows >= 0 && it->second.rows != rows) ||
            (cols >= 0 && it->second.cols != cols) || it->second.offset + it->second.bytes > size)
            throw std::runtime_error("dense.txt: " + name + " is missing, of another kind or shape");
        return (const char*) dense_ + it->second.offset;
    };
    const int D = g_.dim, Q = g_.n_head * g_.head_dim, KV = g_.n_kv_head * g_.head_dim, CS = g_.codebook_size;
    try {
        auto mat = [&](const std::string& name, int rows, int cols) {
            const auto it = retyped.find(name);
            return Mat{find(name, "q8_0", rows, cols), rows, cols, it == retyped.end() ? kQ8_0 : it->second};
        };
        auto vec = [&](const std::string& name, int n) { return (const float*) find(name, "f32", 1, n); };
        const int n_layers = g_.n_shared + g_.n_codebooks - 1;
        layers_.resize((size_t) n_layers);
        for (int l = 0; l < n_layers; ++l) {
            const std::string p = l < g_.n_shared ? "shared." + std::to_string(l) + "."
                                                  : "residual." + std::to_string(l - g_.n_shared) + ".";
            Layer& L = layers_[(size_t) l];
            L.embedding = (const uint16_t*) find(p + "embedding", "bf16", CS, D);
            L.enorm = vec(p + "enorm", D);
            L.hnorm = vec(p + "hnorm", D);
            L.final_norm = vec(p + "final_layernorm", D);
            L.attn_norm = vec(p + "attention_norm", D);
            L.ffn_norm = vec(p + "ffn_norm", D);
            L.q_norm = vec(p + "q_norm", g_.head_dim);
            L.k_norm = vec(p + "k_norm", g_.head_dim);
            L.eh_proj = mat(p + "eh_proj", D, 2 * D);
            L.wqkv = mat(p + "wqkv", 2 * Q + 2 * KV, D);
            L.wo = mat(p + "wo", D, Q);
            L.w1 = mat(p + "w1", g_.ffn, D);
            L.w3 = mat(p + "w3", g_.ffn, D);
            if (L.w1.type == kQ8_0 && L.w3.type == kQ8_0 &&
                (const char*) L.w3.w ==
                    (const char*) L.w1.w + guild::kernels::native_mmvq_weight_bytes(kQ8_0, D, g_.ffn))
                L.w13 = Mat{L.w1.w, 2 * g_.ffn, D, kQ8_0};
            L.w2 = mat(p + "w2", D, g_.ffn);
        }
        for (int j = 0; j + 1 < g_.n_codebooks; ++j)
            heads_.push_back(mat("head." + std::to_string(j), CS, D));
        cand_ = mat("head.cand", CS + 1, D);
        codebook_embeddings_ = (const uint16_t*) find("codebook_embeddings", "bf16", (int64_t) CS * g_.n_codebooks, D);
    } catch (const std::exception& e) {
        err = std::string("drama3 tower: ") + e.what();
        return false;
    }

    max_frames_ = max_frames;
    const size_t kv_bytes = (size_t) max_frames * KV * 2;
    for (Layer& L : layers_) {
        if (!cuda_ok(cudaMalloc((void**) &L.k_cache, kv_bytes), "K cache", err) ||
            !cuda_ok(cudaMalloc((void**) &L.v_cache, kv_bytes), "V cache", err))
            return false;
        vram_ += 2 * kv_bytes;
    }
    drama3::QkvShape qs{g_.n_head, g_.n_kv_head, g_.head_dim, g_.rope_dim, g_.rope_base, g_.norm_eps};
    struct Buf {
        float** p;
        size_t n;
    } bufs[] = {{&h_, (size_t) D},           {&e_, (size_t) D},
                {&cat_, (size_t) 2 * D},     {&x_, (size_t) D},
                {&xn_, (size_t) D},          {&qkv_, (size_t) (2 * Q + 2 * KV)},
                {&q_, (size_t) Q},           {&att_, (size_t) Q},
                {&proj_, (size_t) D},        {&ff1_, (size_t) 2 * g_.ffn},
                {&logits_, (size_t) CS + 1}, {&part_, drama3::attn_part_floats(qs, max_frames)}};
    for (const Buf& b : bufs) {
        if (!cuda_ok(cudaMalloc((void**) b.p, b.n * 4), "scratch", err))
            return false;
        vram_ += b.n * 4;
    }
    ff3_ = ff1_ + g_.ffn;
    {
        const size_t P = kPrefillCols;
        struct PBuf {
            float** p;
            size_t n;
        } pb[] = {{&bh_, P * D},
                  {&be_, P * D},
                  {&bcat_, P * 2 * D},
                  {&bx_, P * D},
                  {&bxn_, P * D},
                  {&bq_, P * Q},
                  {&bqkv_, P * (2 * Q + 2 * KV)},
                  {&batt_, P * Q},
                  {&bproj_, P * D},
                  {&bff1_, P * g_.ffn},
                  {&bff3_, P * g_.ffn}};
        for (const PBuf& b : pb) {
            if (!cuda_ok(cudaMalloc((void**) b.p, b.n * 4), "prefill scratch", err))
                return false;
            vram_ += b.n * 4;
        }
        const size_t bxq = guild::kernels::native_q8_1_bytes(std::max({2 * D, g_.ffn, Q}), kPrefillCols);
        const size_t idx = ((size_t) layers_.size() + 11) * P;
        if (!cuda_ok(cudaMalloc(&bxq_, bxq), "prefill q8_1 scratch", err) ||
            !cuda_ok(cudaMalloc((void**) &bidx_, idx * 4), "prefill rows", err))
            return false;
        vram_ += bxq + idx * 4;
        bidx_host_.assign(idx, 0);
    }
    const size_t xq_bytes = guild::kernels::native_q8_1_bytes(std::max({2 * D, g_.ffn, Q}), 1);
    if (!cuda_ok(cudaMalloc(&xq_, xq_bytes), "q8_1 scratch", err) ||
        !cuda_ok(cudaMallocHost((void**) &host_logits_, (size_t) (CS + 1) * 4), "pinned logits", err))
        return false;
    cudaStream_t st = nullptr;
    if (!cuda_ok(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream", err))
        return false;
    stream_ = st;
    // the top-priority stream for step_begin; without one (a backend without priorities) every step stays on stream_
    int least = 0, greatest = 0;
    cudaStream_t ps = nullptr;
    if (cudaDeviceGetStreamPriorityRange(&least, &greatest) == cudaSuccess &&
        cudaStreamCreateWithPriority(&ps, cudaStreamNonBlocking, greatest) == cudaSuccess)
        prio_stream_ = ps;
    else
        cudaGetLastError();
    const char* prio = std::getenv("GUILD_DRAMA3_PRIO");
    set_step_priority(!(prio && prio[0] == '0'));

    // the faster step paths (fuse_part; [w1|w3] was laid out above)
    {
        fuse_ = fuse_part('k') && g_.dim == drama3::kFusedDim &&
                guild::kernels::native_q8_1_bytes(32, 1) == (size_t) drama3::kFusedQ81Bytes;
        std::string why;
        if (fuse_ && !fused_self_check(why)) {
            std::fprintf(stderr,
                         "guild drama3: the fused tower kernels differ from the separate ones (%s); using those\n",
                         why.c_str());
            fuse_ = false;
        }
        fast_sample_ = fuse_part('s') && drama3::sample_fast_ok();
        fast_attn_ = fuse_part('a');
        fast_finish_ = fuse_part('f');
        const int n = (int) layers_.size(), n_begin = std::min(g_.n_shared + 1, n); // step_begin's layers
        int attn = 0, w13 = 0;
        for (int l = 0; l < n; ++l) {
            const bool fast = l < n_begin || fast_finish_;
            attn += fast_attn_ && fast;
            w13 += layers_[(size_t) l].w13.w != nullptr && fuse_ && fast;
        }
        std::fprintf(stderr,
                     "guild drama3: tower step: %s small kernels, %s sampler, staged attention in %d and [w1|w3] in %d "
                     "of %d layers\n",
                     fuse_ ? "fused" : "separate", fast_sample_ ? "AVX-512" : "reference", attn, w13, n);
    }
    return true;
}

bool Drama3Tower::quantize(const float* x, void* xq, int n, std::string& err, void* stream) {
    try {
        guild::kernels::native_quantize_q8_1(x, xq, n, 1, stream);
    } catch (const std::exception& e) {
        err = std::string("drama3 tower quantize: ") + e.what();
        return false;
    }
    return true;
}

bool Drama3Tower::mmvq(const Mat& m, const void* xq, float* y, std::string& err, void* stream) {
    try {
        guild::kernels::native_mmvq(m.type, m.w, xq, y, m.cols, m.rows, 1, stream);
    } catch (const std::exception& e) {
        err = std::string("drama3 tower matvec: ") + e.what();
        return false;
    }
    return true;
}

bool Drama3Tower::matvec(const Mat& m, const float* x, float* y, std::string& err, void* stream, int ncols) {
    void* xq = ncols > 1 ? bxq_ : xq_;
    try {
        guild::kernels::native_quantize_q8_1(x, xq, m.cols, ncols, stream);
        guild::kernels::native_mmvq(m.type, m.w, xq, y, m.cols, m.rows, ncols, stream);
    } catch (const std::exception& e) {
        err = std::string("drama3 tower matvec: ") + e.what();
        return false;
    }
    return true;
}

bool Drama3Tower::fused_self_check(std::string& why) {
    const int D = g_.dim, nb = D / 32;
    const auto st = (cudaStream_t) stream_;
    const Layer& L = layers_[0];
    // a row like the tower's residual stream (normal values, one massive activation) and the projection added to it
    std::vector<float> hx((size_t) D), hy((size_t) D), hh((size_t) D);
    uint64_t s = 0x5eed;
    auto uni = [&] { return (float) ((s = splitmix64(s)) >> 40) * (1.0f / 16777216.0f) - 0.5f; };
    for (int i = 0; i < D; ++i) {
        hx[(size_t) i] = 4.0f * uni();
        hy[(size_t) i] = uni();
        hh[(size_t) i] = 3.0f * uni();
    }
    hx[17] = 1.0e4f;
    float* f = nullptr;
    void* q = nullptr;
    const size_t qb = guild::kernels::native_q8_1_bytes(2 * D, 1);
    if (cudaMalloc((void**) &f, (size_t) 10 * D * 4) != cudaSuccess || cudaMalloc(&q, 2 * qb) != cudaSuccess) {
        cudaFree(f);
        cudaGetLastError();
        why = "no scratch";
        return false;
    }
    float *x0 = f, *x1 = f + D, *y = f + 2 * D, *h = f + 3 * D, *n0 = f + 4 * D, *n1 = f + 5 * D, *c0 = f + 6 * D,
          *c1 = f + 8 * D;
    void *q0 = q, *q1 = (char*) q + qb;
    bool ok = cudaMemcpyAsync(x0, hx.data(), (size_t) D * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(x1, hx.data(), (size_t) D * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(y, hy.data(), (size_t) D * 4, cudaMemcpyHostToDevice, st) == cudaSuccess &&
              cudaMemcpyAsync(h, hh.data(), (size_t) D * 4, cudaMemcpyHostToDevice, st) == cudaSuccess;
    if (!ok)
        why = "upload";
    auto same = [&](const void* a, const void* b, size_t bytes, const char* what) {
        if (!ok)
            return;
        std::vector<char> ha(bytes), hb(bytes);
        if (cudaMemcpyAsync(ha.data(), a, bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaMemcpyAsync(hb.data(), b, bytes, cudaMemcpyDeviceToHost, st) != cudaSuccess ||
            cudaStreamSynchronize(st) != cudaSuccess) {
            ok = false;
            why = std::string("CUDA error at ") + what;
            return;
        }
        if (std::memcmp(ha.data(), hb.data(), bytes) != 0) {
            ok = false;
            why = what;
        }
    };
    const float eps = g_.norm_eps;
    try {
        // x += y; norm; Q8_1 (the ffn norm's place), then the same row normed with an output (the final norm's)
        drama3::add(x0, y, D, st);
        drama3::rms_norm(x0, L.ffn_norm, n0, D, eps, st);
        guild::kernels::native_quantize_q8_1(n0, q0, D, 1, st);
        drama3::add_norm_q8(x1, y, L.ffn_norm, eps, n1, q1, st);
        same(x0, x1, (size_t) D * 4, "add");
        same(n0, n1, (size_t) D * 4, "rms_norm");
        same(q0, q1, (size_t) nb * drama3::kFusedQ81Bytes, "q8_1");
        // the layer input: an embedding row with layer 0's eight adds, and h
        drama3::EmbedAdd add;
        for (int k = 2; k < g_.n_codebooks; ++k)
            add.rows[add.n++] = k * g_.codebook_size + (k * 389) % g_.codebook_size;
        drama3::embed(L.embedding, 77, add, codebook_embeddings_, n0, D, st);
        drama3::rms_norm(n0, L.enorm, c0, D, eps, st);
        drama3::rms_norm(h, L.hnorm, c0 + D, D, eps, st);
        guild::kernels::native_quantize_q8_1(c0, q0, 2 * D, 1, st);
        drama3::embed_norm2_q8(L.embedding, 77, add, codebook_embeddings_, h, L.enorm, L.hnorm, eps, q1, st, c1);
        same(c0, c1, (size_t) 2 * D * 4, "embed + rms_norm");
        same(q0, q1, (size_t) 2 * nb * drama3::kFusedQ81Bytes, "embed q8_1");
    } catch (const std::exception& e) {
        ok = false;
        why = e.what();
    }
    cudaStreamSynchronize(st);
    cudaFree(f);
    cudaFree(q);
    return ok && cuda_ok(cudaGetLastError(), "fused self-check", why);
}

bool Drama3Tower::layer_forward(Layer& L, int row, const drama3::EmbedAdd& add, int64_t pos, bool head, bool finish,
                                std::string& err, void* stream) {
    const auto st = (cudaStream_t) stream;
    const int D = g_.dim, Q = g_.n_head * g_.head_dim;
    const float eps = g_.norm_eps;
    const drama3::QkvShape qs{g_.n_head, g_.n_kv_head, g_.head_dim, g_.rope_dim, g_.rope_base, eps};
    // the staged attention and [w1|w3]: in step_begin's layers, and in step_finish's only with f (fast_finish_)
    const bool fast = !finish || fast_finish_;
    const auto attend = fast_attn_ && fast ? drama3::attention_fast : drama3::attention;
    if (fuse_) {
        // the same operations as below in fewer kernels: the norms take their adds and write the next matvec's Q8_1
        // blocks themselves, and w1 and w3 read one quantization
        drama3::embed_norm2_q8(L.embedding, row, add, codebook_embeddings_, h_, L.enorm, L.hnorm, eps, xq_, st);
        if (!mmvq(L.eh_proj, xq_, x_, err, stream))
            return false;
        drama3::add_norm_q8(x_, nullptr, L.attn_norm, eps, nullptr, xq_, st);
        if (!mmvq(L.wqkv, xq_, qkv_, err, stream))
            return false;
        drama3::qk_norm_rope(qkv_, L.q_norm, L.k_norm, q_, L.k_cache, L.v_cache, qs, pos, st);
        if (!attend(q_, L.k_cache, L.v_cache, qkv_ + Q, part_, att_, qs, pos + 1, st)) {
            err = "drama3 tower: unsupported attention shape";
            return false;
        }
        if (!quantize(att_, xq_, Q, err, stream) || !mmvq(L.wo, xq_, proj_, err, stream))
            return false;
        drama3::add_norm_q8(x_, proj_, L.ffn_norm, eps, nullptr, xq_, st);
        if (L.w13.w && fast ? !mmvq(L.w13, xq_, ff1_, err, stream) // ff1_ then ff3_ = ff1_ + ffn
                            : !mmvq(L.w1, xq_, ff1_, err, stream) || !mmvq(L.w3, xq_, ff3_, err, stream))
            return false;
        try {
            guild::kernels::native_swiglu_quantize_q8_1(ff1_, ff3_, xq_, g_.ffn, 1, stream);
        } catch (const std::exception& ex) {
            err = std::string("drama3 tower ffn: ") + ex.what();
            return false;
        }
        if (!mmvq(L.w2, xq_, proj_, err, stream))
            return false;
        drama3::add_norm_q8(x_, proj_, L.final_norm, eps, h_, head ? xq_ : nullptr, st);
        return cuda_ok(cudaGetLastError(), "layer launch", err);
    }
    drama3::embed(L.embedding, row, add, codebook_embeddings_, e_, D, st);
    drama3::rms_norm(e_, L.enorm, cat_, D, eps, st);
    drama3::rms_norm(h_, L.hnorm, cat_ + D, D, eps, st);
    if (!matvec(L.eh_proj, cat_, x_, err, stream))
        return false;
    drama3::rms_norm(x_, L.attn_norm, xn_, D, eps, st);
    if (!matvec(L.wqkv, xn_, qkv_, err, stream))
        return false;
    drama3::qk_norm_rope(qkv_, L.q_norm, L.k_norm, q_, L.k_cache, L.v_cache, qs, pos, st);
    if (!attend(q_, L.k_cache, L.v_cache, qkv_ + Q, part_, att_, qs, pos + 1, st)) {
        err = "drama3 tower: unsupported attention shape";
        return false;
    }
    if (!matvec(L.wo, att_, proj_, err, stream))
        return false;
    drama3::add(x_, proj_, D, st);
    drama3::rms_norm(x_, L.ffn_norm, xn_, D, eps, st);
    if (!matvec(L.w1, xn_, ff1_, err, stream) || !matvec(L.w3, xn_, ff3_, err, stream))
        return false;
    try {
        guild::kernels::native_swiglu_quantize_q8_1(ff1_, ff3_, xq_, g_.ffn, 1, stream);
        guild::kernels::native_mmvq(L.w2.type, L.w2.w, xq_, proj_, g_.ffn, D, 1, stream);
    } catch (const std::exception& ex) {
        err = std::string("drama3 tower ffn: ") + ex.what();
        return false;
    }
    drama3::add(x_, proj_, D, st);
    drama3::rms_norm(x_, L.final_norm, h_, D, eps, st);
    return cuda_ok(cudaGetLastError(), "layer launch", err);
}

int Drama3Tower::sample(const float* logits, int n, const Drama3Sampling& sp, std::string& err) {
    int c = fast_sample_ ? drama3::sample_fast(logits, n, sp, draws_) : -2;
    if (c == -2)
        c = drama3::sample_reference(logits, n, sp, draws_);
    if (c < 0) {
        err = "drama3 tower: non-finite logits; refusing to sample invalid audio";
        return -1;
    }
    ++draws_;
    return c;
}

// The reference's sample_top_k_top_p (fish_omni/tower.py): logits / T sorted descending; drop rank >= top_k and
// every token whose cumulative probability (over the full softmax) passes top_p, except the first; draw from the
// softmax of the rest.  The draw is the engine's (inverse CDF of one uniform), not torch's exponential race.
int drama3::sample_reference(const float* logits, int n, const Drama3Sampling& sp, uint64_t draw) {
    const float temp = std::max(sp.temperature, 1e-5f);
    float mx = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(logits[i]))
            return -1;
        mx = std::max(mx, logits[i] / temp);
    }
    double z = 0.0;
    for (int i = 0; i < n; ++i)
        z += std::exp((double) (logits[i] / temp - mx));
    const int k = std::max(1, std::min(sp.top_k, n));
    std::vector<int> idx((size_t) n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
    double cum = 0.0;
    int kept = 0;
    std::vector<double> p((size_t) k);
    for (int r = 0; r < k; ++r) {
        const double pr = std::exp((double) (logits[idx[(size_t) r]] / temp - mx)) / z;
        cum += pr;
        if (r > 0 && cum > sp.top_p)
            break;
        p[(size_t) r] = pr;
        kept = r + 1;
    }
    double tot = 0.0;
    for (int r = 0; r < kept; ++r)
        tot += p[(size_t) r];
    const uint64_t bits = splitmix64(sp.seed * 0x100000001b3ull + draw);
    const double u = (double) (bits >> 11) * (1.0 / 9007199254740992.0) * tot;
    double acc = 0.0;
    for (int r = 0; r < kept; ++r) {
        acc += p[(size_t) r];
        if (u < acc)
            return idx[(size_t) r];
    }
    return idx[(size_t) kept - 1];
}

#if defined(__x86_64__) && defined(__GNUC__) && !defined(_MSC_VER)
namespace {

#include "drama3_exp_table.inc"

#define GUILD_D3_AVX512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,fma")))

// std::exp of eight doubles as glibc's __exp_fma computes it (e_exp.c with FMA contraction): x = k ln2 / 128 + r,
// exp(x) = 2^(k / 128) (1 + tail + r + r^2 (C2 + r C3) + r^4 (C4 + r C5)).  Lanes outside its fast path (|x| < 2^-54,
// |x| >= 512, inf, NaN) are std::exp's own.  sample_fast_ok() checks it against std::exp before it is used.
GUILD_D3_AVX512 inline __m512d exp8(__m512d x) {
    const __m512i bits = _mm512_castpd_si512(x);
    const __m512i abstop = _mm512_and_si512(_mm512_srli_epi64(bits, 52), _mm512_set1_epi64(0x7ff));
    // glibc's test: abstop - top12(0x1p-54) < top12(512) - top12(0x1p-54), unsigned
    const __mmask8 fast =
        _mm512_cmplt_epu64_mask(_mm512_sub_epi64(abstop, _mm512_set1_epi64(0x3c9)), _mm512_set1_epi64(0x3f));
    const __m512d shift = _mm512_set1_pd(kExpConsts[1]);
    const __m512d z = _mm512_fmadd_pd(x, _mm512_set1_pd(kExpConsts[0]), shift);
    const __m512i ki = _mm512_castpd_si512(z);
    const __m512d kd = _mm512_sub_pd(z, shift);
    __m512d r = _mm512_fmadd_pd(kd, _mm512_set1_pd(kExpConsts[2]), x);
    r = _mm512_fmadd_pd(kd, _mm512_set1_pd(kExpConsts[3]), r);
    const __m512i idx = _mm512_slli_epi64(_mm512_and_si512(ki, _mm512_set1_epi64(127)), 1);
    const __m512i top = _mm512_slli_epi64(ki, 45);
    const auto* tab = (const long long*) kExpTable;
    const __m512d tail = _mm512_castsi512_pd(_mm512_i64gather_epi64(idx, tab, 8));
    const __m512i sb =
        _mm512_add_epi64(_mm512_i64gather_epi64(_mm512_add_epi64(idx, _mm512_set1_epi64(1)), tab, 8), top);
    const __m512d p23 = _mm512_fmadd_pd(r, _mm512_set1_pd(kExpConsts[5]), _mm512_set1_pd(kExpConsts[4]));
    const __m512d tr = _mm512_add_pd(r, tail);
    const __m512d r2 = _mm512_mul_pd(r, r);
    const __m512d p45 = _mm512_fmadd_pd(r, _mm512_set1_pd(kExpConsts[7]), _mm512_set1_pd(kExpConsts[6]));
    const __m512d q = _mm512_fmadd_pd(p23, r2, tr);
    const __m512d r4 = _mm512_mul_pd(r2, r2);
    const __m512d tmp = _mm512_fmadd_pd(r4, p45, q);
    const __m512d scale = _mm512_castsi512_pd(sb);
    __m512d y = _mm512_fmadd_pd(scale, tmp, scale);
    if (fast != 0xff) {
        alignas(64) double xs[8], ys[8];
        _mm512_store_pd(xs, x);
        _mm512_store_pd(ys, y);
        for (int i = 0; i < 8; ++i)
            if (!((fast >> i) & 1))
                ys[i] = std::exp(xs[i]);
        y = _mm512_load_pd(ys);
    }
    return y;
}

constexpr int kFastMaxN = 8192, kFastMaxK = 64;

// sample_reference's draw with the same numbers: mx is the same maximum (order-free once every logit is finite; a
// zero's sign does not reach exp), x_i the same float expression, e_i = exp8(x_i) = std::exp's bits, z their sum in
// index order, the top k the same indices in the same order (the strict order logit descending, index ascending; a
// row enters only past the k-th, so a later index never displaces an equal logit), and p_r = e_i / z as before.
GUILD_D3_AVX512 int sample_fast_avx512(const float* logits, int n, const Drama3Sampling& sp, uint64_t draw) {
    const float temp = std::max(sp.temperature, 1e-5f);
    const int k = std::max(1, std::min(sp.top_k, n));
    if (k > kFastMaxK || n > kFastMaxN)
        return -2;
    // the scratch on the heap, once a thread (static TLS of this size would be copied into every thread's block)
    struct alignas(64) Scratch {
        float xs[kFastMaxN];
        double es[kFastMaxN];
    };
    static thread_local std::unique_ptr<Scratch> scratch;
    if (!scratch)
        scratch = std::make_unique<Scratch>();
    float* const xs = scratch->xs;
    double* const es = scratch->es;
    // the finite check and the max, 16 at a time
    const __m512 vt = _mm512_set1_ps(temp);
    __m512 vmx = _mm512_set1_ps(-INFINITY);
    __mmask16 bad = 0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 v = _mm512_loadu_ps(logits + i);
        bad |= _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(v), _mm512_set1_epi32(0x7f800000)),
                                       _mm512_set1_epi32(0x7f800000));
        vmx = _mm512_max_ps(vmx, _mm512_div_ps(v, vt));
    }
    if (bad)
        return -1;
    float mx = _mm512_reduce_max_ps(vmx);
    for (; i < n; ++i) {
        if (!std::isfinite(logits[i]))
            return -1;
        mx = std::max(mx, logits[i] / temp);
    }
    const __m512 vm = _mm512_set1_ps(mx);
    for (i = 0; i + 16 <= n; i += 16)
        _mm512_store_ps(xs + i, _mm512_sub_ps(_mm512_div_ps(_mm512_loadu_ps(logits + i), vt), vm));
    for (; i < n; ++i)
        xs[i] = logits[i] / temp - mx;
    for (i = 0; i + 8 <= n; i += 8)
        _mm512_store_pd(es + i, exp8(_mm512_cvtps_pd(_mm256_load_ps(xs + i))));
    for (; i < n; ++i)
        es[i] = std::exp((double) xs[i]);
    double z = 0.0;
    for (i = 0; i < n; ++i)
        z += es[i];
    // the best k so far, sorted
    int top[kFastMaxK];
    int nt = 0;
    auto better = [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
    auto insert = [&](int c) {
        if (nt == k && !better(c, top[k - 1]))
            return;
        int j = nt < k ? nt++ : k - 1;
        while (j > 0 && better(c, top[j - 1])) {
            top[j] = top[j - 1];
            --j;
        }
        top[j] = c;
    };
    for (i = 0; i < n && nt < k; ++i)
        insert(i);
    for (; i + 16 <= n; i += 16) {
        // only a logit above the k-th's can enter (an equal one has the larger index)
        const __mmask16 m =
            _mm512_cmp_ps_mask(_mm512_loadu_ps(logits + i), _mm512_set1_ps(logits[top[k - 1]]), _CMP_GT_OQ);
        for (unsigned mm = m; mm; mm &= mm - 1)
            insert(i + __builtin_ctz(mm));
    }
    for (; i < n; ++i)
        insert(i);
    double cum = 0.0;
    int kept = 0;
    double p[kFastMaxK];
    for (int r = 0; r < k; ++r) {
        const double pr = es[top[r]] / z;
        cum += pr;
        if (r > 0 && cum > sp.top_p)
            break;
        p[r] = pr;
        kept = r + 1;
    }
    double tot = 0.0;
    for (int r = 0; r < kept; ++r)
        tot += p[r];
    const uint64_t bits = splitmix64(sp.seed * 0x100000001b3ull + draw);
    const double u = (double) (bits >> 11) * (1.0 / 9007199254740992.0) * tot;
    double acc = 0.0;
    for (int r = 0; r < kept; ++r) {
        acc += p[r];
        if (u < acc)
            return top[r];
    }
    return top[kept - 1];
}

// the first m of xs[8] where exp8 differs from std::exp (the rest are set to 0)
GUILD_D3_AVX512 int exp8_differ(double* xs, int m) {
    alignas(64) double in[8], ys[8];
    for (int j = 0; j < 8; ++j)
        in[j] = j < m ? xs[j] : 0.0;
    _mm512_store_pd(ys, exp8(_mm512_load_pd(in)));
    int bad = 0;
    for (int j = 0; j < m; ++j) {
        const double e = std::exp(in[j]);
        bad += std::memcmp(&e, &ys[j], 8) != 0;
    }
    return bad;
}

// float bit patterns lo, lo + step, ... < hi (NaNs skipped) whose exp8 differs from std::exp
uint64_t exp8_mismatches(uint64_t lo, uint64_t hi, uint64_t step) {
    uint64_t bad = 0;
    double xs[8];
    int m = 0;
    for (uint64_t u = lo; u < hi; u += step) {
        const uint32_t v = (uint32_t) u;
        float f;
        std::memcpy(&f, &v, 4);
        if (std::isnan(f))
            continue;
        xs[m++] = (double) f;
        if (m == 8) {
            bad += (uint64_t) exp8_differ(xs, m);
            m = 0;
        }
    }
    if (m)
        bad += (uint64_t) exp8_differ(xs, m);
    return bad;
}

bool fast_sampler_check() {
    if (!guild::kernels::cpu::cpu_avx512bw_ok())
        return false;
    // the sampler's inputs are logit / T - max <= 0 as doubles: about a million float patterns spread over [-200, 0],
    // a slice of every exponent near 0 and past 512, and the edges of glibc's fast path
    uint64_t bad = exp8_mismatches(0x80000000ull, 0xc3480000ull, 1077);
    bad += exp8_mismatches(0x00000000ull, 0x7f800001ull, 1u << 16);
    bad += exp8_mismatches(0x80000000ull, 0xff800001ull, 1u << 16);
    for (const float f : {0.0f, -0.0f, 0x1p-54f, -0x1p-54f, 0x1.fffffep-55f, -0x1.fffffep-55f, -511.99997f, -512.0f,
                          -700.0f, -745.2f, -1000.0f, 88.0f, -INFINITY}) {
        uint32_t v;
        std::memcpy(&v, &f, 4);
        bad += exp8_mismatches(v, (uint64_t) v + 1, 1);
    }
    return bad == 0;
}

} // namespace

int drama3::sample_fast(const float* logits, int n, const Drama3Sampling& sp, uint64_t draw) {
    return sample_fast_ok() ? sample_fast_avx512(logits, n, sp, draw) : -2;
}

bool drama3::sample_fast_ok() {
    static const bool ok = fast_sampler_check();
    return ok;
}

uint64_t drama3::exp_mismatches(uint64_t lo, uint64_t hi, uint64_t step) {
    return guild::kernels::cpu::cpu_avx512bw_ok() ? exp8_mismatches(lo, hi, step) : 0;
}
#else
int drama3::sample_fast(const float*, int, const Drama3Sampling&, uint64_t) {
    return -2;
}
bool drama3::sample_fast_ok() {
    return false;
}
uint64_t drama3::exp_mismatches(uint64_t, uint64_t, uint64_t) {
    return 0;
}
#endif

bool Drama3Tower::step(const float* hidden, int64_t pos, const int32_t* frame, const int32_t* forced,
                       const int32_t* history, int n_history, const Drama3Sampling& sp, int32_t* out, std::string& err,
                       float* cand_logits, float* code_logits, float* layer_out) {
    return step_begin(hidden, pos, frame, forced, history, n_history, sp, out, err, cand_logits, code_logits,
                      layer_out) &&
           step_finish(err);
}

bool Drama3Tower::step_begin(const float* hidden, int64_t pos, const int32_t* frame, const int32_t* forced,
                             const int32_t* history, int n_history, const Drama3Sampling& sp, int32_t* out,
                             std::string& err, float* cand_logits, float* code_logits, float* layer_out) {
    cur_.next_layer = (int) layers_.size(); // step_finish does nothing unless this step begins
    if (pos < 0 || pos >= max_frames_) {
        err = "drama3 tower: position " + std::to_string(pos) + " is past the tower's " + std::to_string(max_frames_) +
              " frames";
        return false;
    }
    // every step_begin launch goes to `st`: the top-priority stream unless set_step_priority(false)
    const auto st = (cudaStream_t) (top_ ? prio_stream_ : stream_);
    const int D = g_.dim, CS = g_.codebook_size;
    if (!cuda_ok(cudaMemcpyAsync(h_, hidden, (size_t) D * 4, cudaMemcpyDeviceToDevice, st), "hidden copy", err))
        return false;

    // the main token: eos or <|semantic:c0|>, over eos + the band
    int32_t main = 0;
    if (!forced || cand_logits) {
        if (!matvec(cand_, h_, logits_, err, st) || !read_logits(CS + 1, cand_logits, err, st))
            return false;
    }
    if (forced) {
        main = forced[0];
    } else {
        auto pick = [&](const Drama3Sampling& s) -> int32_t {
            const int c = sample(host_logits_, CS + 1, s, err);
            return c < 0 ? -1 : (c == 0 ? g_.eos : g_.semantic_start + c - 1);
        };
        main = pick(sp);
        if (main < 0)
            return false;
        // repetition-aware sampling (the training recipe): an audio token that repeats one of the last 16 main tokens
        // is drawn again with T 1.0, top-k 30, top-p 0.95
        bool repeated = false;
        for (int i = std::max(0, n_history - 16); i < n_history && g_.is_semantic(main); ++i)
            repeated |= history[i] == main;
        if (repeated) {
            Drama3Sampling ras = sp;
            ras.temperature = 1.0f;
            ras.top_k = 30;
            ras.top_p = 0.95f;
            main = pick(ras);
            if (main < 0)
                return false;
        }
    }
    cur_ = Step{};
    cur_.pos = pos;
    cur_.frame = frame;
    cur_.forced = forced;
    cur_.sp = sp;
    cur_.out = out;
    cur_.code_logits = code_logits;
    cur_.layer_out = layer_out;
    cur_.semantic = g_.is_semantic(main);
    cur_.codes[0] = std::clamp(main - g_.semantic_start, 0, CS - 1);
    // through the first residual layer, whose head gives c1
    const int split = std::min(g_.n_shared + 1, (int) layers_.size());
    // synchronized: `hidden` (the backbone's) is free to change once this returns
    if (!step_layers(0, split, false, err, st) || !cuda_ok(cudaStreamSynchronize(st), "tower step", err))
        return false;
    out[0] = main;
    out[1] = cur_.codes[0];
    if (g_.n_codebooks > 1)
        out[2] = cur_.codes[1];
    return true;
}

bool Drama3Tower::step_finish(std::string& err) {
    if (!step_layers(cur_.next_layer, (int) layers_.size(), true, err, stream_) ||
        !cuda_ok(cudaStreamSynchronize((cudaStream_t) stream_), "tower step", err))
        return false;
    for (int k = 2; cur_.out && k < g_.n_codebooks; ++k)
        cur_.out[1 + k] = cur_.codes[k];
    return true;
}

bool Drama3Tower::read_logits(int n, float* also, std::string& err, void* stream) {
    const auto st = (cudaStream_t) stream;
    if (!cuda_ok(cudaMemcpyAsync(host_logits_, logits_, (size_t) n * 4, cudaMemcpyDeviceToHost, st), "logits", err) ||
        !cuda_ok(cudaStreamSynchronize(st), "tower step", err))
        return false;
    if (also)
        std::memcpy(also, host_logits_, (size_t) n * 4);
    return true;
}

bool Drama3Tower::step_layers(int l0, int l1, bool finish, std::string& err, void* stream) {
    const auto st = (cudaStream_t) stream;
    const int D = g_.dim, CS = g_.codebook_size, NC = g_.n_codebooks;
    Step& c = cur_;
    for (int l = l0; l < l1; ++l) {
        const bool shared = l < g_.n_shared;
        const int j = l - g_.n_shared; // residual index
        const int code = shared ? c.codes[0] : c.codes[j];
        // frame pos's codes the backbone did not see (2..NC-1), summed into the first shared layer's input
        drama3::EmbedAdd unseen;
        if (l == 0 && g_.is_semantic(c.frame[0]))
            for (int k = 2; k < NC; ++k)
                unseen.rows[unseen.n++] = k * CS + std::clamp(c.frame[1 + k], 0, CS - 1);
        // a teacher-forced step takes the next code from `forced`, and runs the head only when its logits are asked for
        const bool head = !shared && (!c.forced || c.code_logits);
        if (!layer_forward(layers_[(size_t) l], c.semantic ? std::clamp(code, 0, CS - 1) : -1, unseen, c.pos, head,
                           finish, err, stream))
            return false;
        if (c.layer_out &&
            !cuda_ok(cudaMemcpyAsync(c.layer_out + (size_t) l * D, h_, (size_t) D * 4, cudaMemcpyDeviceToHost, st),
                     "layer readback", err))
            return false;
        c.next_layer = l + 1;
        if (shared)
            continue;
        // fused: the final norm left h_'s Q8_1 blocks in xq_
        if (head && (!(fuse_ ? mmvq(heads_[(size_t) j], xq_, logits_, err, stream)
                             : matvec(heads_[(size_t) j], h_, logits_, err, stream)) ||
                     !read_logits(CS, c.code_logits ? c.code_logits + (size_t) j * CS : nullptr, err, stream)))
            return false;
        if (c.forced) {
            c.codes[j + 1] = c.forced[2 + j];
        } else {
            const int s = sample(host_logits_, CS, c.sp, err);
            if (s < 0)
                return false;
            c.codes[j + 1] = s;
        }
    }
    return true;
}

bool Drama3Tower::prefill(const float* hidden, int64_t pos0, int n, const int32_t* rows, std::string& err,
                          float* layer_out) {
    if (pos0 < 0 || n < 0 || pos0 + n > max_frames_) {
        err = "drama3 tower: the prompt runs past the tower's " + std::to_string(max_frames_) + " frames";
        return false;
    }
    for (int c0 = 0; c0 < n; c0 += kPrefillCols) {
        const int m = std::min(kPrefillCols, n - c0);
        if (!prefill_chunk(hidden + (size_t) c0 * g_.dim, pos0 + c0, m, rows + (size_t) c0 * g_.width(), err,
                           layer_out ? layer_out + (size_t) c0 * layers_.size() * g_.dim : nullptr))
            return false;
    }
    return true;
}

bool Drama3Tower::prefill_chunk(const float* hidden, int64_t pos0, int n, const int32_t* rows, std::string& err,
                                float* layer_out) {
    const auto st = (cudaStream_t) stream_;
    const int D = g_.dim, W = g_.width(), CS = g_.codebook_size, NC = g_.n_codebooks;
    const int Q = g_.n_head * g_.head_dim, KV = g_.n_kv_head * g_.head_dim, QKV = 2 * Q + 2 * KV;
    const int NL = (int) layers_.size(), P = kPrefillCols;
    const float eps = g_.norm_eps;
    // bidx_: [NL][P] each (layer, row)'s embedding row, [P][9] the first shared layer's adds, [P] their counts, and
    // [P] zero counts for the other layers
    int32_t* const idx = bidx_host_.data();
    int32_t* const add = idx + (size_t) NL * P;
    int32_t* const add_n = add + (size_t) 9 * P;
    for (int i = 0; i < n; ++i) {
        const int32_t* frame = rows + (size_t) i * W;
        const int32_t* next = frame + W; // the forced frame: `step`'s `forced`
        const bool semantic = g_.is_semantic(next[0]);
        const int c0 = std::clamp(next[0] - g_.semantic_start, 0, CS - 1);
        for (int l = 0; l < NL; ++l) {
            const int j = l - g_.n_shared; // residual j embeds code j (code 0 = c0, code j = next[1 + j])
            const int code = l < g_.n_shared || j == 0 ? c0 : next[1 + j];
            idx[(size_t) l * P + i] = semantic ? std::clamp(code, 0, CS - 1) : -1;
        }
        int na = 0;
        if (g_.is_semantic(frame[0]))
            for (int k = 2; k < NC; ++k)
                add[(size_t) i * 9 + na++] = k * CS + std::clamp(frame[1 + k], 0, CS - 1);
        add_n[i] = na;
        add_n[P + i] = 0;
    }
    if (!cuda_ok(cudaMemcpyAsync(bidx_, idx, bidx_host_.size() * 4, cudaMemcpyHostToDevice, st), "prefill rows", err) ||
        !cuda_ok(cudaMemcpyAsync(bh_, hidden, (size_t) n * D * 4, cudaMemcpyDeviceToDevice, st), "prefill hidden", err))
        return false;
    const int32_t* const d_add = bidx_ + (size_t) NL * P;
    const int32_t* const d_add_n = d_add + (size_t) 9 * P;
    const drama3::QkvShape qs{g_.n_head, g_.n_kv_head, g_.head_dim, g_.rope_dim, g_.rope_base, eps};
    for (int l = 0; l < NL; ++l) {
        Layer& L = layers_[(size_t) l];
        drama3::embed_rows(L.embedding, bidx_ + (size_t) l * P, d_add, l == 0 ? d_add_n : d_add_n + P,
                           codebook_embeddings_, be_, D, n, st);
        drama3::rms_norm(be_, L.enorm, bcat_, D, eps, st, n, D, 2 * D);
        drama3::rms_norm(bh_, L.hnorm, bcat_ + D, D, eps, st, n, D, 2 * D);
        if (!matvec(L.eh_proj, bcat_, bx_, err, stream_, n))
            return false;
        drama3::rms_norm(bx_, L.attn_norm, bxn_, D, eps, st, n);
        if (!matvec(L.wqkv, bxn_, bqkv_, err, stream_, n))
            return false;
        drama3::qk_norm_rope(bqkv_, L.q_norm, L.k_norm, bq_, L.k_cache, L.v_cache, qs, pos0, st, n);
        for (int i = 0; i < n; ++i) // row i sees positions 0 .. pos0 + i
            if (!drama3::attention(bq_ + (size_t) i * Q, L.k_cache, L.v_cache, bqkv_ + (size_t) i * QKV + Q, part_,
                                   batt_ + (size_t) i * Q, qs, pos0 + i + 1, st)) {
                err = "drama3 tower: unsupported attention shape";
                return false;
            }
        if (!matvec(L.wo, batt_, bproj_, err, stream_, n))
            return false;
        drama3::add(bx_, bproj_, n * D, st);
        drama3::rms_norm(bx_, L.ffn_norm, bxn_, D, eps, st, n);
        if (!matvec(L.w1, bxn_, bff1_, err, stream_, n) || !matvec(L.w3, bxn_, bff3_, err, stream_, n))
            return false;
        try {
            guild::kernels::native_swiglu_quantize_q8_1(bff1_, bff3_, bxq_, g_.ffn, n, stream_);
            guild::kernels::native_mmvq(L.w2.type, L.w2.w, bxq_, bproj_, g_.ffn, D, n, stream_);
        } catch (const std::exception& ex) {
            err = std::string("drama3 tower ffn: ") + ex.what();
            return false;
        }
        drama3::add(bx_, bproj_, n * D, st);
        drama3::rms_norm(bx_, L.final_norm, bh_, D, eps, st, n);
        if (layer_out) // tests: [row][layer][dim]
            for (int i = 0; i < n; ++i)
                if (!cuda_ok(cudaMemcpyAsync(layer_out + ((size_t) i * NL + l) * D, bh_ + (size_t) i * D,
                                             (size_t) D * 4, cudaMemcpyDeviceToHost, st),
                             "layer readback", err))
                    return false;
    }
    // the host arrays are reused by the next chunk: wait for this one
    return cuda_ok(cudaStreamSynchronize(st), "tower prefill", err);
}

bool Drama3Tower::read_kv(int l, int64_t pos, uint16_t* k, uint16_t* v) const {
    if (l < 0 || l >= (int) layers_.size() || pos < 0 || pos >= max_frames_)
        return false;
    const size_t row = (size_t) g_.n_kv_head * g_.head_dim;
    return cudaMemcpy(k, layers_[(size_t) l].k_cache + (size_t) pos * row, row * 2, cudaMemcpyDeviceToHost) ==
               cudaSuccess &&
           cudaMemcpy(v, layers_[(size_t) l].v_cache + (size_t) pos * row, row * 2, cudaMemcpyDeviceToHost) ==
               cudaSuccess;
}

} // namespace guild::core
