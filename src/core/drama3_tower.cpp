// src/core/drama3_tower.cpp - Drama 3's codebook tower (include/guild/core/drama3_tower.hpp).
#include "guild/core/drama3_tower.hpp"

#include "guild/core/drama3_tower_kernels.hpp"
#include "guild/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
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
    for (void* p :
         {(void*) h_, (void*) e_, (void*) cat_, (void*) x_, (void*) xn_, (void*) qkv_, (void*) q_, (void*) att_,
          (void*) proj_, (void*) ff1_, (void*) ff3_, (void*) logits_, (void*) part_, xq_, dense_})
        cudaFree(p);
    for (void* p : {(void*) bh_, (void*) be_, (void*) bcat_, (void*) bx_, (void*) bxn_, (void*) bqkv_, (void*) bq_,
                    (void*) batt_, (void*) bproj_, (void*) bff1_, (void*) bff3_, bxq_, (void*) bidx_})
        cudaFree(p);
    if (host_logits_)
        cudaFreeHost(host_logits_);
    if (stream_)
        cudaStreamDestroy((cudaStream_t) stream_);
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
    const uint64_t size = (uint64_t) bin.tellg();
    std::vector<char> host(size);
    bin.seekg(0);
    bin.read(host.data(), (std::streamsize) size);
    if (!bin) {
        err = "drama3 tower: short read of dense.bin";
        return false;
    }
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
            return Mat{find(name, "q8_0", rows, cols), rows, cols};
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
    } bufs[] = {{&h_, (size_t) D},
                {&e_, (size_t) D},
                {&cat_, (size_t) 2 * D},
                {&x_, (size_t) D},
                {&xn_, (size_t) D},
                {&qkv_, (size_t) (2 * Q + 2 * KV)},
                {&q_, (size_t) Q},
                {&att_, (size_t) Q},
                {&proj_, (size_t) D},
                {&ff1_, (size_t) g_.ffn},
                {&ff3_, (size_t) g_.ffn},
                {&logits_, (size_t) CS + 1},
                {&part_, drama3::attn_part_floats(qs, max_frames)}};
    for (const Buf& b : bufs) {
        if (!cuda_ok(cudaMalloc((void**) b.p, b.n * 4), "scratch", err))
            return false;
        vram_ += b.n * 4;
    }
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
    return true;
}

bool Drama3Tower::matvec(const Mat& m, const float* x, float* y, std::string& err, int ncols) {
    void* xq = ncols > 1 ? bxq_ : xq_;
    try {
        guild::kernels::native_quantize_q8_1(x, xq, m.cols, ncols, stream_);
        guild::kernels::native_mmvq(kQ8_0, m.w, xq, y, m.cols, m.rows, ncols, stream_);
    } catch (const std::exception& e) {
        err = std::string("drama3 tower matvec: ") + e.what();
        return false;
    }
    return true;
}

bool Drama3Tower::layer_forward(Layer& L, const float* e, int64_t pos, std::string& err) {
    const auto st = (cudaStream_t) stream_;
    const int D = g_.dim, Q = g_.n_head * g_.head_dim;
    const float eps = g_.norm_eps;
    drama3::rms_norm(e, L.enorm, cat_, D, eps, st);
    drama3::rms_norm(h_, L.hnorm, cat_ + D, D, eps, st);
    if (!matvec(L.eh_proj, cat_, x_, err))
        return false;
    drama3::rms_norm(x_, L.attn_norm, xn_, D, eps, st);
    if (!matvec(L.wqkv, xn_, qkv_, err))
        return false;
    const drama3::QkvShape qs{g_.n_head, g_.n_kv_head, g_.head_dim, g_.rope_dim, g_.rope_base, eps};
    drama3::qk_norm_rope(qkv_, L.q_norm, L.k_norm, q_, L.k_cache, L.v_cache, qs, pos, st);
    if (!drama3::attention(q_, L.k_cache, L.v_cache, qkv_ + Q, part_, att_, qs, pos + 1, st)) {
        err = "drama3 tower: unsupported attention shape";
        return false;
    }
    if (!matvec(L.wo, att_, proj_, err))
        return false;
    drama3::add(x_, proj_, D, st);
    drama3::rms_norm(x_, L.ffn_norm, xn_, D, eps, st);
    if (!matvec(L.w1, xn_, ff1_, err) || !matvec(L.w3, xn_, ff3_, err))
        return false;
    try {
        guild::kernels::native_swiglu_quantize_q8_1(ff1_, ff3_, xq_, g_.ffn, 1, stream_);
        guild::kernels::native_mmvq(kQ8_0, L.w2.w, xq_, proj_, g_.ffn, D, 1, stream_);
    } catch (const std::exception& ex) {
        err = std::string("drama3 tower ffn: ") + ex.what();
        return false;
    }
    drama3::add(x_, proj_, D, st);
    drama3::rms_norm(x_, L.final_norm, h_, D, eps, st);
    return cuda_ok(cudaGetLastError(), "layer launch", err);
}

// The reference's sample_top_k_top_p (fish_omni/tower.py): logits / T sorted descending; drop rank >= top_k and
// every token whose cumulative probability (over the full softmax) passes top_p, except the first; draw from the
// softmax of the rest.  The draw is the engine's (inverse CDF of one uniform), not torch's exponential race.
int Drama3Tower::sample(const float* logits, int n, const Drama3Sampling& sp, std::string& err) {
    const float temp = std::max(sp.temperature, 1e-5f);
    float mx = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(logits[i])) {
            err = "drama3 tower: non-finite logits; refusing to sample invalid audio";
            return -1;
        }
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
    const uint64_t bits = splitmix64(sp.seed * 0x100000001b3ull + draws_++);
    const double u = (double) (bits >> 11) * (1.0 / 9007199254740992.0) * tot;
    double acc = 0.0;
    for (int r = 0; r < kept; ++r) {
        acc += p[(size_t) r];
        if (u < acc)
            return idx[(size_t) r];
    }
    return idx[(size_t) kept - 1];
}

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
    const auto st = (cudaStream_t) stream_;
    const int D = g_.dim, CS = g_.codebook_size;
    if (!cuda_ok(cudaMemcpyAsync(h_, hidden, (size_t) D * 4, cudaMemcpyDeviceToDevice, st), "hidden copy", err))
        return false;

    // the main token: eos or <|semantic:c0|>, over eos + the band
    int32_t main = 0;
    if (!forced || cand_logits) {
        if (!matvec(cand_, h_, logits_, err) || !read_logits(CS + 1, cand_logits, err))
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
    if (!step_layers(0, split, err) || !cuda_ok(cudaStreamSynchronize(st), "tower step", err))
        return false;
    out[0] = main;
    out[1] = cur_.codes[0];
    if (g_.n_codebooks > 1)
        out[2] = cur_.codes[1];
    return true;
}

bool Drama3Tower::step_finish(std::string& err) {
    if (!step_layers(cur_.next_layer, (int) layers_.size(), err) ||
        !cuda_ok(cudaStreamSynchronize((cudaStream_t) stream_), "tower step", err))
        return false;
    for (int k = 2; cur_.out && k < g_.n_codebooks; ++k)
        cur_.out[1 + k] = cur_.codes[k];
    return true;
}

bool Drama3Tower::read_logits(int n, float* also, std::string& err) {
    const auto st = (cudaStream_t) stream_;
    if (!cuda_ok(cudaMemcpyAsync(host_logits_, logits_, (size_t) n * 4, cudaMemcpyDeviceToHost, st), "logits", err) ||
        !cuda_ok(cudaStreamSynchronize(st), "tower step", err))
        return false;
    if (also)
        std::memcpy(also, host_logits_, (size_t) n * 4);
    return true;
}

bool Drama3Tower::step_layers(int l0, int l1, std::string& err) {
    const auto st = (cudaStream_t) stream_;
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
        drama3::embed(layers_[(size_t) l].embedding, c.semantic ? std::clamp(code, 0, CS - 1) : -1, unseen,
                      codebook_embeddings_, e_, D, st);
        if (!layer_forward(layers_[(size_t) l], e_, c.pos, err))
            return false;
        if (c.layer_out &&
            !cuda_ok(cudaMemcpyAsync(c.layer_out + (size_t) l * D, h_, (size_t) D * 4, cudaMemcpyDeviceToHost, st),
                     "layer readback", err))
            return false;
        c.next_layer = l + 1;
        if (shared)
            continue;
        // a teacher-forced step takes the next code from `forced`, and runs the head only when its logits are asked for
        if ((!c.forced || c.code_logits) &&
            (!matvec(heads_[(size_t) j], h_, logits_, err) ||
             !read_logits(CS, c.code_logits ? c.code_logits + (size_t) j * CS : nullptr, err)))
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
        if (!matvec(L.eh_proj, bcat_, bx_, err, n))
            return false;
        drama3::rms_norm(bx_, L.attn_norm, bxn_, D, eps, st, n);
        if (!matvec(L.wqkv, bxn_, bqkv_, err, n))
            return false;
        drama3::qk_norm_rope(bqkv_, L.q_norm, L.k_norm, bq_, L.k_cache, L.v_cache, qs, pos0, st, n);
        for (int i = 0; i < n; ++i) // row i sees positions 0 .. pos0 + i
            if (!drama3::attention(bq_ + (size_t) i * Q, L.k_cache, L.v_cache, bqkv_ + (size_t) i * QKV + Q, part_,
                                   batt_ + (size_t) i * Q, qs, pos0 + i + 1, st)) {
                err = "drama3 tower: unsupported attention shape";
                return false;
            }
        if (!matvec(L.wo, batt_, bproj_, err, n))
            return false;
        drama3::add(bx_, bproj_, n * D, st);
        drama3::rms_norm(bx_, L.ffn_norm, bxn_, D, eps, st, n);
        if (!matvec(L.w1, bxn_, bff1_, err, n) || !matvec(L.w3, bxn_, bff3_, err, n))
            return false;
        try {
            guild::kernels::native_swiglu_quantize_q8_1(bff1_, bff3_, bxq_, g_.ffn, n, stream_);
            guild::kernels::native_mmvq(kQ8_0, L.w2.w, bxq_, bproj_, g_.ffn, D, n, stream_);
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
