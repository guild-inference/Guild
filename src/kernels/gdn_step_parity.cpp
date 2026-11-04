// src/kernels/gdn_step_parity.cpp - the verify windows' GDN recurrence (gdn_step_norm_multi without n_keep) and the
// commit (with n_keep) against the kernels before them, bitwise.
//
//     GUILD_GDN_STEP_COLS=0 build/gdn_step_parity --dump ref.bin   # the kernels before
//     GUILD_GDN_STEP_COLS=1 build/gdn_step_parity --compare ref.bin # the column kernels (the CUDA default)
//     [GUILD_GDN_STEP_COLS=0] build/gdn_step_parity --bench        # a window's calls at the model's shape
//     build/gdn_step_parity --profile                              # plain launches of those calls, for ncu
//
// WHY IT EXISTS.  The verify recurrence and the commit split a head's 128 state columns over four blocks
// (gdn_multi.cu, gdn_step_cols_kernel; the verify output's gated RMSNorm in gdn_out_norm_kernel), and each says its
// outputs and the state it writes are bitwise those of gdn_step_norm_multi_kernel and gdn_step_commit_kernel.  The
// switch is an environment variable read once per process, so the two runs are two processes, joined by a file
// (registered as two tests with a fixture).  Each run also checks, on its own kernels, that a T-token verify call
// computes for token t what one-token windows committed one at a time compute (verify == greedy decode), that a
// verify call leaves the state as it was, and that it writes no output outside its tokens.
//
// THE CASES.  The model's shape (16 key heads, 48 value heads: the kernel's constant-head-count build) and three
// others (h_k = h_v, more key heads, a small one), every token count from 1 to 8 with outputs from token 0, the
// middle, the last and none, both output gates (sigmoid and SiLU); the commit with n_keep 0 to T for T = 1..8.  The
// data is random at the magnitudes of the model's tensors, with a few zeros.
#include "guild/kernels/gdn_multi.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace guild::kernels;

constexpr int S = 128;
constexpr int MAX_T = 8;
constexpr uint32_t SENTINEL = 0x7fc0beefu;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

struct Shape {
    int h_k, h_v;
};
const Shape SHAPES[] = {{16, 48}, {8, 8}, {32, 32}, {2, 4}};

template <class T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T)), "cudaMalloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

/// The recorded outputs: appended as raw 32-bit words, each case tagged so a mismatch names it.
struct Record {
    std::vector<uint32_t> words;
    std::vector<std::pair<std::string, size_t>> marks; // case name, first word
    void mark(const std::string& name) {
        marks.emplace_back(name, words.size());
    }
    void add(const float* p, size_t n) {
        const size_t o = words.size();
        words.resize(o + n);
        std::memcpy(words.data() + o, p, n * 4);
    }
};

struct GdnCase {
    int h_k, h_v, C, VD;
    float *state, *state0, *h, *gate, *beta, *z, *gamma, *y;
    int32_t* keep;
    std::vector<float> state_h;
};

GdnCase make_case(const Shape& sh, std::mt19937& rng) {
    GdnCase c{};
    c.h_k = sh.h_k;
    c.h_v = sh.h_v;
    c.C = 2 * S * sh.h_k + S * sh.h_v;
    c.VD = S * sh.h_v;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    std::vector<float> st((size_t) S * c.h_v * S), h((size_t) MAX_T * c.C), g((size_t) MAX_T * c.h_v),
        b((size_t) MAX_T * c.h_v), z((size_t) MAX_T * c.VD), gm(S);
    for (float& v : st)
        v = 0.05f * nd(rng);
    for (int t = 0; t < MAX_T; ++t) {
        // q and k rows L2-normalized per head, as the conv leaves them; v about unit scale
        for (int hd = 0; hd < 2 * c.h_k; ++hd) {
            float ss = 0.0f;
            float* p = h.data() + (size_t) t * c.C + (size_t) hd * S;
            for (int i = 0; i < S; ++i) {
                p[i] = nd(rng);
                ss += p[i] * p[i];
            }
            const float sc = 1.0f / std::sqrt(ss);
            for (int i = 0; i < S; ++i)
                p[i] *= sc;
        }
        for (int i = 2 * S * c.h_k; i < c.C; ++i)
            h[(size_t) t * c.C + i] = nd(rng);
        for (int i = 0; i < c.h_v; ++i) {
            g[(size_t) t * c.h_v + i] = -3.0f * ud(rng);
            b[(size_t) t * c.h_v + i] = ud(rng);
        }
        for (int i = 0; i < c.VD; ++i)
            z[(size_t) t * c.VD + i] = 2.0f * nd(rng);
    }
    for (int i = 0; i < S; ++i)
        gm[i] = 1.0f + 0.1f * nd(rng);
    for (int i = 0; i < 64; ++i)
        st[rng() % st.size()] = 0.0f;
    st[7] = -0.0f;
    h[(size_t) 2 * S * c.h_k + 3] = 0.0f; // a zero v
    c.state_h = st;
    c.state = dev(st);
    c.state0 = dev(st);
    c.h = dev(h);
    c.gate = dev(g);
    c.beta = dev(b);
    c.z = dev(z);
    c.gamma = dev(gm);
    ck(cudaMalloc(&c.y, (size_t) MAX_T * c.VD * 4), "cudaMalloc y");
    ck(cudaMalloc(&c.keep, 4), "cudaMalloc keep");
    return c;
}

void reset_state(GdnCase& c) {
    ck(cudaMemcpy(c.state, c.state0, c.state_h.size() * 4, cudaMemcpyDeviceToDevice), "reset state");
}

void fill_sentinel(float* d, size_t n) {
    std::vector<uint32_t> s(n, SENTINEL);
    ck(cudaMemcpy(d, s.data(), n * 4, cudaMemcpyHostToDevice), "sentinel");
}

int g_fail = 0;
void fail(const std::string& what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++g_fail;
}

void run_gdn(Record& rec, GdnCase& c, cudaStream_t cs) {
    const float eps = 1e-6f;
    const std::string sh = "h_k " + std::to_string(c.h_k) + " h_v " + std::to_string(c.h_v);
    // verify calls: the state must come back untouched, nothing written outside [tb, T)
    for (int silu = 0; silu < 2; ++silu)
        for (int T = 1; T <= MAX_T; ++T) {
            const int tbs[] = {0, T / 2, T - 1, T};
            for (int k = 0; k < 4; ++k) {
                const int tb = tbs[k];
                if (k > 0 && tb == tbs[k - 1])
                    continue;
                reset_state(c);
                fill_sentinel(c.y, (size_t) MAX_T * c.VD);
                gdn_step_norm_multi(c.state, c.h, c.C, c.gate, c.beta, c.z, c.gamma, eps, c.y, c.h_k, c.h_v, T, nullptr,
                                    cs, tb, silu != 0);
                ck(cudaStreamSynchronize(cs), "verify call");
                const std::string name =
                    sh + " verify T " + std::to_string(T) + " tb " + std::to_string(tb) + (silu ? " silu" : " sigmoid");
                const std::vector<float> y = host(c.y, (size_t) MAX_T * c.VD);
                const std::vector<float> st = host(c.state, c.state_h.size());
                if (std::memcmp(st.data(), c.state_h.data(), st.size() * 4) != 0)
                    fail(name + ": the state changed");
                for (size_t i = 0; i < y.size(); ++i) {
                    uint32_t u;
                    std::memcpy(&u, &y[i], 4);
                    const int t = (int) (i / c.VD);
                    if ((t < tb || t >= T) != (u == SENTINEL)) {
                        fail(name + ": output " + std::to_string(i) + (u == SENTINEL ? " not written" : " written"));
                        break;
                    }
                }
                rec.mark(name);
                rec.add(y.data() + (size_t) tb * c.VD, (size_t) (T - tb) * c.VD);
            }
        }
    // commit calls: T = the graph's token count, n_keep 0..T
    for (int T = 1; T <= MAX_T; ++T)
        for (int n = 0; n <= T; ++n) {
            reset_state(c);
            ck(cudaMemcpy(c.keep, &n, 4, cudaMemcpyHostToDevice), "keep");
            gdn_step_norm_multi(c.state, c.h, c.C, c.gate, c.beta, c.z, c.gamma, eps, c.y, c.h_k, c.h_v, T, c.keep, cs,
                                T, false);
            ck(cudaStreamSynchronize(cs), "commit call");
            const std::vector<float> st = host(c.state, c.state_h.size());
            rec.mark(sh + " commit T " + std::to_string(T) + " n_keep " + std::to_string(n));
            rec.add(st.data(), st.size());
        }
    // verify == greedy: a T-token verify's output for token t against t one-token commits and a one-token verify
    for (int silu = 0; silu < 2; ++silu) {
        reset_state(c);
        fill_sentinel(c.y, (size_t) MAX_T * c.VD);
        gdn_step_norm_multi(c.state, c.h, c.C, c.gate, c.beta, c.z, c.gamma, eps, c.y, c.h_k, c.h_v, MAX_T, nullptr, cs,
                            0, silu != 0);
        const std::vector<float> yw = host(c.y, (size_t) MAX_T * c.VD);
        const int one = 1;
        ck(cudaMemcpy(c.keep, &one, 4, cudaMemcpyHostToDevice), "keep");
        for (int t = 0; t < MAX_T; ++t) {
            const size_t ho = (size_t) t * c.C, go = (size_t) t * c.h_v, zo = (size_t) t * c.VD;
            gdn_step_norm_multi(c.state, c.h + ho, c.C, c.gate + go, c.beta + go, c.z + zo, c.gamma, eps, c.y + zo,
                                c.h_k, c.h_v, 1, nullptr, cs, 0, silu != 0);
            gdn_step_norm_multi(c.state, c.h + ho, c.C, c.gate + go, c.beta + go, c.z + zo, c.gamma, eps, c.y, c.h_k,
                                c.h_v, MAX_T, c.keep, cs, MAX_T, false);
        }
        const std::vector<float> y1 = host(c.y, (size_t) MAX_T * c.VD);
        if (std::memcmp(yw.data(), y1.data(), yw.size() * 4) != 0)
            fail(sh + (silu ? " silu" : " sigmoid") + ": an 8-token verify differs from one-token windows");
    }
    reset_state(c);
}

Record run_all() {
    cudaStream_t cs;
    ck(cudaStreamCreate(&cs), "stream");
    std::mt19937 rng(1234);
    Record rec;
    for (const Shape& sh : SHAPES) {
        GdnCase c = make_case(sh, rng);
        run_gdn(rec, c, cs);
        for (float* p : {c.state, c.state0, c.h, c.gate, c.beta, c.z, c.gamma, c.y})
            cudaFree(p);
        cudaFree(c.keep);
    }
    cudaStreamDestroy(cs);
    return rec;
}

/// A window's GDN calls at the model's shape over 36 layers' states (113 MB, as in the engine), in a graph.
void bench() {
    const Shape sh{16, 48};
    const int L = 36;
    std::mt19937 rng(7);
    GdnCase c = make_case(sh, rng);
    std::vector<float*> states(L);
    for (float*& p : states) {
        ck(cudaMalloc(&p, c.state_h.size() * 4), "cudaMalloc");
        ck(cudaMemcpy(p, c.state0, c.state_h.size() * 4, cudaMemcpyDeviceToDevice), "copy");
    }
    cudaStream_t cs;
    ck(cudaStreamCreate(&cs), "stream");
    struct Shp {
        const char* name;
        int T, tb, keep; // keep < 0: a verify call
    };
    const Shp shapes[] = {{"verify T1 (1 token, 1 out)", 1, 0, -1},     {"verify T2 g0 (1 token, 1 out)", 1, 0, -1},
                          {"verify T4 g0 (2 tokens, 2 out)", 2, 0, -1}, {"verify T4 g1 (4 tokens, 2 out)", 4, 2, -1},
                          {"verify T3 g1 (3 tokens, 2 out)", 3, 1, -1}, {"commit MT 4, n_keep 1", 4, 4, 1},
                          {"commit MT 4, n_keep 2", 4, 4, 2},           {"commit MT 4, n_keep 4", 4, 4, 4}};
    for (const Shp& s : shapes) {
        if (s.keep >= 0)
            ck(cudaMemcpy(c.keep, &s.keep, 4, cudaMemcpyHostToDevice), "keep");
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "capture");
        for (int l = 0; l < L; ++l)
            gdn_step_norm_multi(states[l], c.h, c.C, c.gate, c.beta, c.z, c.gamma, 1e-6f, c.y, sh.h_k, sh.h_v, s.T,
                                s.keep >= 0 ? c.keep : nullptr, cs, s.tb, true);
        ck(cudaStreamEndCapture(cs, &graph), "end capture");
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
        // the first shape also brings the GPU's clocks up (~0.4 s of launches)
        const int warm = &s == shapes ? 400 : 5;
        for (int i = 0; i < warm; ++i)
            ck(cudaGraphLaunch(exec, cs), "launch");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int reps = 50;
        cudaEventRecord(e0, cs);
        for (int i = 0; i < reps; ++i)
            ck(cudaGraphLaunch(exec, cs), "launch");
        cudaEventRecord(e1, cs);
        ck(cudaEventSynchronize(e1), "sync");
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        std::printf("%-34s %7.2f us a call (%d layers in a graph)\n", s.name, 1000.0f * ms / reps / L, L);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
    }
}

/// Plain launches of the window's shapes over 36 layers' states, for a kernel profiler.
void profile_calls() {
    const Shape sh{16, 48};
    const int L = 36;
    std::mt19937 rng(7);
    GdnCase c = make_case(sh, rng);
    std::vector<float*> states(L);
    for (float*& p : states) {
        ck(cudaMalloc(&p, c.state_h.size() * 4), "cudaMalloc");
        ck(cudaMemcpy(p, c.state0, c.state_h.size() * 4, cudaMemcpyDeviceToDevice), "copy");
    }
    cudaStream_t cs;
    ck(cudaStreamCreate(&cs), "stream");
    const int two = 2;
    ck(cudaMemcpy(c.keep, &two, 4, cudaMemcpyHostToDevice), "keep");
    for (int l = 0; l < L; ++l)
        gdn_step_norm_multi(states[l], c.h, c.C, c.gate, c.beta, c.z, c.gamma, 1e-6f, c.y, sh.h_k, sh.h_v, 1, nullptr,
                            cs, 0, true);
    for (int l = 0; l < L; ++l)
        gdn_step_norm_multi(states[l], c.h, c.C, c.gate, c.beta, c.z, c.gamma, 1e-6f, c.y, sh.h_k, sh.h_v, 4, nullptr,
                            cs, 2, true);
    for (int l = 0; l < L; ++l)
        gdn_step_norm_multi(states[l], c.h, c.C, c.gate, c.beta, c.z, c.gamma, 1e-6f, c.y, sh.h_k, sh.h_v, 4, c.keep,
                            cs, 4, true);
    ck(cudaStreamSynchronize(cs), "profile calls");
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "--selftest";
    if (mode == "--bench") {
        bench();
        return 0;
    }
    if (mode == "--profile") {
        profile_calls();
        return 0;
    }
    const Record rec = run_all();
    if (mode == "--dump" && argc > 2) {
        FILE* f = std::fopen(argv[2], "wb");
        if (!f || std::fwrite(rec.words.data(), 4, rec.words.size(), f) != rec.words.size()) {
            std::fprintf(stderr, "cannot write %s\n", argv[2]);
            return 2;
        }
        std::fclose(f);
        std::printf("gdn_step_parity: %zu cases, %zu words written to %s%s\n", rec.marks.size(), rec.words.size(),
                    argv[2], g_fail ? " (with self-check failures)" : "");
        return g_fail ? 1 : 0;
    }
    if (mode == "--compare" && argc > 2) {
        FILE* f = std::fopen(argv[2], "rb");
        std::vector<uint32_t> ref(rec.words.size());
        if (!f || std::fread(ref.data(), 4, ref.size(), f) != ref.size() || std::fgetc(f) != EOF) {
            std::fprintf(stderr, "cannot read %s, or its size differs\n", argv[2]);
            return 2;
        }
        std::fclose(f);
        int bad = 0;
        for (size_t k = 0; k < rec.marks.size(); ++k) {
            const size_t a = rec.marks[k].second, b = k + 1 < rec.marks.size() ? rec.marks[k + 1].second : ref.size();
            for (size_t i = a; i < b; ++i)
                if (ref[i] != rec.words[i]) {
                    float u, v;
                    std::memcpy(&u, &ref[i], 4);
                    std::memcpy(&v, &rec.words[i], 4);
                    std::fprintf(stderr, "DIFF %s: word %zu: %.9g against %.9g\n", rec.marks[k].first.c_str(), i - a, v,
                                 u);
                    ++bad;
                    break;
                }
        }
        std::printf("gdn_step_parity: %zu cases, %d differ from %s; self-checks %s\n", rec.marks.size(), bad, argv[2],
                    g_fail ? "FAILED" : "passed");
        return (bad || g_fail) ? 1 : 0;
    }
    std::printf("gdn_step_parity: %zu cases; self-checks %s\n", rec.marks.size(), g_fail ? "FAILED" : "passed");
    return g_fail ? 1 : 0;
}
