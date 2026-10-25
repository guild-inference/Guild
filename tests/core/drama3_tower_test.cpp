// tests/core/drama3_tower_test.cpp - Drama3Tower against the reference's own numbers.
//
//     drama3_tower_test <tower dir> <reference dir>
//
// The reference dir holds raw arrays from fish-speech's modules run in float32 (meta.txt names them and their shapes):
// the prompt rows, the frames it sampled, the backbone's final-norm hidden state at every position, and per decode
// step the main token's 4097 candidate logits, the 13 tower layers' outputs and the 9 heads' logits.  The prompt goes
// through the batched path (Drama3Tower::prefill), whose K/V must equal one step per row's bit for bit.  The tower is
// fed the reference's hidden states and teacher-forced with its frames, so each step sees exactly the reference's
// inputs; what differs is only the tower's own arithmetic (Q8_0 weights, Q8_1 activations, FP16 K/V).  Pass: every
// compared vector has cosine >= 0.999 to the reference, and the sampling distributions (softmax at temperature 1) are
// close: mean KL(reference || tower) <= 0.01 nats and the worst row <= 0.1.  (The argmax is printed, not judged: rows
// of near-equal logits, common in the late codebooks, flip it under any rounding.)
#include "guild/core/drama3_tower.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Array {
    std::vector<int64_t> shape;
    std::vector<char> data;
    template <typename T> const T* as() const {
        return reinterpret_cast<const T*>(data.data());
    }
};

bool read_dir(const std::string& dir, std::map<std::string, Array>& out, std::string& reason) {
    std::ifstream meta(dir + "/meta.txt");
    if (!meta)
        return false;
    std::string line;
    while (std::getline(meta, line)) {
        std::istringstream ls(line);
        std::string name;
        ls >> name;
        if (name == "reason") {
            ls >> reason;
            continue;
        }
        Array a;
        int64_t d = 0, n = 1;
        while (ls >> d) {
            a.shape.push_back(d);
            n *= d;
        }
        std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
        a.data.resize((size_t) n * 4);
        if (!f.read(a.data.data(), (std::streamsize) a.data.size()))
            return false;
        out[name] = std::move(a);
    }
    return true;
}

double cosine(const float* a, const float* b, int n) {
    double ab = 0, aa = 0, bb = 0;
    for (int i = 0; i < n; ++i) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / std::sqrt(aa * bb + 1e-30);
}

double kl(const float* ref, const float* got, int n) {
    double mr = -1e30, mg = -1e30;
    for (int i = 0; i < n; ++i) {
        mr = std::max(mr, (double) ref[i]);
        mg = std::max(mg, (double) got[i]);
    }
    double zr = 0, zg = 0;
    for (int i = 0; i < n; ++i) {
        zr += std::exp(ref[i] - mr);
        zg += std::exp(got[i] - mg);
    }
    double d = 0;
    for (int i = 0; i < n; ++i) {
        const double lr = ref[i] - mr - std::log(zr), lg = got[i] - mg - std::log(zg);
        d += std::exp(lr) * (lr - lg);
    }
    return d;
}

int argmax(const float* a, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (a[i] > a[best])
            best = i;
    return best;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: drama3_tower_test <tower dir> <reference dir>\n");
        return 2;
    }
    std::map<std::string, Array> ref;
    std::string reason;
    if (!read_dir(argv[2], ref, reason)) {
        std::printf("SKIP: no reference vectors in %s\n", argv[2]);
        return 77;
    }
    guild::core::Drama3Tower tower;
    std::string err;
    if (!tower.load(argv[1], 4096, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    const auto& g = tower.geometry();
    const int D = g.dim, W = g.width(), CS = g.codebook_size, NL = g.n_shared + g.n_codebooks - 1;
    const Array &prompt = ref["prompt"], &frames = ref["frames"], &hidden = ref["hidden"];
    const Array &cand = ref["cand_logits"], &layers = ref["tower_h"], &heads = ref["code_logits"];
    const int R = (int) prompt.shape[0], n = (int) frames.shape[0], steps = (int) cand.shape[0];
    std::printf("prompt %d rows, %d frames (%s), %d decode steps recorded\n", R, n, reason.c_str(), steps);
    float* h_dev = nullptr;
    cudaMalloc(&h_dev, (size_t) D * 4);
    std::vector<int32_t> out((size_t) W);
    guild::core::Drama3Sampling sp;
    // cudaMemcpy from pageable memory may return before the bytes land; the tower reads on its own stream, so wait
    auto upload = [](float* dst, const float* src, size_t floats) {
        cudaMemcpy(dst, src, floats * 4, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    };
    auto run = [&](guild::core::Drama3Tower& t, int p, const int32_t* row, const int32_t* forced, float* c, float* cl,
                   float* lo) {
        upload(h_dev, hidden.as<float>() + (size_t) p * D, (size_t) D);
        if (!t.step(h_dev, p, row, forced, nullptr, 0, sp, out.data(), err, c, cl, lo)) {
            std::fprintf(stderr, "step %d: %s\n", p, err.c_str());
            std::exit(1);
        }
    };
    // The prompt through the batched path, and through one step per row on a second tower: their K/V must be the
    // same bits (the multi-column MMVQ equals a one-column call; the attention runs per row either way).
    {
        float* hp = nullptr;
        cudaMalloc(&hp, (size_t) R * D * 4);
        upload(hp, hidden.as<float>(), (size_t) R * D);
        if (!tower.prefill(hp, 0, R - 1, prompt.as<int32_t>(), err)) {
            std::fprintf(stderr, "prefill: %s\n", err.c_str());
            return 1;
        }
        cudaFree(hp);
        guild::core::Drama3Tower rowwise;
        if (!rowwise.load(argv[1], 4096, err)) {
            std::fprintf(stderr, "load: %s\n", err.c_str());
            return 1;
        }
        for (int p = 0; p + 1 < R; ++p)
            run(rowwise, p, prompt.as<int32_t>() + (size_t) p * W, prompt.as<int32_t>() + (size_t) (p + 1) * W, nullptr,
                nullptr, nullptr);
        const int KV = g.n_kv_head * g.head_dim;
        std::vector<uint16_t> ka(KV), va(KV), kb(KV), vb(KV);
        int bad = 0;
        for (int l = 0; l < NL; ++l)
            for (int p = 0; p + 1 < R; ++p) {
                tower.read_kv(l, p, ka.data(), va.data());
                rowwise.read_kv(l, p, kb.data(), vb.data());
                bad += ka != kb || va != vb;
            }
        std::printf("prompt K/V, batched against one step per row: %d of %d (layer, position) rows differ\n", bad,
                    NL * (R - 1));
        if (bad) {
            std::printf("FAIL\n");
            return 1;
        }
    }
    std::vector<float> c((size_t) CS + 1), cl((size_t) (g.n_codebooks - 1) * CS), lo((size_t) NL * D);
    double worst_cos = 1.0;
    int argmax_rows = 0, argmax_same = 0;
    double kl_sum = 0, kl_worst = 0;
    auto add_kl = [&](const float* ref_row, const float* got, int n) {
        const double d = kl(ref_row, got, n);
        kl_sum += d;
        kl_worst = std::max(kl_worst, d);
    };
    std::vector<double> layer_cos((size_t) NL, 1.0);
    const int checked = std::min(steps, n); // a step whose frame is eos has no forced codes to replay
    for (int s = 0; s < checked; ++s) {
        const int p = R - 1 + s;
        const int32_t* row =
            s == 0 ? prompt.as<int32_t>() + (size_t) (R - 1) * W : frames.as<int32_t>() + (size_t) (s - 1) * W;
        run(tower, p, row, frames.as<int32_t>() + (size_t) s * W, c.data(), cl.data(), lo.data());
        const float* rc = cand.as<float>() + (size_t) s * (CS + 1);
        worst_cos = std::min(worst_cos, cosine(c.data(), rc, CS + 1));
        argmax_same += argmax(c.data(), CS + 1) == argmax(rc, CS + 1);
        add_kl(rc, c.data(), CS + 1);
        ++argmax_rows;
        for (int l = 0; l < NL; ++l) {
            const double cs = cosine(lo.data() + (size_t) l * D, layers.as<float>() + ((size_t) s * NL + l) * D, D);
            layer_cos[(size_t) l] = std::min(layer_cos[(size_t) l], cs);
            worst_cos = std::min(worst_cos, cs);
        }
        for (int j = 0; j + 1 < g.n_codebooks; ++j) {
            const float* a = cl.data() + (size_t) j * CS;
            const float* b = heads.as<float>() + ((size_t) s * (g.n_codebooks - 1) + j) * CS;
            worst_cos = std::min(worst_cos, cosine(a, b, CS));
            argmax_same += argmax(a, CS) == argmax(b, CS);
            add_kl(b, a, CS);
            ++argmax_rows;
        }
    }
    for (int l = 0; l < NL; ++l)
        std::printf("layer %2d: worst cosine %.6f\n", l, layer_cos[(size_t) l]);
    const double agree = argmax_rows ? (double) argmax_same / argmax_rows : 0.0;
    const double kl_mean = argmax_rows ? kl_sum / argmax_rows : 0.0;
    std::printf("%d steps: worst cosine %.6f; KL mean %.5f, worst %.5f nats; argmax agreement %.3f (%d rows)\n",
                checked, worst_cos, kl_mean, kl_worst, agree, argmax_rows);
    const bool pass = checked > 0 && worst_cos >= 0.999 && kl_mean <= 0.01 && kl_worst <= 0.1;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
