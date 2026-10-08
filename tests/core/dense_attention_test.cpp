#include "../check.hpp"
#include "guild/kernels/qsa.hpp"
#include "guild/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

// Half precision to float conversion helper
float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) {
            uint32_t u = sign;
            float f;
            std::memcpy(&f, &u, 4);
            return f;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp += (127 - 15);
    }
    uint32_t u = sign | (static_cast<uint32_t>(exp) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint16_t f32_to_f16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = ((u >> 23) & 0xFF) - (127 - 15);
    uint32_t mant = u & 0x007FFFFF;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

// Independent CPU reference for full causal GQA attention
void cpu_dense_gqa(const float* q, const uint16_t* k_pool, const uint16_t* v_pool,
                   const int32_t* page_table, int64_t n_kv, const guild::kernels::QsaShapes& s,
                   float* out) {
    const int64_t NH = s.n_head;
    const int64_t NKV = s.n_head_kv;
    const int64_t HD = s.head_dim;
    const int64_t G = NH / NKV;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));

    for (int64_t h = 0; h < NH; ++h) {
        const int64_t kvh = h / G;
        const float* q_h = q + h * HD;

        std::vector<float> scores(static_cast<size_t>(n_kv));
        float max_s = -1e30f;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int64_t page = page_table[j / s.page_size];
            const int64_t slot = (page * NKV + kvh) * s.page_size + (j % s.page_size);
            const uint16_t* k_cell = k_pool + slot * HD;

            float dot = 0.0f;
            for (int64_t d = 0; d < HD; ++d) {
                dot += q_h[d] * f16_to_f32(k_cell[d]);
            }
            float sc = dot * scale;
            scores[static_cast<size_t>(j)] = sc;
            if (sc > max_s) max_s = sc;
        }

        float sum_exp = 0.0f;
        for (int64_t j = 0; j < n_kv; ++j) {
            float e = std::exp(scores[static_cast<size_t>(j)] - max_s);
            scores[static_cast<size_t>(j)] = e;
            sum_exp += e;
        }

        float* out_h = out + h * HD;
        for (int64_t d = 0; d < HD; ++d) out_h[d] = 0.0f;

        for (int64_t j = 0; j < n_kv; ++j) {
            const float w = scores[static_cast<size_t>(j)] / sum_exp;
            const int64_t page = page_table[j / s.page_size];
            const int64_t slot = (page * NKV + kvh) * s.page_size + (j % s.page_size);
            const uint16_t* v_cell = v_pool + slot * HD;

            for (int64_t d = 0; d < HD; ++d) {
                out_h[d] += w * f16_to_f32(v_cell[d]);
            }
        }
    }
}

void test_dense_attention_at_length(int64_t n_kv) {
    std::cout << "Testing dense causal attention at n_kv = " << n_kv << "... " << std::flush;

    guild::kernels::QsaShapes s;
    s.n_head = 16;
    s.n_head_kv = 2;
    s.head_dim = 256;
    s.page_size = 4;
    s.idx_n_head = 0; // Dense attention, no sparse indexer

    const int64_t NH = s.n_head;
    const int64_t NKV = s.n_head_kv;
    const int64_t HD = s.head_dim;
    const int64_t n_pages = (n_kv + s.page_size - 1) / s.page_size;
    const int64_t total_slots = n_pages * NKV * s.page_size;

    std::mt19937 rng(42 + static_cast<uint32_t>(n_kv));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> h_q(static_cast<size_t>(NH * HD));
    for (auto& x : h_q) x = dist(rng);

    std::vector<uint16_t> h_k_pool(static_cast<size_t>(total_slots * HD));
    std::vector<uint16_t> h_v_pool(static_cast<size_t>(total_slots * HD));
    for (auto& x : h_k_pool) x = f32_to_f16(dist(rng) * 0.5f);
    for (auto& x : h_v_pool) x = f32_to_f16(dist(rng) * 0.5f);

    std::vector<int32_t> h_page_table(static_cast<size_t>(n_pages));
    for (int64_t i = 0; i < n_pages; ++i) h_page_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);

    // Compute CPU reference
    std::vector<float> ref_attn(static_cast<size_t>(NH * HD), 0.0f);
    cpu_dense_gqa(h_q.data(), h_k_pool.data(), h_v_pool.data(), h_page_table.data(), n_kv, s, ref_attn.data());

    // GPU allocation
    float* d_q = nullptr;
    uint16_t* d_k_pool = nullptr;
    uint16_t* d_v_pool = nullptr;
    int32_t* d_page_table = nullptr;
    int32_t* d_step = nullptr;
    int32_t* d_ids = nullptr;
    float* d_scratch = nullptr;
    float* d_attn = nullptr;

    const int64_t cap = n_kv;
    const uint64_t scratch_floats = guild::kernels::qsa_decode_attn_scratch_floats(cap, s);

    CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_k_pool, h_k_pool.size() * sizeof(uint16_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_v_pool, h_v_pool.size() * sizeof(uint16_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_page_table, h_page_table.size() * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_step, guild::kernels::kStepCount * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_ids, static_cast<size_t>(cap) * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMalloc(&d_scratch, scratch_floats * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_attn, ref_attn.size() * sizeof(float)) == cudaSuccess);

    std::vector<int32_t> h_ids(static_cast<size_t>(cap));
    for (int64_t i = 0; i < cap; ++i) h_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i);

    std::vector<int32_t> h_step(guild::kernels::kStepCount, 0);
    h_step[guild::kernels::kStepPos] = static_cast<int32_t>(n_kv - 1);
    h_step[guild::kernels::kStepNKv] = static_cast<int32_t>(n_kv);
    h_step[guild::kernels::kStepNBid] = static_cast<int32_t>(n_kv / s.page_size);
    h_step[guild::kernels::kStepWidth] = static_cast<int32_t>(guild::kernels::qsa_selection_width(n_kv, s));

    CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_k_pool, h_k_pool.data(), h_k_pool.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_v_pool, h_v_pool.data(), h_v_pool.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_page_table, h_page_table.data(), h_page_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_step, h_step.data(), h_step.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_ids, h_ids.data(), h_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemset(d_attn, 0, ref_attn.size() * sizeof(float)) == cudaSuccess);

    guild::kernels::QsaAttnPools pools;
    pools.k_pool = d_k_pool;
    pools.v_pool = d_v_pool;
    pools.page_table = d_page_table;

    guild::kernels::qsa_decode_attn_batch(d_q, pools, d_ids, d_step, cap, s, d_scratch, d_attn, 1, nullptr);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> gpu_attn(ref_attn.size(), 0.0f);
    CHECK(cudaMemcpy(gpu_attn.data(), d_attn, gpu_attn.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    // Compute max absolute error and relative L2 error
    float max_abs_diff = 0.0f;
    double sum_diff_sq = 0.0;
    double sum_ref_sq = 0.0;

    for (size_t i = 0; i < ref_attn.size(); ++i) {
        float diff = std::abs(gpu_attn[i] - ref_attn[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        sum_diff_sq += diff * diff;
        sum_ref_sq += ref_attn[i] * ref_attn[i];
    }
    double rel_l2 = std::sqrt(sum_diff_sq / (sum_ref_sq + 1e-12));

    std::cout << "max_abs=" << max_abs_diff << ", rel_l2=" << rel_l2 << " -> ";
    CHECK(max_abs_diff <= 2e-3f); // FP16 summation order tolerance
    CHECK(rel_l2 <= 1e-3);
    std::cout << "PASSED\n";

    cudaFree(d_q);
    cudaFree(d_k_pool);
    cudaFree(d_v_pool);
    cudaFree(d_page_table);
    cudaFree(d_step);
    cudaFree(d_ids);
    cudaFree(d_scratch);
    cudaFree(d_attn);
}

} // namespace

int main() {
    std::cout << "============================================================\n";
    std::cout << "GENERIC DENSE CAUSAL ATTENTION NUMERICAL PARITY VERIFICATION\n";
    std::cout << "============================================================\n";

    // 1. Short contexts
    test_dense_attention_at_length(16);
    test_dense_attention_at_length(256);
    test_dense_attention_at_length(1024);

    // 2. Exact boundary tests: 2,048, 2,050, 2,051, 2,052
    test_dense_attention_at_length(2048);
    test_dense_attention_at_length(2050);
    test_dense_attention_at_length(2051); // Old Qwen selection bound
    test_dense_attention_at_length(2052); // Beyond old Qwen selection bound

    // 3. Medium & long contexts: 4K, 8K, 16K, 32K
    test_dense_attention_at_length(4096);
    test_dense_attention_at_length(8192);
    test_dense_attention_at_length(16384);
    test_dense_attention_at_length(32768);

    std::cout << "============================================================\n";
    std::cout << "ALL DENSE CAUSAL ATTENTION NUMERICAL PARITY TESTS PASSED!\n";
    std::cout << "============================================================\n";
    return 0;
}
