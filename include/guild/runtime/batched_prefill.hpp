#pragma once

#include "guild/core/layout.hpp"
#include "guild/core/weights.hpp"
#include "guild/core/session.hpp"
#include "guild/core/expert_source.hpp"
#include "guild/kernels/cpu/pool.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace guild::runtime {

struct BatchedPrefillStats {
    int64_t prompt_tokens = 0;
    int64_t chunks_processed = 0;
    double ms_total = 0.0;
    double ms_embed = 0.0;
    double ms_mixer = 0.0;
    double ms_moe = 0.0;
    double tok_per_sec = 0.0;
    double ms_cpu_gu = 0, ms_cpu_quant = 0, ms_cpu_down = 0;
    double gpu_dense_ms = 0, gpu_gdn_ms = 0, gpu_attention_ms = 0;
    double cpu_group_ms = 0, host_wait_ms = 0;
    uint64_t expert_bytes = 0, activation_bytes = 0, device_bytes = 0;
    int64_t expert_groups = 0, gpu_groups = 0;
};

class BatchedPrefill {
public:
    BatchedPrefill();
    ~BatchedPrefill();

    BatchedPrefill(const BatchedPrefill&) = delete;
    BatchedPrefill& operator=(const BatchedPrefill&) = delete;

    // Initializes device and host buffers for batched prefill
    bool init(const core::WeightTable& wt,
              const core::ModelGeometry& g,
              core::SessionState& ss,
              core::ExpertSource& src,
              kernels::cpu::ExpertPool& pool,
              void* stream,
              std::string& err, int64_t requested_batch = 0,
              const core::ExpertCache* cache = nullptr);

    // Ingests prompt tokens [pos0, pos0 + n) in batches of batch_size_
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    void set_batch_size(int64_t b);
    void set_expert_cache(const core::ExpertCache* cache);
    int64_t batch_size() const { return batch_size_; }

    const BatchedPrefillStats& stats() const { return stats_; }

    std::function<bool()> should_stop;
    // Diagnostic only: device rows [T,N], routing [T,K]; stream is synchronized.
    std::function<void(int64_t, int64_t, int64_t, const float*, const int32_t*, const float*)> on_layer;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int64_t batch_size_ = 32;
    BatchedPrefillStats stats_;
};

} // namespace guild::runtime
