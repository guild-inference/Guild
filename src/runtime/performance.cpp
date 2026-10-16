#include "guild/runtime/performance.hpp"
#include "guild/core/validation.hpp"
#include "guild/kernels/iq_kernels.hpp"
#include "guild/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>

namespace guild::runtime {
namespace {

using Clock = std::chrono::steady_clock;
struct CudaFailure : std::runtime_error {
    cudaError_t status;
    explicit CudaFailure(cudaError_t value) : std::runtime_error(cudaGetErrorString(value)), status(value) {}
};
struct Scratch {
    std::vector<void*> buffers;
    ~Scratch() { for (auto p : buffers) cudaFree(p); }
    template<class T> T* alloc(size_t count) {
        T* p = nullptr;
        const auto status = cudaMalloc(&p, count * sizeof(T));
        if (status != cudaSuccess) throw CudaFailure(status);
        buffers.push_back(p);
        return p;
    }
};

void check(cudaError_t status) {
    if (status != cudaSuccess) throw CudaFailure(status);
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

CacheSizing size_expert_cache(const kernels::cpu::ExpertLayout& layout,
                             const std::vector<std::pair<int32_t, int32_t>>& ranking,
                             const std::vector<uint8_t>& gpu_layers, uint64_t budget,
                             int64_t max_slots) {
    CacheSizing result;
    if (max_slots <= 0 || layout.n_layers <= 0 || layout.n_expert <= 0) return result;
    std::vector<uint8_t> seen(size_t(layout.n_layers * layout.n_expert), 0);
    for (const auto& pair : ranking) {
        const auto layer = pair.first, expert = pair.second;
        if (layer < 0 || expert < 0 || layer >= layout.n_layers || expert >= layout.n_expert ||
            size_t(layer) >= gpu_layers.size() || !gpu_layers[size_t(layer)]) continue;
        const auto index = size_t(layer * layout.n_expert + expert);
        if (seen[index]) continue;
        seen[index] = 1;
        const uint64_t bytes = layout.blob_bytes(layer);
        if (!bytes || bytes > uint64_t(INT64_MAX) - 255) continue;
        const auto aligned = (bytes + 255) & ~uint64_t(255);
        if (aligned > budget - result.bytes) continue;
        result.pairs.push_back(pair);
        result.slot_bytes.push_back(int64_t(bytes));
        result.bytes += aligned;
        if (int64_t(result.pairs.size()) >= max_slots) break;
    }
    return result;
}

bool calibrate_experts(const kernels::cpu::ExpertLayout& layout, core::ExpertSource& source,
                       kernels::cpu::ExpertPool& pool, double fraction, void* stream,
                       ExpertPerformance& result, std::string& err) try {
    result = {};
    err.clear();
    if (!std::isfinite(fraction) || (fraction < 0 && fraction != -1) || fraction > 1)
        throw std::invalid_argument("expert fraction must be auto (-1) or 0..1");
    result.gpu_layers.resize(size_t(layout.n_layers), 0);
    result.pcie_numerators.resize(size_t(layout.n_layers), 0);
    if (!layout.native) return true; // no generic GPU kernel for canonical fixed-layout experts
    const auto cs = static_cast<cudaStream_t>(stream);
    for (int64_t l = 0; l < layout.n_layers; ++l) {
        const auto& f = layout.fmt[size_t(l)];
        result.gpu_layers[size_t(l)] = kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff);
    }
    std::map<std::tuple<int, int, int64_t, int64_t>, int> measured;
    for (int64_t layer = 0; layer < layout.n_layers; ++layer) {
        const auto& f = layout.fmt[size_t(layer)];
        const bool supported = kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        result.gpu_layers[size_t(layer)] = supported;
        if (!supported || fraction == 0 || !source.pcie_layer(layer)) continue;
        if (fraction >= 0) {
            result.pcie_numerators[size_t(layer)] = std::clamp(int(fraction * 256 + 0.5), 0, 256);
            continue;
        }
        const auto format = std::make_tuple(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        if (const auto it = measured.find(format); it != measured.end()) {
            result.pcie_numerators[size_t(layer)] = it->second;
            continue;
        }
        constexpr int max_groups = 32; // exceed small LLCs; exercise the real multi-worker schedule
        const int groups = int(std::min<int64_t>(max_groups, layout.n_expert));
        constexpr int tokens = 2;
        std::vector<const uint8_t*> blobs;
        for (int g = 0; g < groups; ++g) {
            const int e = int((int64_t(g) * layout.n_expert / groups + 7) % layout.n_expert);
            if (!source.pinned(layer, e)) continue;
            if (const auto* blob = source.blob(layer, e)) blobs.push_back(blob);
        }
        if (blobs.empty()) continue;
        const int count = int(blobs.size()), entries = count * tokens;
        Scratch scratch;
        auto* weights = scratch.alloc<uint8_t>(size_t(count) * f.bytes);
        auto* x = scratch.alloc<float>(size_t(tokens) * f.n_embd);
        auto* xq = scratch.alloc<uint8_t>(size_t(tokens) * f.n_embd / 32 * 36);
        auto* gpu_out = scratch.alloc<float>(size_t(entries) * f.n_embd);
        auto* work = scratch.alloc<uint8_t>(kernels::native_expert_scratch_bytes(entries, f.n_ff));
        auto* pointers = scratch.alloc<unsigned long long>(count);
        auto* starts = scratch.alloc<int32_t>(count + 1);
        auto* dst = scratch.alloc<int32_t>(entries);
        auto* token_ids = scratch.alloc<int32_t>(entries);
        auto* n_groups = scratch.alloc<int32_t>(1);
        std::vector<float> input(size_t(tokens) * f.n_embd);
        for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 31) - 15) / 16;
        std::vector<uint8_t> activations(size_t(tokens) * kernels::cpu::kNativeActBytes);
        std::vector<kernels::cpu::ActQ> canonical(tokens);
        for (int t = 0; t < tokens; ++t) {
            if (kernels::cpu::q2_native_kernels(f.gu_type))
                kernels::cpu::act_quant_any(input.data() + size_t(t) * f.n_embd, int(f.n_embd), canonical[t]);
            else pool.quantize_native(f, input.data() + size_t(t) * f.n_embd,
                                                activations.data() + size_t(t) * kernels::cpu::kNativeActBytes);
        }
        std::vector<float> cpu_out(size_t(entries) * f.n_embd);
        std::vector<kernels::cpu::ExpertJobMulti> jobs(count);
        std::vector<unsigned long long> ptr(count);
        std::vector<int32_t> offsets(count + 1), destinations(entries), token_index(entries);
        for (int g = 0; g < count; ++g) {
            jobs[g].blob = blobs[g]; jobs[g].nt = tokens;
            ptr[g] = reinterpret_cast<unsigned long long>(weights + size_t(g) * f.bytes);
            offsets[g] = g * tokens;
            for (int t = 0; t < tokens; ++t) {
                jobs[g].nact[t] = activations.data() + size_t(t) * kernels::cpu::kNativeActBytes;
                jobs[g].act[t] = &canonical[t];
                jobs[g].out[t] = cpu_out.data() + size_t(g * tokens + t) * f.n_embd;
                destinations[g * tokens + t] = g * tokens + t;
                token_index[g * tokens + t] = t;
            }
        }
        offsets[count] = entries;
        check(cudaMemcpyAsync(x, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, cs));
        check(cudaMemcpyAsync(pointers, ptr.data(), ptr.size() * sizeof(ptr[0]), cudaMemcpyHostToDevice, cs));
        check(cudaMemcpyAsync(starts, offsets.data(), offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs));
        check(cudaMemcpyAsync(dst, destinations.data(), entries * sizeof(int32_t), cudaMemcpyHostToDevice, cs));
        check(cudaMemcpyAsync(token_ids, token_index.data(), entries * sizeof(int32_t), cudaMemcpyHostToDevice, cs));
        check(cudaMemcpyAsync(n_groups, &count, sizeof(count), cudaMemcpyHostToDevice, cs));
        kernels::quantize_q8_1_rows(x, tokens, f.n_embd, xq, cs);
        const auto gpu_layout = kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        auto copy = [&] {
            for (int g = 0; g < count; ++g)
                check(cudaMemcpyAsync(weights + size_t(g) * f.bytes, blobs[g], f.bytes, cudaMemcpyHostToDevice, cs));
        };
        auto gpu = [&] {
            kernels::native_expert_grouped(gpu_layout, pointers, starts, n_groups, dst, token_ids,
                                          count, entries, xq, work, gpu_out, cs);
        };
        copy(); gpu(); check(cudaStreamSynchronize(cs));
        pool.run_split_multi_native(f, jobs.data(), count);
        std::vector<double> cpu_times, gpu_times, copy_times;
        for (int repeat = 0; repeat < 5; ++repeat) {
            auto start = Clock::now();
            pool.run_split_multi_native(f, jobs.data(), count);
            cpu_times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
            start = Clock::now();
            copy(); check(cudaStreamSynchronize(cs));
            copy_times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
            start = Clock::now();
            gpu(); check(cudaStreamSynchronize(cs));
            gpu_times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        }
        std::vector<float> values(cpu_out.size());
        check(cudaMemcpyAsync(values.data(), gpu_out, values.size() * sizeof(float), cudaMemcpyDeviceToHost, cs));
        check(cudaStreamSynchronize(cs));
        if (!core::validate_finite(cpu_out.data(), cpu_out.size(), "calibration CPU output", err) ||
            !core::validate_finite(values.data(), values.size(), "calibration GPU output", err)) return false;
        const double cpu = median(cpu_times), gpu_ms = median(gpu_times), copy_ms = median(copy_times);
        // Equalize exposed CPU and GPU+link work; all terms are measured on this format and pool.
        const int numerator = std::clamp(int(256 * cpu / (cpu + gpu_ms + copy_ms) + 0.5), 0, 256);
        measured[format] = numerator;
        result.pcie_numerators[size_t(layer)] = numerator;
        result.cpu_ms += cpu; result.gpu_ms += gpu_ms; result.copy_ms += copy_ms;
        ++result.measured_formats;
    }
    return true;
} catch (const CudaFailure& e) {
    if (e.status == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        err.clear(); // calibration is optional when the fixed runtime leaves no scratch space
        return true;
    }
    err = std::string("expert performance calibration: ") + e.what();
    return false;
} catch (const std::exception& e) {
    err = std::string("expert performance calibration: ") + e.what();
    return false;
}

bool tune_expert_split(core::Verifier& verifier, core::SessionState& state, const core::ModelGeometry& geometry,
                       core::ExpertDispatch& dispatch, core::PoolMultiFn pool, void* user, int max_t,
                       void* stream, ExpertPerformance& performance, std::string& err) try {
    if (!performance.measured_formats) return true;
    const auto cs = static_cast<cudaStream_t>(stream);
    const auto initial = performance.pcie_numerators;
    const int rows = std::min(4, max_t);
    std::vector<int32_t> tokens(rows), output(rows);
    for (int t = 0; t < rows; ++t) tokens[t] = int32_t((uint64_t(t) * 104729 + 17) % uint64_t(verifier.vocab()));
    auto trial = [&](const std::vector<int>& numerators, double& elapsed) {
        dispatch.pcie_layer_numerators = numerators;
        if (!verifier.wait_commit(err)) return false;
        core::session_zero(state, geometry, nullptr, cs);
        check(cudaStreamSynchronize(cs));
        const auto start = Clock::now();
        if (!verifier.run(rows, tokens.data(), 0, pool, user, output.data(), err)) return false;
        elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        return true;
    };
    double ignored;
    if (!trial(initial, ignored)) return false; // capture and warm before timing
    std::vector<int> best(initial.size(), 0);
    double best_ms = 0;
    std::vector<double> baseline;
    for (int repeat = 0; repeat < 2; ++repeat) {
        double value;
        if (!trial(best, value)) return false;
        baseline.push_back(value);
    }
    const double cpu_ms = best_ms = median(baseline);
    double chosen_scale = 0;
    for (double scale : {0.5, 0.75, 1.0, 1.25, 1.5}) {
        std::vector<int> candidate(initial.size());
        for (size_t l = 0; l < candidate.size(); ++l) candidate[l] = std::clamp(int(initial[l] * scale + 0.5), 0, 256);
        std::vector<double> times;
        for (int repeat = 0; repeat < 2; ++repeat) {
            double value;
            if (!trial(candidate, value)) return false;
            times.push_back(value);
        }
        const auto value = median(times);
        if (value < best_ms * 0.99) { best_ms = value; best = std::move(candidate); chosen_scale = scale; }
    }
    performance.pcie_numerators = best;
    dispatch.pcie_layer_numerators = best;
    core::session_zero(state, geometry, nullptr, cs);
    check(cudaStreamSynchronize(cs));
    const auto bounds = std::minmax_element(best.begin(), best.end());
    std::fprintf(stderr, "guild performance: complete-window calibration CPU %.2f ms, selected %.2f ms; "
                         "scale %.2f, PCIe %d..%d/256\n", cpu_ms, best_ms, chosen_scale,
                         best.empty() ? 0 : *bounds.first, best.empty() ? 0 : *bounds.second);
    return true;
} catch (const std::exception& e) {
    err = std::string("expert window calibration: ") + e.what();
    return false;
}

} // namespace guild::runtime
