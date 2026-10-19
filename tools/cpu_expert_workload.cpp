// CPU-only expert workload over real GGUF weights; no CUDA calls.
#include "guild/artifact/gguf_reader.hpp"
#include "guild/kernels/cpu/pool.hpp"
#include "guild/kernels/cpu/expert_layout.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cpu = guild::kernels::cpu;
using Clock = std::chrono::steady_clock;
int main(int argc, char** argv) try {
    if (argc != 9)
        throw std::invalid_argument(
            "usage: cpu_expert_workload MODEL LAYER GROUPS TOKENS ITERATIONS CORES CSV REPETITIONS");
    // argv[8] is the number of repetitions (kept explicit for counter runs).
    const int layer = std::stoi(argv[2]), groups = std::stoi(argv[3]), tokens = std::stoi(argv[4]);
    const int iterations = std::stoi(argv[5]), cores = std::stoi(argv[6]), repeats = std::stoi(argv[8]);
    if (groups < 1 || groups > 256 || tokens < 1 || tokens > 8 || iterations < 1 || cores < 1 || repeats < 1)
        throw std::invalid_argument("invalid workload shape");
    guild::GgufModel model(guild::gguf_split_paths(argv[1]));
    const char* roles[] = {"gate", "up", "down"};
    const guild::TensorInfo* tensor[3]{};
    const uint8_t* weights[3]{};
    for (int role = 0; role < 3; ++role) {
        size_t shard = 0;
        tensor[role] = model.find("blk." + std::to_string(layer) + ".ffn_" + roles[role] + "_exps.weight", &shard);
        if (!tensor[role])
            throw std::runtime_error("missing tensor");
        weights[role] = model.shard(shard).tensor_data(*tensor[role]);
    }
    cpu::NativeFmt fmt;
    std::string error;
    if (!cpu::native_fmt(tensor[0]->type, tensor[2]->type, tensor[0]->shape[0], tensor[0]->shape[1], fmt, error))
        throw std::runtime_error(error);
    if (fmt.n_embd > cpu::H || fmt.n_ff > cpu::FF || cpu::q2_native_kernels(fmt.gu_type) ||
        cpu::q2_native_kernels(fmt.d_type))
        throw std::invalid_argument("CPU workload requires native ggml expert formats within the pool's geometry");
    std::vector<uint8_t> blobs(size_t(groups) * fmt.bytes), act(size_t(tokens) * cpu::kNativeActBytes);
    std::vector<float> x(size_t(tokens) * fmt.n_embd), out(size_t(groups) * tokens * fmt.n_embd);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = float(int((i * 17) % 101) - 50) / 51.f;
    for (int t = 0; t < tokens; ++t)
        cpu::native_quant_act(fmt, x.data() + size_t(t) * fmt.n_embd, act.data() + size_t(t) * cpu::kNativeActBytes);
    std::vector<cpu::ExpertJobMulti> jobs(groups);
    for (int group = 0; group < groups; ++group) {
        const int expert = group;
        auto* b = blobs.data() + size_t(group) * fmt.bytes;
        std::memcpy(b, weights[0] + size_t(expert) * fmt.up_off, fmt.up_off);
        std::memcpy(b + fmt.up_off, weights[1] + size_t(expert) * fmt.up_off, fmt.up_off);
        std::memcpy(b + fmt.down_off, weights[2] + size_t(expert) * (fmt.bytes - fmt.down_off),
                    fmt.bytes - fmt.down_off);
        jobs[group].blob = b;
        jobs[group].nt = tokens;
        for (int t = 0; t < tokens; ++t) {
            jobs[group].nact[t] = act.data() + size_t(t) * cpu::kNativeActBytes;
            jobs[group].out[t] = out.data() + size_t(group * tokens + t) * fmt.n_embd;
        }
    }
    cpu::ExpertPool pool(cores == 1 ? 1 : cores - 1, true, cores != 1, cpu::PoolAffinity::All);
    auto previous = cpu::pin_current_thread(pool.host_core());
    struct Restore {
        cpu::ThreadAffinity& p;
        ~Restore() {
            cpu::restore_thread_affinity(p);
        }
    } restore{previous};
    auto run = [&] { pool.run_split_multi_native(fmt, jobs.data(), groups, tokens > 1); };
    for (int i = 0; i < 10; ++i)
        run();
    std::ofstream csv(argv[7]);
    if (!csv)
        throw std::runtime_error("cannot open CSV");
    csv << "run,layer,groups,tokens,cores,iterations,weight_bytes_per_iteration,wall_ms,avx512bw\n"
        << std::setprecision(12);
    for (int r = 0; r < repeats; ++r) {
        auto start = Clock::now();
        for (int i = 0; i < iterations; ++i)
            run();
        const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        csv << r << ',' << layer << ',' << groups << ',' << tokens << ',' << cores << ',' << iterations << ','
            << uint64_t(groups) * fmt.bytes << ',' << elapsed << ',' << cpu::cpu_avx512bw_ok() << '\n';
        csv.flush();
    }
    std::cout << "CPU expert workload complete; checksum " << out.front() << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
