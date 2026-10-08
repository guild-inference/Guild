#include "../check.hpp"
#include "../../src/runtime/model_impl.hpp"
#include "guild/server/engine.hpp"
#include "guild/server/native_engine.hpp"
#include "guild/models/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;
using namespace guild;

std::vector<int32_t> token_ids(const std::string& path) {
    std::ifstream file(path);
    CHECK(file.is_open());
    std::string text((std::istreambuf_iterator<char>(file)), {});
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream input(text);
    std::vector<int32_t> ids;
    int64_t id;
    while (input >> id) { CHECK(id >= 0 && id <= INT32_MAX); ids.push_back(int32_t(id)); }
    CHECK(input.eof() && !ids.empty());
    return ids;
}

std::vector<float> read_logits(const std::string& path, int64_t vocab) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open());
    std::vector<float> logits(size_t(vocab), 0);
    CHECK(bool(file.read(reinterpret_cast<char*>(logits.data()), std::streamsize(logits.size() * sizeof(float)))));
    CHECK(file.peek() == std::char_traits<char>::eof());
    std::string err;
    CHECK(core::validate_finite(logits.data(), logits.size(), "reference", err));
    return logits;
}

void write_logits(const fs::path& path, const std::vector<float>& logits) {
    std::ofstream file(path, std::ios::binary);
    CHECK(bool(file.write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * sizeof(float)))));
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: qwen_runtime_parity_test MODEL PACK IDS REFERENCE_LOGITS REFERENCE_ENGINE PROFILE OUT_DIR\n";
        return 2;
    }
    // Use the same declared CPU arithmetic in both executables. The default 2
    // is the shipped dispatch rule; 1 can be requested for window-size parity.
    const std::string mt_min = std::getenv("GUILD_IQ_MT_MIN") ? std::getenv("GUILD_IQ_MT_MIN") : "2";
    setenv("GUILD_IQ_MT_MIN", mt_min.c_str(), 1);
    setenv("STRATA_IQ_MT_MIN", mt_min.c_str(), 1);
    const auto ids = token_ids(argv[3]);
    const fs::path out_dir(argv[7]);
    fs::create_directories(out_dir);

    // Run the pre-extraction engine first and release its GPU/host allocations
    // before loading the candidate. Both receive the same supplied token IDs.
    server::GuildProcessEngineOptions ro;
    ro.executable = fs::absolute(argv[5]).string();
    ro.tokenizer_dir = std::string(argv[2]) + "/tokenizer";
    ro.args = {"--pack", argv[2], "--native", argv[1], "--expert-profile", argv[6],
               "--expert-cache", "off", "--prefill", "512", "--spec", "2", "--suffix-draft", "0",
               "--pcie-frac", "0", "--max-context", "2048", "--kv", "fp16", "--kv-host-only",
               "--resident-budget-gib", "56", "--prompt-cache", "0", "--short-read", "0"};
    server::GuildProcessEngine reference(ro);
    CHECK(reference.start());
    server::InferenceRequest request;
    request.prompt_tokens = ids;
    request.max_tokens = 8;
    server::GenerationResult expected;
    CHECK(reference.generate(request, expected));
    CHECK(expected.completion_tokens > 0 && !expected.tokens.empty());
    reference.stop();

    server::NativeInferenceEngineOptions opts;
    opts.paths.primary_model_path = argv[1];
    opts.paths.pack_dir = argv[2];
    opts.paths.tokenizer_dir = std::string(argv[2]) + "/tokenizer";
    opts.paths.expert_profile_path = argv[6];
    opts.plan.context_length = 2048;
    opts.plan.mtp_spec_tokens = 2;
    opts.plan.kv_mode = memory::KvMode::HostOnly;
    opts.plan.prefill_chunk = 512;
    server::NativeInferenceEngine candidate(opts);
    std::string err;
    if (!candidate.init(err)) { std::cerr << err << '\n'; return 1; }
    auto* m = candidate.model()->impl();
    CHECK(m->ple_required && m->ss.ple.ready());
    CHECK(m->ple_table.bytes_read() == 0);
    CHECK(candidate.model()->descriptor().attn.vocab_size == m->ver->vocab());
    const auto wanted = read_logits(argv[4], m->ver->vocab());

    // Compare the matching batched-prompt/one-row-target configuration to the
    // independently built old engine's full vocabulary, not just its argmax.
    request.max_tokens = 1;
    server::GenerationResult first;
    CHECK(candidate.generate(request, first));
    CHECK(first.completion_tokens == 1);
    std::vector<float> got(wanted.size());
    CHECK(m->ver->copy_logits(0, got.data()));
    write_logits(out_dir / "candidate-first-logits.bin", got);
    double max_abs = 0, squared_error = 0, squared_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double difference = double(got[i]) - wanted[i];
        max_abs = std::max(max_abs, std::abs(difference));
        squared_error += difference * difference;
        squared_ref += double(wanted[i]) * wanted[i];
    }
    const double relative_l2 = std::sqrt(squared_error / std::max(squared_ref, 1e-30));
    const bool bitwise = std::memcmp(got.data(), wanted.data(), got.size() * sizeof(float)) == 0;
    std::cout << std::setprecision(9) << "reference logits: bitwise=" << bitwise << " max_abs=" << max_abs
              << " relative_l2=" << relative_l2 << '\n';
    CHECK(max_abs <= 1e-5 && relative_l2 <= 1e-6);
    CHECK(first.tokens[0].token_id == expected.tokens[0].token_id);
    CHECK(m->ple_table.bytes_read() > 0);
    CHECK(m->ss.ple_prev[0] == ids[ids.size() - 2] && m->ss.ple_prev[1] == ids.back());
    std::vector<float> history(size_t(kernels::NG_HIST * kernels::NG_HC_DIM));
    CHECK(cudaMemcpy(history.data(), m->ss.ple_hist, history.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(core::validate_finite(history.data(), history.size(), "PLE history", err));
    CHECK(std::any_of(history.begin(), history.end(), [](float v) { return v != 0; }));

    request.max_tokens = 8;
    for (int repeat = 0; repeat < 2; ++repeat) {
        server::GenerationResult result;
        CHECK(candidate.generate(request, result));
        CHECK(result.tokens.size() == expected.tokens.size());
        for (size_t i = 0; i < result.tokens.size(); ++i) CHECK(result.tokens[i].token_id == expected.tokens[i].token_id);
        std::cout << "matched greedy continuation " << repeat << ": " << result.text << '\n';
    }

    // Match the per-token control's arithmetic for the PLE ablation below.
    core::session_zero(m->ss, m->g, nullptr, m->main_stream);
    CHECK(cudaStreamSynchronize(m->main_stream) == cudaSuccess);
    int32_t control_pick = -1;
    for (size_t p = 0; p < ids.size(); ++p) {
        CHECK(m->ver->run(1, &ids[p], int64_t(p), runtime::drive_pool_multi, &m->drive, &control_pick, err));
        CHECK(m->ver->commit(1, err));
    }
    std::vector<float> sequential_ple(got.size());
    CHECK(m->ver->copy_logits(0, sequential_ple.data()));
    write_logits(out_dir / "diagnostic-sequential-ple-logits.bin", sequential_ple);

    // A fresh, explicitly ablated verifier is a negative control. Existing
    // graphs bake in the operation, so merely toggling a pointer is not an ablation.
    auto* table = m->ss.ple.table;
    {
        core::Verifier ablated;
        core::VerifyHits hits;
        hits.d_res = m->d_res;
        hits.h_res = m->host_res.data();
        hits.blob = int64_t(kernels::cpu::expert_layout().max_blob);
        m->ss.ple.table = nullptr;
        CHECK(ablated.init(m->wt, m->g, m->ss, hits, &m->native_head, 2, err));
        m->drive.d.plan = ablated.plan_sink();
        core::session_zero(m->ss, m->g, nullptr, m->main_stream);
        CHECK(cudaStreamSynchronize(m->main_stream) == cudaSuccess);
        int32_t pick = -1;
        for (size_t p = 0; p < ids.size(); ++p) {
            CHECK(ablated.run(1, &ids[p], int64_t(p), runtime::drive_pool_multi, &m->drive, &pick, err));
            CHECK(ablated.commit(1, err));
        }
        std::vector<float> no_ple(got.size());
        CHECK(ablated.copy_logits(0, no_ple.data()));
        write_logits(out_dir / "diagnostic-no-ple-logits.bin", no_ple);
        double ablation_delta = 0;
        for (size_t i = 0; i < got.size(); ++i) ablation_delta = std::max(ablation_delta, std::abs(double(no_ple[i]) - sequential_ple[i]));
        std::cout << "PLE ablation max_abs_delta=" << ablation_delta << '\n';
        CHECK(ablation_delta > 1e-3);
    }
    m->ss.ple.table = table;
    m->drive.d.plan = m->ver->plan_sink();
    // Disconnection through the production session must be a failure, not the
    // diagnostic ablation above. No token may be emitted.
    m->ss.ple.table = nullptr;
    int emitted = 0;
    server::GenerationResult rejected;
    CHECK(!candidate.generate_stream(request, [&](const auto&) { ++emitted; return true; }, rejected));
    CHECK(emitted == 0 && rejected.error_message.find("required PLE") != std::string::npos);
    CHECK(!candidate.is_ready());
    m->ss.ple.table = table;

    std::ofstream report(out_dir / "parity.txt");
    report << "reference_engine_sha256=" << models::Sha256::hash_file(ro.executable) << '\n'
           << "ids_sha256=" << models::Sha256::hash_file(argv[3]) << '\n'
           << "iq_mt_min=" << mt_min << '\n'
           << "vocab=" << wanted.size() << " bitwise=" << bitwise << " max_abs=" << max_abs
           << " relative_l2=" << relative_l2 << '\n' << "greedy_ids=";
    for (const auto& token : expected.tokens) report << token.token_id << ',';
    report << "\nPLE_connected_and_ablation_detected=1\n";
    std::puts("qwen_runtime_parity_test: real Qwen logits, greedy continuation, PLE state and failure checks PASS");
}
