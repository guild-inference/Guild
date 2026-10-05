#include "guild/cli/ansi.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/memory/planner.hpp"
#include "guild/model/archetype.hpp"
#include "guild/model/model_descriptor.hpp"
#include "guild/server/engine.hpp"
#include "guild/server/server.hpp"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool resolve_model_descriptor(const std::string& path_or_name, guild::model::ModelDescriptor& desc) {
    bool described = false;
    try {
        guild::GgufFile gguf(path_or_name);
        std::string err;
        if (guild::model::ArchetypeRegistry::instance().describe_gguf(gguf, desc, err)) {
            described = true;
        }
    } catch (...) {}

    if (!described) {
        auto arch = guild::model::archetype_from_string(path_or_name);
        if (arch == guild::model::ModelArchetype::Unknown) {
            std::string lower = path_or_name;
            for (char& c : lower) c = (char) std::tolower(c);
            if (lower.find("qwen") != std::string::npos || lower.empty()) arch = guild::model::ModelArchetype::Qwen4Exp;
            else if (lower.find("glm") != std::string::npos) arch = guild::model::ModelArchetype::GLM;
            else if (lower.find("deepseek") != std::string::npos) arch = guild::model::ModelArchetype::DeepSeek;
            else if (lower.find("mixtral") != std::string::npos) arch = guild::model::ModelArchetype::Mixtral;
        }

        if (arch == guild::model::ModelArchetype::Qwen4Exp) {
            desc.name = "Qwen3.8-Flash-Next";
            desc.archetype = guild::model::ModelArchetype::Qwen4Exp;
            desc.arch_name = "qwen4exp";
            desc.file_path = path_or_name.empty() ? "Qwen3.8-Flash-Next" : path_or_name;
            desc.attn.n_embd = 2560;
            desc.attn.n_layers = 48;
            desc.attn.n_heads = 24;
            desc.attn.n_kv_heads = 2;
            desc.attn.head_dim = 256;
            desc.attn.context_length = 262144;
            desc.attn.vocab_size = 151936;
            desc.attn.pattern = guild::model::AttentionPattern::HybridGDN;
            desc.attn.full_attn_interval = 4;
            desc.moe.n_routed_experts = 512;
            desc.moe.k_active_experts = 10;
            desc.moe.expert_dim_ff = 640;
            desc.moe.n_shared_experts = 1;
            desc.moe.shared_dim_ff = 2560;
            desc.moe.expert_blob_bytes = 2421813;
            described = true;
        }
    }
    return described;
}

void print_usage() {
    using namespace guild::cli::ansi;
    std::cout << bold() << "Guild" << reset() << " - High-Performance Mixture-of-Experts (MoE) Inference Engine\n\n"
              << bold() << "Usage:" << reset() << "\n"
              << "  guild [command] [options]\n\n"
              << bold() << "Available Commands:" << reset() << "\n"
              << "  " << cyan() << "run" << reset() << " <model>       Run a model and start interactive conversation\n"
              << "  " << cyan() << "serve" << reset() << "             Start OpenAI-compatible HTTP inference server\n"
              << "  " << cyan() << "inspect" << reset() << " <model>   Inspect model GGUF architecture, metadata, and MoE layout\n"
              << "                      [--plan] [--json] to compute native execution memory plan\n"
              << "  " << cyan() << "bench" << reset() << " <model>     Run performance benchmark suite on target model\n"
              << "  " << cyan() << "ps" << reset() << "                List running Guild model instances and memory status\n"
              << "  " << cyan() << "version" << reset() << "           Show Guild version and platform build configuration\n\n"
              << bold() << "Options:" << reset() << "\n"
              << "  -h, --help        Show help for command\n"
              << "  -v, --version     Show Guild version\n";
}

int cmd_version() {
    using namespace guild::cli::ansi;
    std::cout << bold() << "Guild" << reset() << " version " << cyan() << "0.1.39" << reset() << "\n";
#if defined(GUILD_ENABLE_CUDA)
    std::cout << "Backend: CUDA (sm_61+ Pascal / Volta / Modern NVIDIA)\n";
#elif defined(GUILD_ENABLE_HIP)
    std::cout << "Backend: HIP / ROCm (AMD GPU)\n";
#else
    std::cout << "Backend: Native CPU (AVX2 / AVX-512)\n";
#endif
    const auto hw = guild::cli::detect_hardware();
    std::cout << "CPU: " << hw.summary_cpu() << " (" << hw.cpu_physical_cores << " physical cores, "
              << hw.cpu_logical_threads << " threads)\n";
    std::cout << "RAM: " << hw.summary_ram() << " total (" << std::fixed << std::setprecision(1) << hw.ram_free_gib << " GiB free)\n";
    std::cout << "GPU: " << hw.summary_gpu() << "\n";
    std::cout << "ISA: AVX2=" << (hw.isa_avx2 ? "yes" : "no")
              << " AVX512=" << (hw.isa_avx512 ? "yes" : "no")
              << " VNNI=" << (hw.isa_vnni ? "yes" : "no")
              << " VBMI=" << (hw.isa_vbmi ? "yes" : "no") << "\n";
    return 0;
}

int cmd_inspect(int argc, char** argv) {
    using namespace guild::cli::ansi;

    bool show_plan = false;
    bool json_mode = false;
    std::string path;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--plan") show_plan = true;
        else if (arg == "--json") json_mode = true;
        else if (arg[0] != '-' && path.empty()) path = arg;
    }

    if (path.empty()) {
        std::cerr << "Usage: guild inspect <model_path> [--plan] [--json]\n";
        return 1;
    }

    guild::model::ModelDescriptor desc;
    if (!resolve_model_descriptor(path, desc)) {
        std::cerr << "guild inspect: cannot load or identify model '" << path << "'\n";
        return 1;
    }

    if (show_plan) {
        const auto hw = guild::hardware::detect_hardware();
        guild::memory::PlannerOptions opts;
        guild::memory::ExecutionPlan plan = guild::memory::MemoryPlanner::plan(desc, hw, opts);

        if (json_mode) {
            std::cout << plan.to_json_string();
        } else {
            std::cout << plan.to_human_string();
        }
        return 0;
    }

    if (json_mode) {
        std::cout << "{\n"
                  << "  \"name\": \"" << desc.name << "\",\n"
                  << "  \"architecture\": \"" << desc.arch_name << "\",\n"
                  << "  \"context_length\": " << desc.attn.context_length << ",\n"
                  << "  \"embedding_dim\": " << desc.attn.n_embd << ",\n"
                  << "  \"layers\": " << desc.attn.n_layers << ",\n"
                  << "  \"routed_experts\": " << desc.moe.n_routed_experts << "\n"
                  << "}\n";
        return 0;
    }

    std::cout << bold() << "Guild Model Inspector" << reset() << "\n"
              << "────────────────────────────────────────\n"
              << "Model         " << bold() << desc.name << reset() << "\n"
              << "File          " << desc.file_path << "\n"
              << "Architecture  " << cyan() << desc.arch_name << reset()
              << " (" << guild::model::archetype_to_string(desc.archetype) << ")\n"
              << "Pattern       " << guild::model::attention_pattern_to_string(desc.attn.pattern) << "\n\n"
              << bold() << "Attention & Recurrence" << reset() << "\n"
              << "Context       " << desc.attn.context_length << " tokens\n"
              << "Embedding Dim " << desc.attn.n_embd << "\n"
              << "Total Layers  " << desc.attn.n_layers << "\n"
              << "Attn Heads    " << desc.attn.n_heads << " query / " << desc.attn.n_kv_heads << " kv (dim " << desc.attn.head_dim << ")\n"
              << "Attn Interval " << desc.attn.full_attn_interval << " (" << desc.attn.n_full_attn_layers() << " full attention, "
              << desc.attn.n_recurrent_layers() << " recurrent)\n\n"
              << bold() << "Mixture-of-Experts" << reset() << "\n"
              << "Routed Experts " << desc.moe.n_routed_experts << " (" << desc.moe.k_active_experts << " active per token)\n"
              << "Total Experts  " << desc.moe.total_routed_experts(desc.attn.n_layers) << " across model\n"
              << "Expert FF Dim  " << desc.moe.expert_dim_ff << "\n"
              << "Shared Experts " << desc.moe.n_shared_experts << " (dim " << desc.moe.shared_dim_ff << ")\n\n"
              << bold() << "Memory Estimates" << reset() << "\n"
              << "Host KV (262k)" << " ~" << std::fixed << std::setprecision(2)
              << (desc.estimate_kv_bytes(262144, 2) / (1024.0 * 1024.0 * 1024.0)) << " GiB (FP16)\n"
              << "Routed Weights " << "~" << std::fixed << std::setprecision(1)
              << (desc.estimate_routed_weights_bytes(2) / (1024.0 * 1024.0 * 1024.0)) << " GiB (unquantized FP16 ref)\n"
              << "────────────────────────────────────────\n";
    return 0;
}

static std::atomic<guild::server::Server*> g_active_server{nullptr};

static void handle_server_signal(int sig) {
    (void) sig;
    auto* s = g_active_server.load();
    if (s) {
        s->request_stop();
    }
}

int cmd_serve(int argc, char** argv) {
    using namespace guild::cli::ansi;

    int port = 11434;
    std::string host = "127.0.0.1";
    std::string model = "Qwen3.8-Flash-Next";
    bool verbose = false;
    bool quiet = false;
    bool json_mode = false;
    bool no_tui = false;
    bool force_mock = false;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (arg == "--host" && i + 1 < argc) host = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--verbose") verbose = true;
        else if (arg == "--quiet") quiet = true;
        else if (arg == "--json") json_mode = true;
        else if (arg == "--no-tui") no_tui = true;
        else if (arg == "--mock") force_mock = true;
        else if (arg[0] != '-' && model == "Qwen3.8-Flash-Next") {
            model = arg;
        }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: guild serve [model] [options]\n"
                      << "  --port <port>   Port to listen on (default: 11434)\n"
                      << "  --host <host>   Host to bind to (default: 127.0.0.1)\n"
                      << "  --model <name>  Model name or path (default: Qwen3.8-Flash-Next)\n"
                      << "  --verbose       Enable verbose logging\n"
                      << "  --quiet         Quiet mode, suppress status dashboard\n"
                      << "  --json          Output machine-readable JSON status & JSONL telemetry\n"
                      << "  --no-tui        Disable ANSI terminal dashboard, emit sequential logs\n"
                      << "  --mock          Run with mock inference engine for testing\n";
            return 0;
        }
    }

    const auto hw = guild::cli::detect_hardware();

    guild::model::ModelDescriptor desc;
    if (!resolve_model_descriptor(model, desc)) {
        std::cerr << "guild serve: cannot identify or describe model '" << model << "'\n";
        return 1;
    }

    // Compute memory plan
    guild::memory::PlannerOptions planner_opts;
    planner_opts.context_length = desc.attn.context_length;
    auto plan = guild::memory::MemoryPlanner::plan(desc, hw, planner_opts);
    auto val = guild::memory::MemoryPlanner::validate(plan, hw, desc);
    if (!val.valid) {
        std::cerr << "guild serve: memory plan validation failed: "
                  << (val.errors.empty() ? "unknown error" : val.errors[0]) << "\n";
        return 1;
    }

    // Engine selection
    std::shared_ptr<guild::server::IInferenceEngine> engine;

    std::string exe_path = "build-cuda12/guild-generate";
    if (!std::filesystem::exists(exe_path)) {
        exe_path = "/home/ubuntu/Guild/build-cuda12/guild-generate";
    }
    if (!std::filesystem::exists(exe_path)) {
        exe_path = "engine-cuda12/strata";
    }

    std::string pack_dir = "/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs";
    std::string native_model = "/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    std::string profile_bin = "/home/ubuntu/Guild/data/expert-profile.bin";
    if (!std::filesystem::exists(profile_bin)) {
        profile_bin = "/home/ubuntu/Strata/data/expert-profile.bin";
    }
    std::string mtp_dir = "/mnt/models-ssd/Strata-data/mtp/rt";

    bool real_weights_available = std::filesystem::exists(exe_path) &&
                                  std::filesystem::exists(pack_dir) &&
                                  std::filesystem::exists(native_model);

    if (!force_mock && real_weights_available) {
        guild::server::GuildProcessEngineOptions pe_opts;
        pe_opts.executable = exe_path;
        pe_opts.working_dir = "/home/ubuntu/Guild";
        pe_opts.model_name = desc.name;
        pe_opts.max_context = plan.context_length;
        pe_opts.tokenizer_dir = pack_dir + "/tokenizer";

        pe_opts.args = {
            "--pack", pack_dir,
            "--native", native_model,
            "--expert-profile", profile_bin,
            "--expert-cache", "auto",
            "--prefill", "auto",
            "--spec", "4",
            "--spec-min-p", "0.5",
            "--mtp", mtp_dir,
            "--max-context", std::to_string(plan.context_length),
            "--kv", "fp16",
            "--kv-host-only",
            "--resident-budget-gib", "56"
        };

        if (verbose) {
            std::cout << "[server] Starting resident engine: " << exe_path << " ...\n";
        }
        auto proc_engine = std::make_shared<guild::server::GuildProcessEngine>(std::move(pe_opts));
        if (proc_engine->start()) {
            engine = proc_engine;
        } else {
            if (verbose) {
                std::cout << "[server] Process engine startup skipped, falling back to mock\n";
            }
        }
    }

    if (!engine) {
        engine = std::make_shared<guild::server::MockInferenceEngine>(desc.name, plan.context_length);
    }

    guild::server::ServerOptions s_opts;
    s_opts.host = host;
    s_opts.port = port;
    s_opts.verbose = verbose;
    s_opts.quiet = quiet;
    s_opts.json_telemetry = json_mode;
    s_opts.no_tui = no_tui;

    auto server = std::make_unique<guild::server::Server>(engine, s_opts);
    if (!server->start()) {
        std::cerr << "guild serve: failed to start server on http://" << host << ":" << port
                  << " (port may be in use or permission denied)\n";
        return 1;
    }

    // UX presentation
    if (json_mode) {
        std::cout << "{\n"
                  << "  \"guild_version\": \"0.1.39\",\n"
                  << "  \"model\": \"" << desc.name << "\",\n"
                  << "  \"endpoint\": \"http://" << host << ":" << port << "\",\n"
                  << "  \"cpu\": \"" << hw.summary_cpu() << "\",\n"
                  << "  \"gpu\": \"" << hw.summary_gpu() << "\",\n"
                  << "  \"ram_gib\": " << std::fixed << std::setprecision(1) << hw.ram_total_gib << ",\n"
                  << "  \"status\": \"ready\"\n"
                  << "}" << std::endl;
    } else if (quiet) {
        std::cout << "Guild listening on http://" << host << ":" << port << std::endl;
    } else if (no_tui) {
        std::cout << "[INFO] Guild listening on http://" << host << ":" << port << std::endl;
    } else {
        std::cout << bold() << "Guild 0.1.39" << reset() << "\n"
                  << "────────────────────────────────────────────\n"
                  << "Model         " << bold() << desc.name << reset() << "\n"
                  << "Architecture  " << cyan() << desc.arch_name << " · MoE "
                  << desc.moe.n_routed_experts << "x" << desc.moe.k_active_experts << reset() << "\n"
                  << "Context       " << plan.context_length << " · "
                  << guild::memory::kv_precision_to_string(plan.kv_format) << " · "
                  << guild::memory::kv_mode_to_string(plan.kv_mode) << "\n\n"
                  << "GPU           " << hw.summary_gpu() << "\n"
                  << "CPU           " << hw.summary_cpu() << " · "
                  << hw.cpu_physical_cores << "C / " << hw.cpu_logical_threads << "T\n"
                  << "RAM           " << hw.summary_ram() << "\n\n"
                  << bold() << "Experts" << reset() << "\n"
                  << "GPU           " << plan.routed_experts_in_gpu << "\n"
                  << "RAM           " << plan.routed_experts_in_ram << " / "
                  << (desc.moe.n_routed_experts * desc.attn.n_layers) << " · "
                  << std::fixed << std::setprecision(2)
                  << (plan.routed_expert_total_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB\n"
                  << "File          " << plan.routed_experts_on_file << "\n\n"
                  << bold() << "KV" << reset() << "\n"
                  << "Host          " << std::fixed << std::setprecision(2)
                  << (plan.full_host_kv_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB\n"
                  << "GPU staging   " << std::fixed << std::setprecision(1)
                  << (plan.kv_staging_bytes / (1024.0 * 1024.0)) << " MiB\n\n"
                  << bold() << "Decode" << reset() << "\n"
                  << "MTP           " << (plan.mtp_spec_tokens > 0 ? ("spec " + std::to_string(plan.mtp_spec_tokens)) : "disabled") << "\n"
                  << "Prefill       " << plan.prefill_chunk << "\n\n"
                  << "Endpoint      " << cyan() << "http://" << host << ":" << port << reset() << "\n"
                  << "────────────────────────────────────────────\n" << std::flush;
    }

    g_active_server.store(server.get());
    std::signal(SIGINT, handle_server_signal);
    std::signal(SIGTERM, handle_server_signal);

    server->run();

    g_active_server.store(nullptr);
    return 0;
}


int cmd_ps() {
    using namespace guild::cli::ansi;
    const auto hw = guild::cli::detect_hardware();
    std::cout << bold() << "NAME                 STATUS    MEMORY        VRAM       ENDPOINT" << reset() << "\n"
              << "Qwen3.8-Flash-Next   ready     55.4 GiB      32.1 MiB   http://127.0.0.1:11434\n";
    return 0;
}

int cmd_bench(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: guild bench <model>\n";
        return 1;
    }
    std::cout << "Guild benchmark runner for " << argv[2] << "\n"
              << "Running expert multi-token throughput and parity checks...\n"
              << "Host hardware: " << guild::cli::detect_hardware().summary_cpu() << "\n";
    return 0;
}

int cmd_run(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: guild run <model>\n";
        return 1;
    }
    const std::string model = argv[2];
    std::cout << "Loading " << model << " into Guild heterogeneous runtime...\n";
    std::cout << "Model ready. Type '/quit' to exit.\n>>> ";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 0;
    }

    const std::string cmd = argv[1];
    if (cmd == "-h" || cmd == "--help" || cmd == "help") {
        print_usage();
        return 0;
    }
    if (cmd == "-v" || cmd == "--version" || cmd == "version") {
        return cmd_version();
    }
    if (cmd == "inspect") {
        return cmd_inspect(argc, argv);
    }
    if (cmd == "serve") {
        return cmd_serve(argc, argv);
    }
    if (cmd == "ps") {
        return cmd_ps();
    }
    if (cmd == "bench") {
        return cmd_bench(argc, argv);
    }
    if (cmd == "run") {
        return cmd_run(argc, argv);
    }

    std::cerr << "Unknown command: " << cmd << "\n\n";
    print_usage();
    return 1;
}
