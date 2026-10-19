#include "guild/abi/gkernel_package.hpp"
#include "guild/artifact/gpack.hpp"
#include "guild/cli/ansi.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/gexec/gexec_contract.hpp"
#include "guild/kernels/registry.hpp"
#include "guild/memory/planner.hpp"
#include "guild/model/archetype.hpp"
#include "guild/model/model_descriptor.hpp"
#include "guild/model/runner.hpp"
#include "guild/models/manifest.hpp"
#include "guild/models/sha256.hpp"
#include "guild/models/store.hpp"
#include "guild/models/registry.hpp"
#include "guild/models/downloader.hpp"
#include "guild/server/engine.hpp"
#include "guild/server/json.hpp"
#include "guild/server/native_engine.hpp"
#include "guild/server/server.hpp"
#include "guild/runtime/chat_template.hpp"
#include "guild/runtime/version.hpp"

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

bool parse_prefill_batch(const char* value,int64_t& batch) {
    try {
        size_t used=0;batch=std::stoll(value,&used);
        return used==std::strlen(value) && batch>=0 && batch<=512;
    } catch (...) { return false; }
}

bool resolve_model_descriptor(const std::string& path_or_name, guild::model::ModelDescriptor& desc) {
    bool described = false;
    if (guild::artifact::GPackReader::is_gpack_file(path_or_name)) {
        guild::artifact::GPackReader reader;
        std::string gerr;
        if (reader.open(path_or_name, gerr, false)) {
            desc = reader.descriptor();
            return true;
        }
    }
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
              << "  " << cyan() << "pull" << reset() << " <model>      Download and install a model into local store\n"
              << "  " << cyan() << "list" << reset() << "              List locally installed models in Guild store\n"
              << "  " << cyan() << "show" << reset() << " <model>      Show model manifest, geometry, and execution plan\n"
              << "  " << cyan() << "rm" << reset() << " <model>        Safely remove a model and its unreferenced blobs\n"
              << "  " << cyan() << "import" << reset() << " <path>     Import existing model directory/files into store without copying\n"
              << "  " << cyan() << "serve" << reset() << " [model]     Start OpenAI-compatible HTTP inference server\n"
              << "  " << cyan() << "inspect" << reset() << " <model>   Inspect model GGUF/GPACK architecture, metadata, and MoE layout\n"
              << "                      [--plan] [--json] to compute native execution memory plan\n"
              << "  " << cyan() << "convert" << reset() << " <in>      Convert GGUF/pack or safetensors into portable .gpack container\n"
              << "  " << cyan() << "verify" << reset() << " <gpack>    Verify .gpack container structure and SHA-256 section digests\n"
              << "  " << cyan() << "kernel" << reset() << " <subcmd>   Manage .gkernel/.gkmodule packages\n"
              << "  " << cyan() << "bench" << reset() << " <model>     Run performance benchmark suite on target model\n"
              << "  " << cyan() << "ps" << reset() << "                List running Guild model instances and memory status\n"
              << "  " << cyan() << "version" << reset() << "           Show Guild version and platform build configuration\n\n"
              << bold() << "Options:" << reset() << "\n"
              << "  -h, --help        Show help for command\n"
              << "  -v, --version     Show Guild version\n";
}

int cmd_version(int argc, char** argv) {
    bool json=false;
    for(int i=2;i<argc;++i) {
        if(std::string(argv[i])=="--json")json=true;
        else {std::cerr<<"Usage: guild --version [--json]\n";return 2;}
    }
    std::string output,error;
    const bool ok=guild::runtime::version_report(json,output,error);
    std::cout<<output;
    if(!ok && output.empty())std::cerr<<error<<'\n';
    return ok?0:1;
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
    std::string data_dir;
    bool verbose = false;
    bool quiet = false;
    bool json_mode = false;
    bool no_tui = false;
    bool force_mock = false;

    int64_t prefill_batch = 0;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (arg == "--host" && i + 1 < argc) host = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model = argv[++i];
        else if (arg == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
        else if (arg == "--verbose") verbose = true;
        else if (arg == "--quiet") quiet = true;
        else if (arg == "--json") json_mode = true;
        else if (arg == "--no-tui") no_tui = true;
        else if (arg == "--mock") force_mock = true;
        else if (arg == "--prefill-batch") {
            if (i+1>=argc || !parse_prefill_batch(argv[++i],prefill_batch)) {
                std::cerr<<"--prefill-batch requires 0(auto) or an integer in 1..512\n";
                return 2;
            }
        }
        else if (arg[0] != '-' && model == "Qwen3.8-Flash-Next") {
            model = arg;
        }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: guild serve [model] [options]\n"
                      << "  --port <port>   Port to listen on (default: 11434)\n"
                      << "  --host <host>   Host to bind to (default: 127.0.0.1)\n"
                      << "  --model <name>  Model name or path (default: Qwen3.8-Flash-Next)\n"
                      << "  --data-dir PATH Custom data directory for Guild model store\n"
                      << "  --verbose       Enable verbose logging\n"
                      << "  --quiet         Quiet mode, suppress status dashboard\n"
                      << "  --json          Output machine-readable JSON status & JSONL telemetry\n"
                      << "  --no-tui        Disable ANSI terminal dashboard, emit sequential logs\n"
                      << "  --mock          Run with mock inference engine for testing\n";
            std::cout << "  --prefill-batch N Generic prompt batch (0 auto, 1..512)\n";
            return 0;
        }
    }

    const auto hw = guild::cli::detect_hardware();

    guild::models::StoreOptions st_opts;
    st_opts.custom_data_dir = data_dir;
    guild::models::ModelStore store(st_opts);
    store.init();

    guild::model::ModelDescriptor desc;
    auto manifest_opt = guild::models::ModelRegistry::instance().resolve(model, store);
    if (manifest_opt.has_value()) {
        desc = manifest_opt->to_descriptor();
    } else if (!resolve_model_descriptor(model, desc)) {
        std::cerr << "guild serve: cannot identify or describe model '" << model << "'\n";
        return 1;
    }

    // Compute memory plan
    guild::memory::PlannerOptions planner_opts;
    planner_opts.context_length = desc.attn.context_length;
    auto plan = guild::memory::MemoryPlanner::plan(desc, hw, planner_opts);
    plan.prefill_batch = prefill_batch;
    auto val = guild::memory::MemoryPlanner::validate(plan, hw, desc);
    if (!force_mock && !val.valid) {
        std::cerr << "guild serve: memory plan validation failed: "
                  << (val.errors.empty() ? "unknown error" : val.errors[0]) << "\n";
        return 1;
    }

    // Engine selection
    std::shared_ptr<guild::server::IInferenceEngine> engine;

    std::string native_model;
    if (manifest_opt.has_value()) {
        const auto* prim = manifest_opt->find_file_by_role("primary");
        if (!prim) prim = manifest_opt->find_file_by_role("shard");
        if (prim && !prim->local_path.empty()) {
            native_model = prim->local_path;
        }
    }
    if (native_model.empty() && manifest_opt && manifest_opt->name == "qwen3.8-flash-next" &&
        std::filesystem::exists("/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")) {
        native_model = "/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    }

    std::string pack_dir = "/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs";
    std::string profile_bin = "/home/ubuntu/Guild/data/expert-profile.bin";
    if (!std::filesystem::exists(profile_bin)) {
        profile_bin = "/home/ubuntu/Strata/data/expert-profile.bin";
    }
    std::string mtp_dir = "/mnt/models-ssd/Strata-data/mtp/rt";

    if (manifest_opt.has_value()) {
        if (manifest_opt->metadata.count("pack_dir")) {
            pack_dir = manifest_opt->metadata.at("pack_dir");
        }
        if (manifest_opt->metadata.count("expert_profile")) {
            profile_bin = manifest_opt->metadata.at("expert_profile");
        } else if (desc.archetype != guild::model::ModelArchetype::Qwen4Exp) {
            profile_bin = "";
        }
        if (manifest_opt->metadata.count("mtp_dir")) {
            mtp_dir = manifest_opt->metadata.at("mtp_dir");
        } else if (desc.archetype != guild::model::ModelArchetype::Qwen4Exp) {
            mtp_dir = "";
        }
    }

    std::string tokenizer_dir = pack_dir + "/tokenizer";
    if (manifest_opt.has_value() && manifest_opt->metadata.count("tokenizer_dir")) {
        tokenizer_dir = manifest_opt->metadata.at("tokenizer_dir");
    }

    if (native_model.empty() && guild::artifact::GPackReader::is_gpack_file(model)) {
        native_model = model;
    }
    const bool is_gpack_model = guild::artifact::GPackReader::is_gpack_file(native_model);
    bool real_weights_available = is_gpack_model ||
                                  (std::filesystem::exists(pack_dir) && std::filesystem::exists(native_model));

    if (!force_mock && real_weights_available) {
        guild::server::NativeInferenceEngineOptions n_opts;
        n_opts.model_name = desc.name;
        n_opts.desc = desc;
        n_opts.plan = plan;
        n_opts.paths.primary_model_path = native_model;
        n_opts.paths.pack_dir = pack_dir;
        n_opts.paths.expert_profile_path = profile_bin;
        n_opts.paths.mtp_dir = mtp_dir;
        n_opts.paths.tokenizer_dir = tokenizer_dir;

        auto native_engine = std::make_shared<guild::server::NativeInferenceEngine>(std::move(n_opts));
        std::string n_err;
        if (native_engine->init(n_err)) {
            if (verbose) {
                std::cout << "[server] Instantiated native in-process inference engine\n";
            }
            engine = native_engine;
        } else {
            std::cerr << "guild serve: native model initialization failed: " << n_err << "\n";
            return 1;
        }
    }

    if (force_mock) {
        engine = std::make_shared<guild::server::MockInferenceEngine>(desc.name, plan.context_length);
        std::cerr << "guild serve: explicit --mock execution (no model inference)\n";
    } else if (!engine) {
        std::cerr << "guild serve: model weights or pack are unavailable for '" << model
                  << "'; mock execution requires --mock\n";
        return 1;
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

    g_active_server.store(server.get());
    std::signal(SIGINT, handle_server_signal);
    std::signal(SIGTERM, handle_server_signal);

    // UX presentation
    if (json_mode) {
        std::cout << "{\n"
                  << "  \"guild_version\": \"" << GUILD_VERSION << "\",\n"
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
        std::cout << bold() << "Guild " << GUILD_VERSION << reset() << "\n"
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

    server->run();

    g_active_server.store(nullptr);
    return 0;
}


int cmd_pull(int argc, char** argv) {
    using namespace guild::cli::ansi;
    std::string model_name;
    std::string data_dir;
    bool force = false;
    bool json_mode = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--force") force = true;
        else if (arg == "--json") json_mode = true;
        else if (arg == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
        else if (arg[0] != '-' && model_name.empty()) model_name = arg;
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: guild pull <model> [options]\n"
                      << "  --force          Re-download and overwrite existing files\n"
                      << "  --json           Emit machine-readable JSON progress\n"
                      << "  --data-dir PATH  Custom data directory for Guild model store\n";
            return 0;
        }
    }

    if (model_name.empty()) {
        std::cerr << "guild pull: missing model name. Usage: guild pull <model>\n";
        return 1;
    }

    guild::models::StoreOptions s_opts;
    s_opts.custom_data_dir = data_dir;
    guild::models::ModelStore store(s_opts);
    store.init();

    auto manifest_opt = guild::models::ModelRegistry::instance().resolve(model_name, store);
    if (!manifest_opt.has_value()) {
        manifest_opt = guild::models::ModelRegistry::instance().find_builtin(model_name);
    }
    if (!manifest_opt.has_value()) {
        std::cerr << "guild pull: unknown model '" << model_name << "'. Available models:\n";
        for (const auto& b : guild::models::ModelRegistry::instance().list_available()) {
            std::cerr << "  " << b.name << "\n";
        }
        return 1;
    }

    const auto& manifest = *manifest_opt;

    // Check if already installed & verified
    std::string verify_err;
    if (!force && store.has_manifest(manifest.name) && store.verify_model(manifest, false, verify_err)) {
        if (json_mode) {
            std::cout << "{\"status\":\"completed\",\"model\":\"" << manifest.name << "\"}" << std::endl;
        } else {
            std::cout << manifest.name << " is already downloaded and verified.\n";
        }
        return 0;
    }

    if (!json_mode) {
        std::cout << "pulling " << manifest.name << "\n\n";
    }

    guild::models::Downloader downloader;
    guild::models::DownloadOptions d_opts;
    d_opts.force = force;
    d_opts.json_progress = json_mode;

    bool is_interactive = guild::cli::is_tty();
    auto last_render = std::chrono::steady_clock::now();

    d_opts.progress_cb = [&](const guild::models::DownloadProgress& p) -> bool {
        if (json_mode) {
            std::cout << "{\"status\":\"downloading\",\"file\":\"" << p.current_file_name
                      << "\",\"file_index\":" << p.current_file_index
                      << ",\"total_files\":" << p.total_files
                      << ",\"file_downloaded\":" << p.file_downloaded_bytes
                      << ",\"file_total\":" << p.file_total_bytes
                      << ",\"total_downloaded\":" << p.total_downloaded_bytes
                      << ",\"total_expected\":" << p.total_expected_bytes
                      << ",\"speed\":" << p.speed_bytes_sec << "}" << std::endl;
            return true;
        }

        auto now = std::chrono::steady_clock::now();
        double delta = std::chrono::duration<double>(now - last_render).count();
        if (delta >= 0.1 || p.file_downloaded_bytes == p.file_total_bytes) {
            last_render = now;
            std::string speed_str = guild::models::Downloader::format_speed(p.speed_bytes_sec);
            std::string file_cur = guild::models::Downloader::format_bytes(p.file_downloaded_bytes);
            std::string file_tot = guild::models::Downloader::format_bytes(p.file_total_bytes);

            if (is_interactive) {
                std::cout << "\r" << std::left << std::setw(16) << p.current_file_name.substr(0, 15)
                          << " " << std::right << std::setw(16) << (file_cur + " / " + file_tot)
                          << "   " << std::setw(11) << speed_str
                          << "   (" << std::fixed << std::setprecision(1) << p.percentage << "%)"
                          << std::flush;
            } else {
                std::cout << p.current_file_name << ": " << file_cur << " / " << file_tot
                          << " (" << speed_str << ")\n" << std::flush;
            }
        }
        return true;
    };

    std::string err_msg;
    bool success = downloader.download_model(manifest, store, d_opts, err_msg);
    if (!success) {
        if (!json_mode) std::cout << "\n";
        std::cerr << "guild pull failed: " << err_msg << "\n";
        return 1;
    }

    if (!json_mode) {
        std::cout << "\n\n"
                  << "downloaded      " << guild::models::Downloader::format_bytes(manifest.expected_size_bytes)
                  << " / " << guild::models::Downloader::format_bytes(manifest.expected_size_bytes) << "\n"
                  << "disk free       " << guild::models::Downloader::format_bytes(guild::models::Downloader::get_available_disk_space(store.models_dir())) << "\n";
    }

    return 0;
}

int cmd_list(int argc, char** argv) {
    using namespace guild::cli::ansi;
    std::string data_dir;
    bool json_mode = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") json_mode = true;
        else if (arg == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: guild list [--json] [--data-dir PATH]\n";
            return 0;
        }
    }

    guild::models::StoreOptions s_opts;
    s_opts.custom_data_dir = data_dir;
    guild::models::ModelStore store(s_opts);
    store.init();

    auto manifests = store.list_manifests();

    if (json_mode) {
        std::cout << "[\n";
        for (size_t i = 0; i < manifests.size(); ++i) {
            const auto& m = manifests[i];
            std::cout << "  {\n"
                      << "    \"name\": \"" << m.name << "\",\n"
                      << "    \"architecture\": \"" << m.architecture << "\",\n"
                      << "    \"quantization\": \"" << m.quantization << "\",\n"
                      << "    \"size_bytes\": " << m.total_size_bytes() << "\n"
                      << "  }" << (i + 1 < manifests.size() ? "," : "") << "\n";
        }
        std::cout << "]\n";
        return 0;
    }

    if (manifests.empty()) {
        std::cout << "No models installed. Run 'guild pull <model>' to download one.\n"
                  << "Available built-in models:\n";
        for (const auto& b : guild::models::ModelRegistry::instance().list_available()) {
            std::cout << "  " << b.name << " (" << b.quantization << ", ~"
                      << guild::models::Downloader::format_bytes(b.expected_size_bytes) << ")\n";
        }
        return 0;
    }

    std::cout << bold() << std::left
              << std::setw(25) << "NAME"
              << std::setw(14) << "ARCH"
              << std::setw(14) << "QUANT"
              << "SIZE" << reset() << "\n";

    for (const auto& m : manifests) {
        std::cout << std::left
                  << std::setw(25) << m.name
                  << std::setw(14) << m.architecture
                  << std::setw(14) << m.quantization
                  << guild::models::Downloader::format_bytes(m.total_size_bytes()) << "\n";
    }
    return 0;
}

int cmd_show(int argc, char** argv) {
    using namespace guild::cli::ansi;
    std::string model_name;
    std::string data_dir;
    bool show_plan = false;
    bool json_mode = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--plan") show_plan = true;
        else if (arg == "--json") json_mode = true;
        else if (arg == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
        else if (arg[0] != '-' && model_name.empty()) model_name = arg;
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: guild show <model> [--plan] [--json] [--data-dir PATH]\n";
            return 0;
        }
    }

    if (model_name.empty()) {
        std::cerr << "Usage: guild show <model> [--plan] [--json]\n";
        return 1;
    }

    guild::models::StoreOptions s_opts;
    s_opts.custom_data_dir = data_dir;
    guild::models::ModelStore store(s_opts);
    store.init();

    auto m_opt = guild::models::ModelRegistry::instance().resolve(model_name, store);
    if (!m_opt.has_value()) {
        std::cerr << "guild show: model '" << model_name << "' not found.\n";
        return 1;
    }

    const auto& m = *m_opt;
    auto desc = m.to_descriptor();

    if (show_plan) {
        const auto hw = guild::cli::detect_hardware();
        guild::memory::PlannerOptions popts;
        popts.context_length = desc.attn.context_length;
        auto plan = guild::memory::MemoryPlanner::plan(desc, hw, popts);

        if (json_mode) {
            std::cout << plan.to_json_string() << "\n";
        } else {
            std::cout << plan.to_human_string() << "\n";
        }
        return 0;
    }

    if (json_mode) {
        std::cout << m.to_json() << "\n";
        return 0;
    }

    std::cout << bold() << "Model Information" << reset() << "\n"
              << "────────────────────────────────────────\n"
              << "Name          " << bold() << m.name << reset() << "\n"
              << "Architecture  " << cyan() << m.architecture << reset() << "\n"
              << "Quantization  " << m.quantization << "\n"
              << "Source        " << m.source << "\n"
              << "Total Size    " << guild::models::Downloader::format_bytes(m.total_size_bytes()) << "\n"
              << "Context       " << m.context_length << " tokens\n";

    if (m.n_routed_experts > 0) {
        std::cout << "\n" << bold() << "Mixture-of-Experts" << reset() << "\n"
                  << "Routed Experts " << m.n_routed_experts << " (" << m.k_active_experts << " active per token)\n"
                  << "Total Experts  " << (m.n_routed_experts * m.n_layers) << "\n"
                  << "Routed Size    ~" << std::fixed << std::setprecision(2)
                  << (desc.moe.expert_blob_bytes * m.n_routed_experts * m.n_layers / (1024.0 * 1024.0 * 1024.0)) << " GiB\n";
    }

    std::cout << "\n" << bold() << "Files" << reset() << "\n";
    for (const auto& f : m.files) {
        std::cout << "  " << std::left << std::setw(36) << f.name
                  << "  " << std::right << std::setw(10) << guild::models::Downloader::format_bytes(f.size_bytes)
                  << "  [" << f.role << "]\n";
        if (!f.local_path.empty()) {
            std::cout << "    path: " << f.local_path << "\n";
        }
    }
    std::cout << "────────────────────────────────────────\n";
    return 0;
}

int cmd_rm(int argc, char** argv) {
    std::string model_name;
    std::string data_dir;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
        else if (arg[0] != '-' && model_name.empty()) model_name = arg;
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: guild rm <model> [--data-dir PATH]\n";
            return 0;
        }
    }

    if (model_name.empty()) {
        std::cerr << "Usage: guild rm <model>\n";
        return 1;
    }

    guild::models::StoreOptions s_opts;
    s_opts.custom_data_dir = data_dir;
    guild::models::ModelStore store(s_opts);
    store.init();

    auto m = store.get_manifest(model_name);
    if (!m.has_value()) {
        std::cerr << "guild rm: model '" << model_name << "' is not installed.\n";
        return 1;
    }

    uint64_t freed_bytes = m->total_size_bytes();
    if (!store.remove_model(model_name)) {
        std::cerr << "guild rm: failed to remove model '" << model_name << "'.\n";
        return 1;
    }

    std::cout << "Removed model '" << model_name << "' (freed "
              << guild::models::Downloader::format_bytes(freed_bytes) << ").\n";
    return 0;
}

int cmd_import(int argc, char** argv) {
    namespace fs = std::filesystem;
    std::string path;
    std::string name;
    std::string data_dir;
    bool copy_mode = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--name" && i + 1 < argc) name = argv[++i];
        else if (arg == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
        else if (arg == "--copy") copy_mode = true;
        else if (arg[0] != '-' && path.empty()) path = arg;
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: guild import <path> [--name <name>] [--copy] [--data-dir PATH]\n"
                      << "  <path>         Directory or GGUF file to import\n"
                      << "  --name <name>  Model name to assign\n"
                      << "  --copy         Copy files instead of symlinking (default: symlink)\n";
            return 0;
        }
    }

    if (path.empty()) {
        std::cerr << "Usage: guild import <path> [--name <name>]\n";
        return 1;
    }

    guild::models::StoreOptions s_opts;
    s_opts.custom_data_dir = data_dir;
    guild::models::ModelStore store(s_opts);
    store.init();

    std::optional<guild::models::ModelManifest> base_manifest;
    if (!name.empty()) {
        base_manifest = guild::models::ModelRegistry::instance().find_builtin(name);
    }
    if (!base_manifest.has_value()) {
        for (const auto& b : guild::models::ModelRegistry::instance().list_available()) {
            if (path.find("unsloth") != std::string::npos || path.find("UD-IQ4_XS") != std::string::npos || path.find("Qwen") != std::string::npos) {
                base_manifest = b;
                break;
            }
        }
    }
    if (!base_manifest.has_value()) {
        guild::models::ModelManifest m;
        m.name = name.empty() ? fs::path(path).filename().string() : name;
        m.architecture = "unknown";
        m.quantization = "unknown";
        base_manifest = m;
    }

    if (!name.empty()) {
        base_manifest->name = name;
    }

    std::string err_msg;
    if (guild::artifact::GPackReader::is_gpack_file(path)) {
        guild::artifact::GPackReader reader;
        if (!reader.open(path, err_msg, false)) {
            std::cerr << "guild import failed: " << err_msg << "\n";
            return 1;
        }
        const auto& d = reader.descriptor();
        guild::models::ModelManifest gm;
        gm.name = name.empty() ? d.name : name;
        if (gm.name.empty()) gm.name = fs::path(path).stem().string();
        gm.architecture = d.arch_name;
        gm.quantization = "gpack.v1";
        gm.source = "local-gpack";
        gm.context_length = d.attn.context_length;
        gm.vocab_size = d.attn.vocab_size;
        gm.n_embd = d.attn.n_embd;
        gm.n_layers = d.attn.n_layers;
        gm.n_heads = d.attn.n_heads;
        gm.n_kv_heads = d.attn.n_kv_heads;
        gm.head_dim = d.attn.head_dim;
        gm.full_attn_interval = d.attn.full_attn_interval;
        gm.n_routed_experts = d.moe.n_routed_experts;
        gm.k_active_experts = d.moe.k_active_experts;
        gm.expert_dim_ff = d.moe.expert_dim_ff;
        gm.n_shared_experts = d.moe.n_shared_experts;
        gm.shared_dim_ff = d.moe.shared_dim_ff;
        gm.expert_blob_bytes = d.moe.expert_blob_bytes;
        guild::models::ModelFile mf;
        mf.name = fs::path(path).filename().string();
        mf.role = "primary";
        mf.size_bytes = reader.file_size();
        mf.local_path = fs::canonical(path).string();
        gm.files.push_back(mf);
        if (!store.save_manifest(gm)) {
            std::cerr << "guild import failed: could not save GPACK manifest to store\n";
            return 1;
        }
        std::cout << "Successfully imported GPACK '" << gm.name << "' into Guild store.\n";
        return 0;
    }

    bool ok = store.import_model_directory(*base_manifest, path, !copy_mode, err_msg);
    if (!ok) {
        std::cerr << "guild import failed: " << err_msg << "\n";
        return 1;
    }

    std::cout << "Successfully imported '" << base_manifest->name << "' into Guild store ("
              << (!copy_mode ? "zero-copy linked" : "copied") << ").\n";
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
    using namespace guild::cli::ansi;
    std::string model = "qwen3.8-flash-next";
    std::string data_dir;
    bool force_mock = false;
    int64_t prefill_batch = 0;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mock") force_mock = true;
        else if (arg == "--prefill-batch") {
            if (i+1>=argc || !parse_prefill_batch(argv[++i],prefill_batch)) {
                std::cerr<<"--prefill-batch requires 0(auto) or an integer in 1..512\n";
                return 2;
            }
        }
        else if (arg == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
        else if (arg[0] != '-' && (model == "qwen3.8-flash-next" || model.empty())) {
            model = arg;
        }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: guild run [model] [options]\n"
                      << "  --mock          Run with mock inference engine for testing\n"
                      << "  --data-dir PATH Custom data directory for Guild model store\n";
            std::cout << "  --prefill-batch N Generic prompt batch (0 auto, 1..512)\n";
            return 0;
        }
    }

    guild::models::StoreOptions st_opts;
    st_opts.custom_data_dir = data_dir;
    guild::models::ModelStore store(st_opts);
    store.init();

    auto m_opt = store.get_manifest(model);
    if (!m_opt.has_value()) {
        auto builtin = guild::models::ModelRegistry::instance().find_builtin(model);
        if (builtin.has_value()) {
            bool has_local = false;
            for (const auto& f : builtin->files) {
                if (!f.local_path.empty() && std::filesystem::exists(f.local_path)) {
                    has_local = true;
                    break;
                }
            }
            if (has_local || force_mock) {
                m_opt = builtin;
            } else {
                std::cerr << "guild run: model '" << model << "' is not installed.\n"
                          << "Run 'guild pull " << builtin->name << "' to download it.\n";
                return 1;
            }
        } else if (std::filesystem::exists(model)) {
            m_opt = guild::models::ModelRegistry::instance().resolve(model, store);
        } else {
            std::cerr << "guild run: unknown model '" << model << "'.\n"
                      << "Run 'guild list' to view installed models.\n";
            return 1;
        }
    }

    const auto& manifest = *m_opt;
    auto desc = manifest.to_descriptor();
    const auto hw = guild::cli::detect_hardware();

    guild::memory::PlannerOptions popts;
    popts.context_length = desc.attn.context_length;
    auto plan = guild::memory::MemoryPlanner::plan(desc, hw, popts);
    plan.prefill_batch = prefill_batch;
    auto val = guild::memory::MemoryPlanner::validate(plan, hw, desc);
    if (!force_mock && !val.valid) {
        std::cerr << "guild run: execution plan validation failed:\n";
        for (const auto& err : val.errors) std::cerr << "  - " << err << "\n";
        return 1;
    }

    // Engine selection
    std::shared_ptr<guild::server::IInferenceEngine> engine;

    std::string native_model;
    const auto* prim = manifest.find_file_by_role("primary");
    if (!prim) prim = manifest.find_file_by_role("shard");
    if (prim && !prim->local_path.empty()) {
        native_model = prim->local_path;
    }
    if (native_model.empty() && manifest.name == "qwen3.8-flash-next" &&
        std::filesystem::exists("/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")) {
        native_model = "/mnt/models-ssd/Strata-data/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf";
    }

    std::string pack_dir = "/mnt/models-ssd/Strata-data/packs/unsloth-ud-iq4_xs";
    std::string profile_bin = "/home/ubuntu/Guild/data/expert-profile.bin";
    if (!std::filesystem::exists(profile_bin)) profile_bin = "/home/ubuntu/Strata/data/expert-profile.bin";
    std::string mtp_dir = "/mnt/models-ssd/Strata-data/mtp/rt";

    if (manifest.metadata.count("pack_dir")) {
        pack_dir = manifest.metadata.at("pack_dir");
    }
    if (manifest.metadata.count("expert_profile")) {
        profile_bin = manifest.metadata.at("expert_profile");
    } else if (desc.archetype != guild::model::ModelArchetype::Qwen4Exp) {
        profile_bin = "";
    }
    if (manifest.metadata.count("mtp_dir")) {
        mtp_dir = manifest.metadata.at("mtp_dir");
    } else if (desc.archetype != guild::model::ModelArchetype::Qwen4Exp) {
        mtp_dir = "";
    }

    std::string tokenizer_dir = pack_dir + "/tokenizer";
    if (manifest.metadata.count("tokenizer_dir")) {
        tokenizer_dir = manifest.metadata.at("tokenizer_dir");
    }

    if (native_model.empty() && guild::artifact::GPackReader::is_gpack_file(model)) {
        native_model = model;
    }
    const bool is_gpack_model = guild::artifact::GPackReader::is_gpack_file(native_model);
    bool real_weights_available = is_gpack_model ||
                                  (std::filesystem::exists(pack_dir) && std::filesystem::exists(native_model));

    if (!force_mock && real_weights_available) {
        guild::server::NativeInferenceEngineOptions n_opts;
        n_opts.model_name = desc.name;
        n_opts.desc = desc;
        n_opts.plan = plan;
        n_opts.paths.primary_model_path = native_model;
        n_opts.paths.pack_dir = pack_dir;
        n_opts.paths.expert_profile_path = profile_bin;
        n_opts.paths.mtp_dir = mtp_dir;
        n_opts.paths.tokenizer_dir = tokenizer_dir;

        auto native_engine = std::make_shared<guild::server::NativeInferenceEngine>(std::move(n_opts));
        std::string n_err;
        if (native_engine->init(n_err)) {
            engine = native_engine;
        } else {
            std::cerr << "guild run: native model initialization failed: " << n_err << "\n";
            return 1;
        }
    }

    if (force_mock) {
        engine = std::make_shared<guild::server::MockInferenceEngine>(desc.name, plan.context_length);
        std::cerr << "guild run: explicit --mock execution (no model inference)\n";
    } else if (!engine) {
        std::cerr << "guild run: model weights or pack are unavailable for '" << model
                  << "'; mock execution requires --mock\n";
        return 1;
    }

    std::cout << bold() << "Guild · " << manifest.name << reset() << "\n"
              << (plan.context_length / 1024) << "K context · "
              << guild::memory::kv_precision_to_string(plan.kv_format) << " "
              << guild::memory::kv_mode_to_string(plan.kv_mode) << "\n\n";

    std::string line;
    while (true) {
        std::cout << bold() << ">>> " << reset() << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/quit" || line == "/exit" || line == "exit") break;

        guild::server::InferenceRequest req;
        req.model = manifest.name;
        if (const auto* tpl = engine->chat_template(); tpl && tpl->is_loaded()) {
            std::vector<guild::runtime::ChatMessage> msgs = {{"user", line}};
            std::string tpl_err;
            if (!tpl->render(msgs, true, req.prompt, tpl_err)) {
                std::cerr << "guild run: template rendering failed: " << tpl_err << "\n";
                return 1;
            }
        } else if (line.rfind("<|im_start|>", 0) == std::string::npos) {
            req.prompt = "<|im_start|>user\n" + line + "<|im_end|>\n<|im_start|>assistant\n";
        } else {
            req.prompt = line;
        }

        std::string enc_err;
        if (!engine->encode(req.prompt, req.prompt_tokens, enc_err)) {
            std::cerr << "guild run: tokenization failed: " << enc_err << "\n";
            return 1;
        }
        req.max_tokens = 512;

        guild::server::GenerationResult gen_res;
        auto stream_cb = [](const guild::server::TokenOutput& tok) -> bool {
            if (tok.is_special) return true;
            std::cout << tok.text << std::flush;
            return true;
        };

        if (!engine->generate_stream(req, stream_cb, gen_res) || gen_res.finish_reason == "error") {
            std::cerr << "\nguild run: inference failed: " << gen_res.error_message << "\n";
            return 1;
        }
        std::cout << "\n\n";
    }

    return 0;
}

int cmd_kernel(int argc, char** argv) {
    using namespace guild::cli::ansi;
    using guild::server::json::JsonValue;
    namespace fs = std::filesystem;

    auto print_kernel_help = []() {
        std::cout << bold() << "Guild Kernel & Runner Package Manager (.gkernel)" << reset() << "\n\n"
                  << "Usage:\n"
                  << "  guild kernel <subcommand> [options]\n\n"
                  << "Subcommands:\n"
                  << "  install <path> [--no-enable] [--allow-unsigned-dev] [--data-dir PATH] [--json]\n"
                  << "  list [--data-dir PATH] [--json]\n"
                  << "  inspect <package_id|path> [--data-dir PATH] [--json]\n"
                  << "  enable <package_id> [--data-dir PATH] [--json]\n"
                  << "  disable <package_id> [--data-dir PATH] [--json]\n"
                  << "  remove <package_id> [--data-dir PATH] [--json]\n"
                  << "  rollback <package_id> [--data-dir PATH] [--json]\n"
                  << "  active [--data-dir PATH] [--json]\n"
                  << "  keygen [--seed <64-hex>] [--json]\n"
                  << "  sign <pkg_dir> --key-id <id> --seed <64-hex> [--bundle <out.gkernel>]\n"
                  << "  trust-key --publisher <pub_id> --key-id <id> --pubkey <64-hex> [--data-dir PATH]\n"
                  << "  revoke-key <key_id> [--data-dir PATH]\n"
                  << "  revoke-release <package_id@version> [--data-dir PATH]\n";
    };

    if (argc < 3) {
        print_kernel_help();
        return 1;
    }

    const std::string sub = argv[2];
    if (sub == "-h" || sub == "--help" || sub == "help") {
        print_kernel_help();
        return 0;
    }

    std::string target;
    std::string data_dir;
    std::string key_id;
    std::string seed_hex;
    std::string seed_file;
    std::string pubkey_hex;
    std::string publisher_id;
    std::string bundle_out;
    bool enable_now = true;
    bool allow_unsigned_dev = false;
    bool json_mode = false;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--data-dir" && i + 1 < argc) data_dir = argv[++i];
        else if (arg == "--key-id" && i + 1 < argc) key_id = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) seed_hex = argv[++i];
        else if (arg == "--seed-file" && i + 1 < argc) seed_file = argv[++i];
        else if (arg == "--pubkey" && i + 1 < argc) pubkey_hex = argv[++i];
        else if (arg == "--publisher" && i + 1 < argc) publisher_id = argv[++i];
        else if (arg == "--bundle" && i + 1 < argc) bundle_out = argv[++i];
        else if (arg == "--no-enable") enable_now = false;
        else if (arg == "--allow-unsigned-dev") allow_unsigned_dev = true;
        else if (arg == "--json") json_mode = true;
        else if (arg == "-h" || arg == "--help") {
            print_kernel_help();
            return 0;
        } else if (!arg.empty() && arg[0] != '-' && target.empty()) {
            target = arg;
        }
    }

    if (!seed_file.empty()) {
        if (!seed_hex.empty()) { std::cerr << "choose --seed or --seed-file\n"; return 1; }
        std::ifstream in(seed_file);
        if (!(in >> seed_hex) || seed_hex.size()!=64) {
            std::cerr << "invalid signing seed file\n";
            return 1;
        }
    }
    if (sub == "keygen") {
        guild::abi::crypto::Ed25519KeyPair kp;
        if (!seed_hex.empty()) {
            if (!guild::abi::crypto::ed25519_keypair_from_seed_hex(seed_hex, kp)) {
                std::cerr << "guild kernel keygen: invalid 64-character hex seed\n";
                return 1;
            }
        } else {
            guild::abi::crypto::ed25519_generate_keypair(kp);
        }
        if (json_mode) {
            std::cout << "{\n"
                      << "  \"algorithm\": \"ed25519\",\n"
                      << "  \"private_seed_hex\": \"" << kp.seed_hex << "\",\n"
                      << "  \"public_key_hex\": \"" << kp.public_key_hex << "\"\n"
                      << "}\n";
        } else {
            std::cout << "Algorithm:       ed25519 (RFC 8032)\n"
                      << "Private Seed:    " << kp.seed_hex << "\n"
                      << "Public Key:      " << kp.public_key_hex << "\n";
        }
        return 0;
    }

    if (sub == "sign") {
        if (target.empty() || key_id.empty() || seed_hex.empty()) {
            std::cerr << "Usage: guild kernel sign <pkg_dir> --key-id <id> --seed <64-hex> [--bundle <out.gkernel>]\n";
            return 1;
        }
        const fs::path pkg_dir(target);
        const fs::path meta_path = pkg_dir / ".gkmetadata";
        std::ifstream mf(meta_path, std::ios::binary);
        if (!mf.good()) {
            std::cerr << "guild kernel sign: cannot open " << meta_path.string() << "\n";
            return 1;
        }
        std::ostringstream ss;
        ss << mf.rdbuf();
        mf.close();
        guild::abi::GKernelMetadata meta;
        std::string err;
        if (guild::abi::GKernelMetadata::parse_json(ss.str(), meta, err) != GUILD_STATUS_OK) {
            std::cerr << "guild kernel sign: invalid .gkmetadata: " << err << "\n";
            return 1;
        }
        // Recompute SHA-256 and byte_size for all declared artifacts on disk
        for (auto& art : meta.artifacts) {
            const fs::path ap = pkg_dir / art.relative_path;
            std::error_code ec;
            if (!fs::is_regular_file(fs::symlink_status(ap, ec))) {
                std::cerr << "guild kernel sign: missing artifact '" << art.relative_path << "'\n";
                return 1;
            }
            art.byte_size = fs::file_size(ap, ec);
            art.sha256 = guild::models::Sha256::hash_file(ap.string());
        }
        if (!meta.sign_with_seed_hex(key_id, seed_hex, err)) {
            std::cerr << "guild kernel sign: signing failed: " << err << "\n";
            return 1;
        }
        {
            std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
            const std::string updated = meta.to_json_string();
            out.write(updated.data(), static_cast<std::streamsize>(updated.size()));
        }
        if (!bundle_out.empty()) {
            if (guild::abi::create_gkernel_bundle(pkg_dir.string(), bundle_out, err) != GUILD_STATUS_OK) {
                std::cerr << "guild kernel sign: bundle creation failed: " << err << "\n";
                return 1;
            }
        }
        std::cout << "Signed package '" << meta.package_id << "@" << meta.version
                  << "' with key '" << key_id << "' (payload_sha256=" << meta.signature.payload_sha256 << ").\n";
        return 0;
    }

    const std::string store_root = data_dir.empty() ? "" : (fs::path(data_dir) / "kernels").string();
    guild::abi::GKernelPackageManager mgr(store_root);
    std::string err;
    if (!mgr.init(err)) {
        std::cerr << "guild kernel: failed to initialize package store: " << err << "\n";
        return 1;
    }
    if (allow_unsigned_dev) {
        std::cerr << "[WARNING] --allow-unsigned-dev enabled: bypassing Ed25519 signature check for local development\n";
        mgr.security_policy().allow_unsigned_dev = true;
    }

    if (sub == "trust-key") {
        if (publisher_id.empty() || key_id.empty() || pubkey_hex.size() != 64) {
            std::cerr << "Usage: guild kernel trust-key --publisher <pub_id> --key-id <id> --pubkey <64-hex> [--data-dir PATH]\n";
            return 1;
        }
        mgr.security_policy().trust_publisher_key(publisher_id, key_id, pubkey_hex);
        if (!mgr.save_security_policy(err)) {
            std::cerr << "guild kernel trust-key: " << err << "\n";
            return 1;
        }
        std::cout << "Trusted publisher '" << publisher_id << "' key '" << key_id << "'.\n";
        return 0;
    }

    if (sub == "revoke-key") {
        if (target.empty()) {
            std::cerr << "Usage: guild kernel revoke-key <key_id> [--data-dir PATH]\n";
            return 1;
        }
        mgr.security_policy().revoke_key(target);
        if (!mgr.save_security_policy(err)) {
            std::cerr << "guild kernel revoke-key: " << err << "\n";
            return 1;
        }
        std::cout << "Revoked signing key '" << target << "'.\n";
        return 0;
    }

    if (sub == "revoke-release") {
        const auto at = target.find('@');
        if (target.empty() || at == std::string::npos) {
            std::cerr << "Usage: guild kernel revoke-release <package_id@version> [--data-dir PATH]\n";
            return 1;
        }
        mgr.security_policy().revoke_release(target.substr(0, at), target.substr(at + 1));
        if (!mgr.save_security_policy(err)) {
            std::cerr << "guild kernel revoke-release: " << err << "\n";
            return 1;
        }
        std::cout << "Revoked package release '" << target << "'.\n";
        return 0;
    }

    if (sub == "install") {
        if (target.empty()) {
            std::cerr << "Usage: guild kernel install <path> [--no-enable] [--data-dir PATH] [--json]\n";
            return 1;
        }
        const guild_status_t st = mgr.install_package(target, enable_now, err);
        if (st != GUILD_STATUS_OK) {
            std::cerr << "guild kernel install failed (" << guild::kernels::status_to_string(st) << "): " << err << "\n";
            return 1;
        }
        auto pkgs = mgr.list_packages();
        if (json_mode) {
            std::cout << "{\"status\":\"installed\",\"enabled\":" << (enable_now ? "true" : "false")
                      << ",\"installed_count\":" << pkgs.size() << "}\n";
        } else {
            std::cout << "Installed extension package from '" << target << "' ("
                      << (enable_now ? "enabled" : "staged, disabled") << ").\n";
        }
        return 0;
    }

    if (sub == "list" || sub == "ls") {
        auto pkgs = mgr.list_packages();
        if (json_mode) {
            std::cout << "[\n";
            for (size_t i = 0; i < pkgs.size(); ++i) {
                const auto& p = pkgs[i];
                std::cout << "  {\n"
                          << "    \"package_id\": \"" << p.package_id << "\",\n"
                          << "    \"version\": \"" << p.active_version << "\",\n"
                          << "    \"package_kind\": \"" << p.metadata.package_kind << "\",\n"
                          << "    \"module_abi\": " << p.metadata.module_abi << ",\n"
                          << "    \"previous_version\": \"" << p.previous_version << "\",\n"
                          << "    \"scope\": \"" << p.metadata.scope << "\",\n"
                          << "    \"publisher_id\": \"" << p.metadata.publisher_id << "\",\n"
                          << "    \"enabled\": " << (p.enabled ? "true" : "false") << "\n"
                          << "  }" << (i + 1 < pkgs.size() ? "," : "") << "\n";
            }
            std::cout << "]\n";
            return 0;
        }
        if (pkgs.empty()) {
            std::cout << "No external extension packages installed.\n";
            return 0;
        }
        std::cout << bold() << std::left
                  << std::setw(32) << "PACKAGE_ID"
                  << std::setw(12) << "VERSION"
                  << std::setw(12) << "PREV"
                  << std::setw(15) << "SCOPE"
                  << std::setw(22) << "PUBLISHER"
                  << "STATUS" << reset() << "\n";
        for (const auto& p : pkgs) {
            std::cout << std::left
                      << std::setw(32) << p.package_id
                      << std::setw(12) << p.active_version
                      << std::setw(12) << (p.previous_version.empty() ? "-" : p.previous_version)
                      << std::setw(15) << p.metadata.scope
                      << std::setw(22) << p.metadata.publisher_id
                      << (p.enabled ? "enabled" : "disabled") << "\n";
        }
        return 0;
    }

    if (sub == "inspect") {
        if (target.empty()) {
            std::cerr << "Usage: guild kernel inspect <package_id|path> [--data-dir PATH] [--json]\n";
            return 1;
        }
        guild::abi::InstalledPackageRecord rec;
        if (mgr.get_package(target, rec)) {
            if (json_mode) {
                std::cout << rec.metadata.to_json_string() << "\n";
            } else {
                std::cout << "Package ID:    " << rec.package_id << "\n"
                          << "Package Kind:  " << rec.metadata.package_kind << "\n"
                          << "Version:       " << rec.active_version << "\n"
                          << "Previous Ver:  " << (rec.previous_version.empty() ? "-" : rec.previous_version) << "\n"
                          << "Status:        " << (rec.enabled ? "enabled" : "disabled") << "\n"
                          << "Scope:         " << rec.metadata.scope << "\n"
                          << "Publisher:     " << rec.metadata.publisher_id << "\n"
                          << "Signing Key:   " << rec.metadata.signature.key_id << " (" << rec.metadata.signature.algorithm << ")\n"
                          << "Staged Dir:    " << rec.staged_dir << "\n";
            }
            return 0;
        }
        // Otherwise inspect a directory or archive on disk without installing
        const fs::path tmp_inspect = fs::temp_directory_path() / ("guild_inspect_" + std::to_string(std::rand()));
        guild::abi::GKernelMetadata meta;
        const guild_status_t st = guild::abi::stage_gkernel_source(target, tmp_inspect.string(), meta, err);
        std::error_code ec;
        fs::remove_all(tmp_inspect, ec);
        if (st != GUILD_STATUS_OK) {
            std::cerr << "guild kernel inspect failed: " << err << "\n";
            return 1;
        }
        if (json_mode) {
            std::cout << meta.to_json_string() << "\n";
        } else {
            std::cout << "Package ID:    " << meta.package_id << "\n"
                      << "Package Kind:  " << meta.package_kind << "\n"
                      << "Version:       " << meta.version << "\n"
                      << "Scope:         " << meta.scope << "\n"
                      << "Publisher:     " << meta.publisher_id << "\n"
                      << "Signing Key:   " << meta.signature.key_id << " (" << meta.signature.algorithm << ")\n";
        }
        return 0;
    }

    if (sub == "enable") {
        if (target.empty()) {
            std::cerr << "Usage: guild kernel enable <package_id> [--data-dir PATH]\n";
            return 1;
        }
        const guild_status_t st = mgr.enable_package(target, err);
        if (st != GUILD_STATUS_OK) {
            std::cerr << "guild kernel enable failed: " << err << "\n";
            return 1;
        }
        std::cout << "Enabled package '" << target << "'.\n";
        return 0;
    }

    if (sub == "disable") {
        if (target.empty()) {
            std::cerr << "Usage: guild kernel disable <package_id> [--data-dir PATH]\n";
            return 1;
        }
        const guild_status_t st = mgr.disable_package(target, err);
        if (st != GUILD_STATUS_OK) {
            std::cerr << "guild kernel disable failed: " << err << "\n";
            return 1;
        }
        std::cout << "Disabled package '" << target << "'.\n";
        return 0;
    }

    if (sub == "remove" || sub == "rm" || sub == "uninstall") {
        if (target.empty()) {
            std::cerr << "Usage: guild kernel remove <package_id> [--data-dir PATH]\n";
            return 1;
        }
        const guild_status_t st = mgr.remove_package(target, err);
        if (st != GUILD_STATUS_OK) {
            std::cerr << "guild kernel remove failed: " << err << "\n";
            return 1;
        }
        std::cout << "Removed package '" << target << "'.\n";
        return 0;
    }

    if (sub == "rollback") {
        if (target.empty()) {
            std::cerr << "Usage: guild kernel rollback <package_id> [--data-dir PATH]\n";
            return 1;
        }
        const guild_status_t st = mgr.rollback_package(target, err);
        if (st != GUILD_STATUS_OK) {
            std::cerr << "guild kernel rollback failed: " << err << "\n";
            return 1;
        }
        guild::abi::InstalledPackageRecord rec;
        mgr.get_package(target, rec);
        std::cout << "Rolled back package '" << target << "' to version '" << rec.active_version << "'.\n";
        return 0;
    }

    if (sub == "active") {
        mgr.load_enabled_packages(err);
        const auto ops = guild::kernels::KernelRegistry::instance().list_operation_snapshots();
        const auto runners = guild::model::ModelRunnerRegistry::instance().list_runner_snapshots();
        if (json_mode) {
            std::cout << "{\n  \"runners\": [\n";
            for (size_t i = 0; i < runners.size(); ++i) {
                const auto& r = runners[i];
                std::cout << "    {\"runner_id\":\"" << r.runner_id << "\",\"arch_id\":\"" << r.arch_id
                          << "\",\"package_id\":\"" << r.package_id << "\",\"enabled\":" << (r.enabled ? "true" : "false")
                          << "}" << (i + 1 < runners.size() ? "," : "") << "\n";
            }
            std::cout << "  ],\n  \"kernels\": [\n";
            for (size_t i = 0; i < ops.size(); ++i) {
                const auto& o = ops[i];
                std::cout << "    {\"op_name\":\"" << o.op_name << "\",\"impl_id\":\"" << o.impl_id
                          << "\",\"package_id\":\"" << o.package_id << "\",\"priority\":" << o.priority
                          << ",\"enabled\":" << (o.enabled ? "true" : "false")
                          << "}" << (i + 1 < ops.size() ? "," : "") << "\n";
            }
            std::cout << "  ]\n}\n";
            return 0;
        }
        std::cout << bold() << "Active Model Runners (" << runners.size() << "):" << reset() << "\n";
        for (const auto& r : runners) {
            std::cout << "  " << std::left << std::setw(28) << r.runner_id
                      << std::setw(16) << r.arch_id
                      << std::setw(32) << r.package_id
                      << (r.enabled ? "enabled" : "disabled") << "\n";
        }
        std::cout << "\n" << bold() << "Registered Hardware Kernels (" << ops.size() << "):" << reset() << "\n";
        for (const auto& o : ops) {
            std::cout << "  " << std::left << std::setw(24) << o.op_name
                      << std::setw(38) << o.impl_id
                      << std::setw(32) << o.package_id
                      << "prio=" << std::setw(6) << o.priority
                      << (o.enabled ? "enabled" : "disabled") << "\n";
        }
        return 0;
    }

    std::cerr << "Unknown 'guild kernel' subcommand: " << sub << "\n";
    print_kernel_help();
    return 1;
}

int cmd_convert(int argc, char** argv) {
    using namespace guild::cli::ansi;
    guild::artifact::GPackBuildOptions opts;
    bool json_mode = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) opts.output_path = argv[++i];
        else if (arg == "--pack-dir" && i + 1 < argc) opts.pack_dir = argv[++i];
        else if (arg == "--tokenizer-dir" && i + 1 < argc) opts.tokenizer_dir = argv[++i];
        else if (arg == "--mtp-dir" && i + 1 < argc) opts.mtp_dir = argv[++i];
        else if (arg == "--ple-gguf" && i + 1 < argc) opts.ple_gguf = argv[++i];
        else if (arg == "--name" && i + 1 < argc) opts.model_name = argv[++i];
        else if (arg == "--shard" && i + 1 < argc) opts.additional_shards.push_back(argv[++i]);
        else if (arg == "--external-experts") opts.embed_experts = false;
        else if (arg == "--no-verify") opts.verify_after_write = false;
        else if (arg == "--json") json_mode = true;
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: guild convert <input_gguf_or_safetensors> -o <output.gpack> [options]\n"
                      << "  -o, --output <path>       Output .gpack path\n"
                      << "  --pack-dir <dir>          Native pack directory (index.txt, dense.bin, tokenizer/)\n"
                      << "  --tokenizer-dir <dir>     Tokenizer directory override\n"
                      << "  --mtp-dir <dir>           Optional MTP draft directory\n"
                      << "  --name <name>             Canonical model name\n"
                      << "  --external-experts        Record GGUF expert layout without duplicating expert blobs\n"
                      << "  --json                    Emit JSON conversion report\n";
            return 0;
        } else if (!arg.empty() && arg[0] != '-' && opts.primary_input_path.empty()) {
            opts.primary_input_path = arg;
        }
    }

    if (opts.primary_input_path.empty() || opts.output_path.empty()) {
        std::cerr << "Usage: guild convert <input> -o <output.gpack> [--pack-dir <dir>] [--external-experts] [--json]\n";
        return 1;
    }

    guild::artifact::GPackBuildReport report;
    std::string err;
    if (!guild::artifact::GPackBuilder::build(opts, report, err)) {
        std::cerr << "guild convert failed: " << err << "\n";
        return 1;
    }

    if (json_mode) {
        std::cout << "{\n"
                  << "  \"status\": \"ok\",\n"
                  << "  \"output_path\": \"" << report.output_path << "\",\n"
                  << "  \"manifest_sha256\": \"" << report.manifest_sha256 << "\",\n"
                  << "  \"total_file_size\": " << report.total_file_size << ",\n"
                  << "  \"section_count\": " << report.section_count << ",\n"
                  << "  \"tensor_count\": " << report.tensor_count << ",\n"
                  << "  \"dense_pack_bytes\": " << report.dense_pack_bytes << ",\n"
                  << "  \"native_dense_bytes\": " << report.native_dense_bytes << ",\n"
                  << "  \"expert_blob_bytes\": " << report.expert_blob_bytes << ",\n"
                  << "  \"tokenizer_bin_bytes\": " << report.tokenizer_bin_bytes << ",\n"
                  << "  \"build_ms\": " << report.build_ms << ",\n"
                  << "  \"verify_ms\": " << report.verify_ms << "\n"
                  << "}\n";
    } else {
        std::cout << bold() << "Built GPACK v1 Container" << reset() << "\n"
                  << "  Output       " << report.output_path << "\n"
                  << "  Size         " << guild::models::Downloader::format_bytes(report.total_file_size) << "\n"
                  << "  Manifest SHA " << report.manifest_sha256 << "\n"
                  << "  Sections     " << report.section_count << " (" << report.tensor_count << " tensors)\n"
                  << "  Build Time   " << std::fixed << std::setprecision(1) << report.build_ms << " ms\n";
    }
    return 0;
}

int cmd_verify(int argc, char** argv) {
    std::string path;
    bool json_mode = false;
    bool quick_mode = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") json_mode = true;
        else if (arg == "--quick") quick_mode = true;
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: guild verify <model.gpack> [--quick] [--json]\n";
            return 0;
        } else if (!arg.empty() && arg[0] != '-' && path.empty()) {
            path = arg;
        }
    }

    if (path.empty()) {
        std::cerr << "Usage: guild verify <model.gpack> [--quick] [--json]\n";
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    guild::artifact::GPackReader reader;
    std::string err;
    if (!reader.open(path, err, !quick_mode)) {
        if (json_mode) {
            std::cout << "{\"valid\":false,\"error\":\"" << err << "\"}\n";
        } else {
            std::cerr << "guild verify failed: " << err << "\n";
        }
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (json_mode) {
        std::cout << "{\n"
                  << "  \"valid\": true,\n"
                  << "  \"path\": \"" << reader.path() << "\",\n"
                  << "  \"format_version\": " << reader.header().format_version << ",\n"
                  << "  \"manifest_sha256\": \"" << reader.manifest_sha256_hex() << "\",\n"
                  << "  \"file_size\": " << reader.file_size() << ",\n"
                  << "  \"sections\": " << reader.sections().size() << ",\n"
                  << "  \"tensors\": " << reader.tensors().size() << ",\n"
                  << "  \"verify_ms\": " << ms << "\n"
                  << "}\n";
    } else {
        std::cout << "GPACK container verified: " << reader.path() << " ("
                  << reader.sections().size() << " sections, "
                  << reader.tensors().size() << " tensors, manifest "
                  << reader.manifest_sha256_hex().substr(0, 16) << "..., "
                  << std::fixed << std::setprecision(1) << ms << " ms)\n";
    }
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
        return cmd_version(argc, argv);
    }
    if (cmd == "kernel" || cmd == "plugin" || cmd == "pkg") {
        return cmd_kernel(argc, argv);
    }
    if (cmd == "pull") {
        return cmd_pull(argc, argv);
    }
    if (cmd == "list" || cmd == "ls") {
        return cmd_list(argc, argv);
    }
    if (cmd == "show") {
        return cmd_show(argc, argv);
    }
    if (cmd == "rm") {
        return cmd_rm(argc, argv);
    }
    if (cmd == "import") {
        return cmd_import(argc, argv);
    }
    if (cmd == "inspect") {
        return cmd_inspect(argc, argv);
    }
    if (cmd == "convert" || cmd == "gpack") {
        return cmd_convert(argc, argv);
    }
    if (cmd == "verify") {
        return cmd_verify(argc, argv);
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
