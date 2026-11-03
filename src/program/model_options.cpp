// src/program/model_options.cpp - the model-loading options and the load, shared by guild-generate and decision_run
// (moved out of src/program/generate.cpp unchanged, apart from the program name in the messages)
#include "guild/program/model_options.hpp"

#include "guild/artifact/gguf_reader.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/memory/planner.hpp"
#include "guild/model/archetype.hpp"
#include "guild/model/model_descriptor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace guild::program {

using runtime::GuildModel;
using runtime::ModelPaths;

namespace {

guild::model::ModelDescriptor make_default_descriptor(const std::string& path) {
    guild::model::ModelDescriptor desc;
    desc.name = "Qwen3.8-Flash-Next";
    desc.archetype = guild::model::ModelArchetype::Qwen4Exp;
    desc.arch_name = "qwen4exp";
    desc.file_path = path.empty() ? "Qwen3.8-Flash-Next" : path;
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
    return desc;
}

} // namespace

const char* model_options_usage() {
    return "  --pack <dir>               Model pack directory\n"
           "  --native <file>            Primary model GGUF file\n"
           "  --expert-profile <file>    Expert profile binary\n"
           "  --mtp <dir>                MTP speculative draft directory\n"
           "  --mtp-optional             Start without the draft layer (MTP or DFlash) when VRAM cannot also hold it\n"
           "  --dflash <dir>             A DFlash 2 drafter (config.json, model.safetensors): drafts in place of the\n"
           "                             MTP layer, which is then not loaded; up to --spec - 1 drafts a window\n"
           "  --dflash-vocab <file>      The DFlash draft head's token subset (int32 ids; default: the whole head)\n"
           "  --ple-gguf <file>          The GGUF shard holding the PLE table (default: found in --native's)\n"
           "  --ple-io <direct|mmap|ram> How the PLE table is read; ram locks it in RAM (rotational disks)\n"
           "  --tokenizer <dir>          Tokenizer directory\n"
           "  --max-context <n>          Maximum context length (default 262144)\n"
           "  --kv <fp16|q8_0|...>       KV cache precision\n"
           "  --kv-host-only             Host-only KV cache with bounded GPU staging\n"
           "  --resident-budget-gib <n>  RAM expert residency budget in GiB\n"
           "  --expert-cache <auto|n>    GPU expert cache slots\n"
           "  --pcie-frac <f>            Fraction of missed experts offloaded from RAM (0..1; default auto)\n"
           "  --prefill <auto|n>         Prefill chunk size\n"
           "  --prefill-rows <n>         Compute each prefill chunk as chunks of n tokens, streaming the experts once\n"
           "                             a chunk (e.g. --prefill 1024 --prefill-rows 512: 512's results where the\n"
           "                             expert products do not depend on rows an expert has; 0: off)\n"
           "  --dense <native|q6_k|q5_k|q4_k>\n"
           "                             The GPU's dense matrices as the GGUF has them (default), or its Q8_0 ones\n"
           "                             of 2048+ rows re-quantized at load (fewer bytes a token; changes the\n"
           "                             output slightly).  GLM-5.3-Flash: its attention matrices (default: Q6_K\n"
           "                             when Q8_0 would leave less context than asked for)\n"
           "  --hc <bf16|q8_0>           The hyper-connection down/up matrices as the pack's bf16 rounding of the\n"
           "                             GGUF's Q8_0 blocks (default), or as those blocks themselves (half the\n"
           "                             bytes a token; changes the output slightly)\n"
           "  --defer-experts <k>        Expert deferral: an MoE layer's experts past its router's top k reach the\n"
           "                             residual before the next layer's experts, not before its attention, so the\n"
           "                             CPU computes them beside the GPU and a window reads its weights once\n"
           "                             (changes the model's computation; 0: off, the default)\n"
           "  --spec <n>                 Speculative draft length (default 4)\n"
           "  --spec-min-p <p>           Speculative min probability (default 0.5)\n"
           "  --rope-scaling <s>         none|linear|yarn: a --max-context past the model's trained context\n"
           "  --rope-scale <f>           the extension factor (default: --max-context / trained context)\n"
           "  --decode-strategy <s>      auto|normal|mtp|lookup|retrieval|adaptive (default auto: MTP when\n"
           "                             --mtp is given, otherwise none)\n"
           "  --pool-workers <n>         CPU expert pool worker count\n"
           "  --dump-routing <file>      Append every decode layer's routed experts (tools/make_profile.py)\n";
}

int parse_model_option(int argc, char** argv, int& i, ModelOptions& o, const char* program) {
    const std::string a = argv[i];
    auto next = [&](const char* what) -> std::string {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "%s requires an argument\n", what);
            std::exit(2);
        }
        return argv[++i];
    };
    if (a == "--pack") {
        o.pack_dir = next("--pack");
    } else if (a == "--native") {
        o.native_path = next("--native");
    } else if (a == "--expert-profile") {
        o.profile_path = next("--expert-profile");
    } else if (a == "--mtp") {
        o.mtp_dir = next("--mtp");
    } else if (a == "--dflash") {
        o.dflash_dir = next("--dflash");
    } else if (a == "--dflash-vocab") {
        o.dflash_vocab = next("--dflash-vocab");
    } else if (a == "--drama3") {
        o.drama3_dir = next("--drama3");
    } else if (a == "--tokenizer") {
        o.tokenizer_dir = next("--tokenizer");
    } else if (a == "--max-context") {
        o.max_context = std::stoll(next("--max-context"));
    } else if (a == "--kv") {
        o.kv_precision = next("--kv");
    } else if (a == "--kv-host-only") {
        o.kv_host_only = true;
    } else if (a == "--resident-budget-gib") {
        o.resident_budget_gib = std::stoi(next("--resident-budget-gib"));
    } else if (a == "--expert-cache") {
        o.expert_cache = next("--expert-cache");
    } else if (a == "--pcie-frac") {
        const std::string value = next("--pcie-frac");
        size_t used = 0;
        o.pcie_fraction = std::stod(value, &used);
        if (used != value.size() || !std::isfinite(o.pcie_fraction) || o.pcie_fraction < 0 || o.pcie_fraction > 1) {
            std::fprintf(stderr, "--pcie-frac requires a finite number in 0..1\n");
            return 2;
        }
    } else if (a == "--prefill") {
        o.prefill = next("--prefill");
    } else if (a == "--prefill-rows") {
        const std::string value = next("--prefill-rows");
        size_t used = 0;
        o.prefill_rows = std::stoll(value, &used);
        if (used != value.size() || o.prefill_rows < 0) {
            std::fprintf(stderr, "--prefill-rows requires a token count (0: off)\n");
            return 2;
        }
    } else if (a == "--defer-experts") {
        const std::string value = next("--defer-experts");
        size_t used = 0;
        o.defer_experts = std::stoi(value, &used);
        if (used != value.size() || o.defer_experts < 0) {
            std::fprintf(stderr, "--defer-experts requires a count of experts (0: off)\n");
            return 2;
        }
    } else if (a == "--dense") {
        o.dense = next("--dense");
        if (o.dense != "native" && o.dense != "q6_k" && o.dense != "q5_k" && o.dense != "q4_k") {
            std::fprintf(stderr, "--dense requires native, q6_k, q5_k or q4_k\n");
            return 2;
        }
    } else if (a == "--hc") {
        o.hc = next("--hc");
        if (o.hc != "bf16" && o.hc != "q8_0") {
            std::fprintf(stderr, "--hc requires bf16 or q8_0\n");
            return 2;
        }
    } else if (a == "--spec") {
        o.spec = std::stoi(next("--spec"));
    } else if (a == "--spec-min-p") {
        o.spec_min_p = std::stof(next("--spec-min-p"));
    } else if (a == "--decode-strategy") {
        o.decode_strategy = next("--decode-strategy");
    } else if (a == "--mtp-optional") {
        o.mtp_optional = true;
    } else if (a == "--ple-gguf") {
        o.ple_gguf = next("--ple-gguf");
    } else if (a == "--ple-io") {
        o.ple_io = next("--ple-io");
        if (o.ple_io != "direct" && o.ple_io != "mmap" && o.ple_io != "ram") {
            std::fprintf(stderr, "%s: --ple-io takes direct, mmap or ram\n", program);
            return 2;
        }
    } else if (a == "--mmap-experts" || a == "--resident-experts") {
        // setup.py's low-RAM modes for the earlier engine; this one copies the experts it has RAM for and reads
        // the rest from the files either way (--resident-budget-gib caps the copy)
        std::fprintf(stderr,
                     "%s: %s: the RAM copy of the experts is sized to the free RAM "
                     "(--resident-budget-gib caps it)\n",
                     program, a.c_str());
    } else if (a == "--rope-scaling") {
        o.rope_scaling = next("--rope-scaling");
    } else if (a == "--rope-scale") {
        o.rope_scale = std::stod(next("--rope-scale"));
    } else if (a == "--dump-routing") {
        const std::string path = next("--dump-routing");
#if defined(_WIN32)
        _putenv_s("GUILD_DUMP_ROUTING", path.c_str());
#else
        setenv("GUILD_DUMP_ROUTING", path.c_str(), 1);
#endif
    } else if (a == "--pool-workers") {
        o.pool_workers = std::stoi(next("--pool-workers"));
    } else {
        return 0;
    }
    return 1;
}

int load_model(ModelOptions& o, const char* program, LoadedModel& out) {
    if (o.tokenizer_dir.empty() && !o.pack_dir.empty()) {
        o.tokenizer_dir = o.pack_dir + "/tokenizer";
    }

    // Hardware detection
    const auto hw = guild::cli::detect_hardware();

    // Model Descriptor
    guild::model::ModelDescriptor desc = make_default_descriptor(o.native_path);
    if (!o.native_path.empty()) {
        try {
            guild::GgufFile gguf(o.native_path);
            std::string d_err;
            guild::model::ModelDescriptor parsed;
            if (guild::model::ArchetypeRegistry::instance().describe_gguf(gguf, parsed, d_err)) {
                desc = parsed;
            }
        } catch (...) {}
    }
    // setup.py writes these for a context past the trained one (393216, 524288): the rope is extended and the
    // model's context grows with it.  Without them such a --max-context is refused, as before.
    if (!o.rope_scaling.empty() && o.rope_scaling != "none") {
        if (o.rope_scaling != "linear" && o.rope_scaling != "yarn") {
            std::fprintf(stderr, "%s: --rope-scaling takes none, linear or yarn\n", program);
            return 2;
        }
        if (desc.rope.orig_ctx <= 0)
            desc.rope.orig_ctx = (double) desc.attn.context_length;
        const double factor =
            o.rope_scale > 0 ? o.rope_scale : std::max(1.0, (double) o.max_context / desc.rope.orig_ctx);
        desc.rope.type = o.rope_scaling;
        desc.rope.factor = factor;
        desc.attn.context_length = (int64_t) std::llround(desc.rope.orig_ctx * factor);
        std::fprintf(stderr, "%s: rope %s x%.3g, context %lld (trained %.0f)\n", program, o.rope_scaling.c_str(),
                     factor, (long long) desc.attn.context_length, desc.rope.orig_ctx);
    } else if (o.rope_scale > 0 && o.rope_scale != 1) {
        std::fprintf(stderr, "%s: --rope-scale needs --rope-scaling linear or yarn\n", program);
        return 2;
    }

    // Execution Plan
    guild::memory::PlannerOptions popts;
    popts.context_length = o.max_context;
    popts.kv_format = guild::memory::kv_precision_from_string(o.kv_precision);
    if (o.kv_host_only) {
        popts.kv_mode = guild::memory::KvMode::HostOnly;
    }
    if (o.resident_budget_gib > 0) {
        popts.resident_budget_gib = o.resident_budget_gib;
    }
    if (o.expert_cache != "auto") {
        if (o.expert_cache == "off")
            popts.gpu_cache_slots = 0;
        else {
            try {
                popts.gpu_cache_slots = std::stoi(o.expert_cache);
            } catch (...) {}
        }
    }
    if (o.pcie_fraction >= 0)
        popts.pcie_expert_fraction = o.pcie_fraction;
    if (o.prefill != "auto") {
        try {
            popts.prefill_chunk = std::stoi(o.prefill);
        } catch (...) {}
    }
    popts.spec_tokens = o.spec;
    if (!std::isfinite(o.spec_min_p) || o.spec_min_p < 0 || o.spec_min_p > 1 || o.pool_workers < 0) {
        std::fprintf(stderr, "--spec-min-p requires 0..1 and --pool-workers requires a nonnegative count\n");
        return 2;
    }
    popts.spec_min_p = o.spec_min_p;
    popts.pool_workers = o.pool_workers;

    auto plan = guild::memory::MemoryPlanner::plan(desc, hw, popts);
    if (!o.decode_strategy.empty())
        plan.decode_strategy = o.decode_strategy; // checked by the model load
    plan.mtp_optional = o.mtp_optional;
    plan.prefill_rows = o.prefill_rows;
    plan.dense_type = o.dense == "q6_k" ? 14 : o.dense == "q5_k" ? 13 : o.dense == "q4_k" ? 12 : 0;
    plan.hc_q8 = o.hc == "q8_0";
    plan.defer_experts = o.defer_experts;

    // Model Paths
    ModelPaths paths;
    paths.primary_model_path = o.native_path;
    paths.pack_dir = o.pack_dir;
    paths.expert_profile_path = o.profile_path;
    paths.mtp_dir = o.mtp_dir;
    paths.dflash_dir = o.dflash_dir;
    paths.dflash_vocab = o.dflash_vocab;
    paths.drama3_dir = o.drama3_dir;
    paths.tokenizer_dir = o.tokenizer_dir;
    paths.ple_gguf = o.ple_gguf;
    paths.ple_io = o.ple_io;

    std::string err_msg;
    // The model and its session, an exception as an error: a VRAM shortfall can surface either way.
    auto& model = out.model;
    auto& session = out.session;
    int64_t& context = out.context;
    context = plan.context_length;
    auto load = [&]() {
        session.reset();
        model.reset();
        try {
            model = GuildModel::load(paths, desc, plan, err_msg);
            if (model) {
                // the context the model holds, at most the one asked for (GLM-5.3-Flash's engine fits it to the VRAM)
                const int64_t held = model->plan().context_length;
                context = held > 0 ? std::min<int64_t>(plan.context_length, held) : plan.context_length;
                session = model->create_session(context);
                if (!session)
                    err_msg = "session creation failed";
            }
        } catch (const std::exception& e) {
            err_msg = e.what();
        }
        return model != nullptr && session != nullptr;
    };
    bool loaded = load();
    // --mtp-optional (setup.py writes it for small cards): the draft layer (MTP or DFlash) is dropped, not the start,
    // when the VRAM does not hold it with the rest.  The model then drafts from the context (decode strategy auto).
    if (!loaded && o.mtp_optional && (!paths.mtp_dir.empty() || !paths.dflash_dir.empty()) &&
        err_msg.find("out of memory") != std::string::npos) {
        std::fprintf(stderr, "%s: %s; --mtp-optional: loading again without the draft layer\n", program,
                     err_msg.c_str());
        paths.mtp_dir.clear();
        paths.dflash_dir.clear();
        loaded = load();
    }
    if (!loaded) {
        std::fprintf(stderr, "%s: model load failed: %s\n", program, err_msg.c_str());
        return 1;
    }
    return 0;
}

} // namespace guild::program
