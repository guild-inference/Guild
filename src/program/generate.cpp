// src/program/generate.cpp - Thin CLI / process wrapper around GuildModel and GuildSession
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "guild/model/model_descriptor.hpp"
#include "guild/model/archetype.hpp"
#include "guild/memory/planner.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/artifact/gguf_reader.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace guild::runtime;

namespace {

void usage() {
    std::cout << "Usage: guild-generate [options]\n"
              << "  --pack <dir>               Model pack directory\n"
              << "  --native <file>            Primary model GGUF file\n"
              << "  --expert-profile <file>    Expert profile binary\n"
              << "  --mtp <dir>                MTP speculative draft directory\n"
              << "  --mtp-optional             Start without the MTP draft layer when VRAM cannot also hold it\n"
              << "  --ple-gguf <file>          The GGUF shard holding the PLE table (default: found in --native's)\n"
              << "  --ple-io <direct|mmap|ram> How the PLE table is read; ram locks it in RAM (rotational disks)\n"
              << "  --tokenizer <dir>          Tokenizer directory\n"
              << "  --max-context <n>          Maximum context length (default 262144)\n"
              << "  --kv <fp16|q8_0|...>       KV cache precision\n"
              << "  --kv-host-only             Host-only KV cache with bounded GPU staging\n"
              << "  --resident-budget-gib <n>  RAM expert residency budget in GiB\n"
              << "  --expert-cache <auto|n>    GPU expert cache slots\n"
              << "  --pcie-frac <f>            Fraction of missed experts offloaded from RAM (0..1; default auto)\n"
              << "  --prefill <auto|n>         Prefill chunk size\n"
              << "  --spec <n>                 Speculative draft length (default 4)\n"
              << "  --spec-min-p <p>           Speculative min probability (default 0.5)\n"
              << "  --rope-scaling <s>         none|linear|yarn: a --max-context past the model's trained context\n"
              << "  --rope-scale <f>           the extension factor (default: --max-context / trained context)\n"
              << "  --decode-strategy <s>      auto|normal|mtp|lookup|retrieval|adaptive (default auto: MTP when\n"
              << "                             --mtp is given, otherwise none)\n"
              << "  --pool-workers <n>         CPU expert pool worker count\n"
              << "  --dump-routing <file>      Append every decode layer's routed experts (tools/make_profile.py)\n"
              << "  --prompt <string>          Input prompt text\n"
              << "  --tokens <ids>             Comma-separated prompt token IDs\n"
              << "  --tokens-file <path>       File containing token IDs\n"
              << "  --max-new <n>              Maximum tokens to generate (default 512)\n"
              << "  --temperature <t>          Sampling temperature (default 0 = greedy)\n"
              << "  --top-p <p>                Top-p nucleus sampling (default 1.0)\n"
              << "  --top-k <k>                Top-k sampling (default 0)\n"
              << "  --seed <s>                 Random seed (default 0)\n"
              << "  --greedy                   Enforce greedy argmax sampling\n"
              << "  --serve                    Run in pipe-based server mode\n"
              << "Text to speech (Fish Audio Drama 3: tools/drama3_backbone_gguf.py, drama3_tower_rt.py, "
                 "drama3_codec_rt.py):\n"
              << "  --drama3 <dir>             The tower/ and codec/ runtime files: the model speaks\n"
              << "  --say <text>               The text to speak (<|speaker:N|> tags, [laughs] ...)\n"
              << "  --wav <path>               Where the audio goes (16-bit PCM, 44.1 kHz; default speech.wav)\n"
              << "  --ref-text <text>          A reference voice's transcript ...\n"
              << "  --ref-codes <path>         ... and its codes, one frame of 10 per line\n"
              << "  (sampling defaults to the model card's: --temperature 1.0 --top-p 0.9 --top-k 30)\n";
}

// 16-bit PCM mono WAV
bool write_wav(const std::string& path, const std::vector<float>& pcm, int rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f)
        return false;
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    const uint32_t data = (uint32_t) pcm.size() * 2;
    f.write("RIFF", 4);
    u32(36 + data);
    f.write("WAVEfmt ", 8);
    u32(16);
    u16(1);
    u16(1);
    u32((uint32_t) rate);
    u32((uint32_t) rate * 2);
    u16(2);
    u16(16);
    f.write("data", 4);
    u32(data);
    for (float x : pcm) {
        const float c = std::max(-1.0f, std::min(1.0f, x));
        u16((uint16_t) (int16_t) std::lrint(c * 32767.0f));
    }
    return (bool) f;
}

bool parse_token_list(const std::string& str, std::vector<int32_t>& tokens) {
    tokens.clear();
    std::string text = str;
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream iss(text);
    std::string item;
    while (iss >> item) {
        try {
            size_t used = 0;
            const int64_t id = std::stoll(item, &used);
            if (used != item.size() || id < 0 || id > INT32_MAX) {
                tokens.clear();
                return false;
            }
            tokens.push_back(static_cast<int32_t>(id));
        } catch (...) {
            tokens.clear();
            return false;
        }
    }
    return !tokens.empty();
}

// `GEN <max_new> [key=value ...] <ids>`: the keys serve/server.py's sampling_keys() writes, and decode_strategy
// (the request's speculative strategy, as --decode-strategy).  Keys this engine has no
// setting for (cvec, pcie_frac, spec_min_p) are accepted and ignored; the field without '=' is the token list.  A
// malformed number is a refused request, never an exception out of the serve loop.
bool parse_gen_line(const std::string& rest, GenerationRequest& req, std::string& err) {
    std::istringstream iss(rest);
    std::string part, tok_str;
    req.max_new_tokens = 4096;
    if (!(iss >> part)) {
        err = "GEN needs a token count";
        return false;
    }
    try {
        size_t used = 0;
        req.max_new_tokens = std::stoi(part, &used);
        if (used != part.size() || req.max_new_tokens < 1) {
            err = "invalid max_new '" + part + "'";
            return false;
        }
        while (iss >> part) {
            const size_t eq = part.find('=');
            if (eq == std::string::npos) {
                tok_str = part;
                continue;
            }
            const std::string key = part.substr(0, eq), value = part.substr(eq + 1);
            if (key == "temperature")
                req.temperature = std::stof(value);
            else if (key == "top_p")
                req.top_p = std::stof(value);
            else if (key == "top_k")
                req.top_k = std::stoi(value);
            else if (key == "min_p")
                req.min_p = std::stof(value);
            else if (key == "penalty_repeat")
                req.penalty_repeat = std::stof(value);
            else if (key == "penalty_freq")
                req.penalty_freq = std::stof(value);
            else if (key == "penalty_present")
                req.penalty_present = std::stof(value);
            else if (key == "penalty_last_n")
                req.penalty_last_n = std::stoi(value);
            else if (key == "seed")
                req.seed = std::stoull(value);
            else if (key == "decode_strategy")
                req.decode_strategy = value; // checked by the session: an unknown one is refused
        }
    } catch (const std::exception&) {
        err = "invalid value in '" + part + "'";
        return false;
    }
    if (!tok_str.empty() && !parse_token_list(tok_str, req.prompt_tokens)) {
        err = "invalid prompt token list";
        return false;
    }
    return true;
}

// The serve loop's stdin.  A reader thread owns it, so a STOP reaches the request that is running (the generation
// holds the main thread for its whole length); every other line is queued for the main thread.  A STOP with no
// request running is dropped: it was meant for one that has already finished.  End of input cancels the running
// request and ends the loop.
class ServeInput {
public:
    ServeInput() {
        // The reader blocks in getline until the parent closes the pipe, so it is detached and co-owns the state.
        std::thread([s = s_] { read(*s); }).detach();
    }
    bool next(std::string& line) {
        std::unique_lock<std::mutex> lk(s_->mu);
        s_->cv.wait(lk, [&] { return !s_->lines.empty() || s_->eof; });
        if (s_->lines.empty())
            return false;
        line = std::move(s_->lines.front());
        s_->lines.pop_front();
        return true;
    }
    void begin(std::shared_ptr<std::atomic<bool>> flag) {
        std::lock_guard<std::mutex> lk(s_->mu);
        s_->active = std::move(flag);
        if (s_->eof)
            s_->active->store(true);
    }
    void end() {
        std::lock_guard<std::mutex> lk(s_->mu);
        s_->active.reset();
    }

private:
    struct State {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> lines;
        std::shared_ptr<std::atomic<bool>> active;
        bool eof = false;
    };
    static void read(State& s) {
        // fd 0 itself, not std::cin: exit() flushes every stdio stream under its lock, and a reader blocked inside
        // getline holds stdin's, so a client that sent QUIT and kept the pipe open waited on the engine forever.
        std::string pending;
        char chunk[4096];
        auto take = [&](std::string line) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            std::lock_guard<std::mutex> lk(s.mu);
            if (line == "STOP") {
                if (s.active)
                    s.active->store(true);
                return;
            }
            s.lines.push_back(std::move(line));
            s.cv.notify_one();
        };
        for (;;) {
#if defined(_WIN32)
            const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
            const ssize_t n = ::read(0, chunk, sizeof chunk);
            if (n < 0 && errno == EINTR)
                continue;
#endif
            if (n <= 0)
                break;
            pending.append(chunk, (size_t) n);
            for (size_t nl; (nl = pending.find('\n')) != std::string::npos;) {
                take(pending.substr(0, nl));
                pending.erase(0, nl + 1);
            }
        }
        if (!pending.empty())
            take(std::move(pending)); // a last line without its newline, as getline gave it
        std::lock_guard<std::mutex> lk(s.mu);
        s.eof = true;
        if (s.active)
            s.active->store(true);
        s.cv.notify_one();
    }
    std::shared_ptr<State> s_ = std::make_shared<State>();
};

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

int main(int argc, char** argv) try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    }

    std::string pack_dir;
    std::string native_path;
    std::string profile_path;
    std::string mtp_dir;
    std::string tokenizer_dir;
    int64_t max_context = 262144;
    std::string kv_precision = "fp16";
    bool kv_host_only = false;
    int resident_budget_gib = 0;
    std::string expert_cache = "auto";
    double pcie_fraction = -1.0;
    std::string prefill = "auto";
    int spec = 4;
    float spec_min_p = 0.5f;
    std::string decode_strategy;
    std::string rope_scaling;
    bool mtp_optional = false;
    std::string ple_gguf, ple_io;
    double rope_scale = 0;
    int pool_workers = 0;
    std::string prompt;
    std::vector<int32_t> prompt_tokens;
    int max_new = 512;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    uint64_t seed = 0;
    bool greedy = true;
    bool serve_mode = false;
    bool sampling_given = false; // --temperature / --top-p / --top-k / --greedy: the speech path keeps the card's
    std::string drama3_dir, say, wav_path = "speech.wav", ref_text, ref_codes_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires an argument\n", what);
                std::exit(2);
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a == "--pack") {
            pack_dir = next("--pack");
        } else if (a == "--native") {
            native_path = next("--native");
        } else if (a == "--expert-profile") {
            profile_path = next("--expert-profile");
        } else if (a == "--mtp") {
            mtp_dir = next("--mtp");
        } else if (a == "--drama3") {
            drama3_dir = next("--drama3");
        } else if (a == "--say") {
            say = next("--say");
        } else if (a == "--wav") {
            wav_path = next("--wav");
        } else if (a == "--ref-text") {
            ref_text = next("--ref-text");
        } else if (a == "--ref-codes") {
            ref_codes_path = next("--ref-codes");
        } else if (a == "--tokenizer") {
            tokenizer_dir = next("--tokenizer");
        } else if (a == "--max-context") {
            max_context = std::stoll(next("--max-context"));
        } else if (a == "--kv") {
            kv_precision = next("--kv");
        } else if (a == "--kv-host-only") {
            kv_host_only = true;
        } else if (a == "--resident-budget-gib") {
            resident_budget_gib = std::stoi(next("--resident-budget-gib"));
        } else if (a == "--expert-cache") {
            expert_cache = next("--expert-cache");
        } else if (a == "--pcie-frac") {
            const std::string value = next("--pcie-frac");
            size_t used = 0;
            pcie_fraction = std::stod(value, &used);
            if (used != value.size() || !std::isfinite(pcie_fraction) || pcie_fraction < 0 || pcie_fraction > 1) {
                std::fprintf(stderr, "--pcie-frac requires a finite number in 0..1\n");
                return 2;
            }
        } else if (a == "--prefill") {
            prefill = next("--prefill");
        } else if (a == "--spec") {
            spec = std::stoi(next("--spec"));
        } else if (a == "--spec-min-p") {
            spec_min_p = std::stof(next("--spec-min-p"));
        } else if (a == "--decode-strategy") {
            decode_strategy = next("--decode-strategy");
        } else if (a == "--mtp-optional") {
            mtp_optional = true;
        } else if (a == "--ple-gguf") {
            ple_gguf = next("--ple-gguf");
        } else if (a == "--ple-io") {
            ple_io = next("--ple-io");
            if (ple_io != "direct" && ple_io != "mmap" && ple_io != "ram") {
                std::fprintf(stderr, "guild-generate: --ple-io takes direct, mmap or ram\n");
                return 2;
            }
        } else if (a == "--mmap-experts" || a == "--resident-experts") {
            // setup.py's low-RAM modes for the earlier engine; this one copies the experts it has RAM for and reads
            // the rest from the files either way (--resident-budget-gib caps the copy)
            std::fprintf(stderr,
                         "guild-generate: %s: the RAM copy of the experts is sized to the free RAM "
                         "(--resident-budget-gib caps it)\n",
                         a.c_str());
        } else if (a == "--rope-scaling") {
            rope_scaling = next("--rope-scaling");
        } else if (a == "--rope-scale") {
            rope_scale = std::stod(next("--rope-scale"));
        } else if (a == "--dump-routing") {
            const std::string path = next("--dump-routing");
#if defined(_WIN32)
            _putenv_s("GUILD_DUMP_ROUTING", path.c_str());
#else
            setenv("GUILD_DUMP_ROUTING", path.c_str(), 1);
#endif
        } else if (a == "--pool-workers") {
            pool_workers = std::stoi(next("--pool-workers"));
        } else if (a == "--prompt") {
            prompt = next("--prompt");
        } else if (a == "--tokens") {
            if (!parse_token_list(next("--tokens"), prompt_tokens)) {
                std::fprintf(stderr, "guild-generate: invalid prompt token list\n");
                return 2;
            }
        } else if (a == "--tokens-file") {
            std::string tpath = next("--tokens-file");
            std::ifstream tf(tpath);
            if (!tf.is_open()) {
                std::fprintf(stderr, "guild-generate: cannot open token file\n");
                return 2;
            }
            std::string content((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
            if (!parse_token_list(content, prompt_tokens)) {
                std::fprintf(stderr, "guild-generate: invalid prompt token file\n");
                return 2;
            }
        } else if (a == "--max-new") {
            max_new = std::stoi(next("--max-new"));
        } else if (a == "--temperature") {
            temperature = std::stof(next("--temperature"));
            sampling_given = true;
            if (temperature > 0.0f)
                greedy = false;
        } else if (a == "--top-p") {
            top_p = std::stof(next("--top-p"));
            sampling_given = true;
        } else if (a == "--top-k") {
            top_k = std::stoi(next("--top-k"));
            sampling_given = true;
        } else if (a == "--seed") {
            seed = std::stoull(next("--seed")); // a 64-bit seed, as a GEN line's
        } else if (a == "--greedy") {
            greedy = true;
            sampling_given = true;
            temperature = 0.0f;
        } else if (a == "--serve") {
            serve_mode = true;
        }
        // Flags of the earlier engine stay accepted (configs still pass them), but said: a mistyped option used to
        // be dropped without a word and the run went on with the default.  Its value, if it has one, is skipped.
        else if (a.rfind("--", 0) == 0) {
            const bool has_value = i + 1 < argc && argv[i + 1][0] != '-';
            std::fprintf(stderr, "guild-generate: ignoring unknown option %s%s%s (see --help)\n", a.c_str(),
                         has_value ? " " : "", has_value ? argv[i + 1] : "");
            if (has_value)
                ++i;
        } else
            std::fprintf(stderr, "guild-generate: ignoring argument %s (see --help)\n", a.c_str());
    }

    if (tokenizer_dir.empty() && !pack_dir.empty()) {
        tokenizer_dir = pack_dir + "/tokenizer";
    }

    // Hardware detection
    const auto hw = guild::cli::detect_hardware();

    // Model Descriptor
    guild::model::ModelDescriptor desc = make_default_descriptor(native_path);
    if (!native_path.empty()) {
        try {
            guild::GgufFile gguf(native_path);
            std::string d_err;
            guild::model::ModelDescriptor parsed;
            if (guild::model::ArchetypeRegistry::instance().describe_gguf(gguf, parsed, d_err)) {
                desc = parsed;
            }
        } catch (...) {}
    }
    // setup.py writes these for a context past the trained one (393216, 524288): the rope is extended and the
    // model's context grows with it.  Without them such a --max-context is refused, as before.
    if (!rope_scaling.empty() && rope_scaling != "none") {
        if (rope_scaling != "linear" && rope_scaling != "yarn") {
            std::fprintf(stderr, "guild-generate: --rope-scaling takes none, linear or yarn\n");
            return 2;
        }
        if (desc.rope.orig_ctx <= 0)
            desc.rope.orig_ctx = (double) desc.attn.context_length;
        const double factor = rope_scale > 0 ? rope_scale : std::max(1.0, (double) max_context / desc.rope.orig_ctx);
        desc.rope.type = rope_scaling;
        desc.rope.factor = factor;
        desc.attn.context_length = (int64_t) std::llround(desc.rope.orig_ctx * factor);
        std::fprintf(stderr, "guild-generate: rope %s x%.3g, context %lld (trained %.0f)\n", rope_scaling.c_str(),
                     factor, (long long) desc.attn.context_length, desc.rope.orig_ctx);
    } else if (rope_scale > 0 && rope_scale != 1) {
        std::fprintf(stderr, "guild-generate: --rope-scale needs --rope-scaling linear or yarn\n");
        return 2;
    }

    // Execution Plan
    guild::memory::PlannerOptions popts;
    popts.context_length = max_context;
    popts.kv_format = guild::memory::kv_precision_from_string(kv_precision);
    if (kv_host_only) {
        popts.kv_mode = guild::memory::KvMode::HostOnly;
    }
    if (resident_budget_gib > 0) {
        popts.resident_budget_gib = resident_budget_gib;
    }
    if (expert_cache != "auto") {
        if (expert_cache == "off")
            popts.gpu_cache_slots = 0;
        else {
            try {
                popts.gpu_cache_slots = std::stoi(expert_cache);
            } catch (...) {}
        }
    }
    if (pcie_fraction >= 0)
        popts.pcie_expert_fraction = pcie_fraction;
    if (prefill != "auto") {
        try {
            popts.prefill_chunk = std::stoi(prefill);
        } catch (...) {}
    }
    popts.spec_tokens = spec;
    if (!std::isfinite(spec_min_p) || spec_min_p < 0 || spec_min_p > 1 || pool_workers < 0) {
        std::fprintf(stderr, "--spec-min-p requires 0..1 and --pool-workers requires a nonnegative count\n");
        return 2;
    }
    popts.spec_min_p = spec_min_p;
    popts.pool_workers = pool_workers;

    auto plan = guild::memory::MemoryPlanner::plan(desc, hw, popts);
    if (!decode_strategy.empty())
        plan.decode_strategy = decode_strategy; // checked by the model load
    plan.mtp_optional = mtp_optional;

    // Model Paths
    ModelPaths paths;
    paths.primary_model_path = native_path;
    paths.pack_dir = pack_dir;
    paths.expert_profile_path = profile_path;
    paths.mtp_dir = mtp_dir;
    paths.drama3_dir = drama3_dir;
    paths.tokenizer_dir = tokenizer_dir;
    paths.ple_gguf = ple_gguf;
    paths.ple_io = ple_io;

    std::string err_msg;
    // The model and its session, an exception as an error: a VRAM shortfall can surface either way.
    std::unique_ptr<GuildModel> model;
    std::unique_ptr<GuildSession> session;
    auto load = [&]() {
        session.reset();
        model.reset();
        try {
            model = GuildModel::load(paths, desc, plan, err_msg);
            if (model) {
                session = model->create_session(plan.context_length);
                if (!session)
                    err_msg = "session creation failed";
            }
        } catch (const std::exception& e) {
            err_msg = e.what();
        }
        return model != nullptr && session != nullptr;
    };
    bool loaded = load();
    // --mtp-optional (setup.py writes it for small cards): the MTP draft layer is dropped, not the start, when the
    // VRAM does not hold it with the rest.  The model then drafts from the context (decode strategy auto).
    if (!loaded && mtp_optional && !paths.mtp_dir.empty() && err_msg.find("out of memory") != std::string::npos) {
        std::fprintf(stderr, "guild-generate: %s; --mtp-optional: loading again without the MTP draft layer\n",
                     err_msg.c_str());
        paths.mtp_dir.clear();
        loaded = load();
    }
    if (!loaded) {
        std::fprintf(stderr, "guild-generate: model load failed: %s\n", err_msg.c_str());
        return 1;
    }

    if (serve_mode) {
        std::cout << "READY " << plan.context_length << " stop\n" << std::flush;

        ServeInput input;
        std::string line;
        while (input.next(line)) {
            if (line.empty())
                continue;
            if (line == "QUIT") {
                break;
            }
            if (line.rfind("GEN ", 0) == 0) {
                GenerationRequest req;
                std::string parse_err;
                if (!parse_gen_line(line.substr(4), req, parse_err)) {
                    std::cout << "ERR " << parse_err << "\n" << std::flush;
                    continue;
                }
                req.cancel_flag = std::make_shared<std::atomic<bool>>(false);

                RuntimeTelemetry telem;
                std::string gen_err;
                auto on_tok = [](const TokenEvent& ev) -> bool {
                    std::cout << "T " << ev.token_id << "\n" << std::flush;
                    return true;
                };
                auto on_pref = [](const PrefillProgressEvent& pe) {
                    std::cout << "PP " << pe.pos << " " << pe.total << " " << pe.ms << " " << pe.tok_s << "\n"
                              << std::flush;
                };

                input.begin(req.cancel_flag);
                bool ok = session->generate(req, on_tok, telem, gen_err, on_pref);
                input.end();
                if (!ok) {
                    std::cout << "ERR " << gen_err << "\n" << std::flush;
                } else {
                    std::cout << "DONE " << telem.completion_tokens << " " << telem.prompt_tokens << " "
                              << telem.prompt_ms << " " << telem.decode_ms << " " << telem.finish_reason << " "
                              << telem.drafts_accepted << " " << telem.drafts_offered << " "
                              << telem.cached_prompt_tokens << " " << telem.gpu_cache_hits << " "
                              << telem.gpu_cache_lookups << " " << telem.ram_expert_hits << " "
                              << telem.file_expert_reads << " " << double(telem.file_read_bytes) / 1e6 << " "
                              << (telem.prompt_tokens - telem.cached_prompt_tokens) << " " << telem.offload_entries
                              << "\n"
                              << std::flush;
                }
            }
        }
        return 0;
    }

    if (model->is_speech_model()) {
        SpeechRequest sreq;
        sreq.text = say.empty() ? prompt : say;
        sreq.max_new_frames = max_new;
        if (sampling_given) {
            sreq.temperature = greedy ? 0.0f : temperature;
            sreq.top_p = top_p;
            sreq.top_k = top_k > 0 ? top_k : 30;
        }
        sreq.seed = seed;
        if (!ref_codes_path.empty()) {
            SpeechReference r;
            r.text = ref_text;
            std::ifstream in(ref_codes_path);
            int32_t v = 0;
            while (in >> v)
                r.codes.push_back(v);
            sreq.references.push_back(std::move(r));
        }
        std::vector<float> audio;
        SpeechTelemetry st;
        std::string serr;
        const bool ok = model->synthesize(
            sreq,
            [&](const SpeechChunk& c) {
                audio.insert(audio.end(), c.pcm, c.pcm + c.samples);
                return true;
            },
            st, serr);
        if (!ok) {
            std::fprintf(stderr, "guild-generate: speech failed: %s\n", serr.c_str());
            return 1;
        }
        if (!write_wav(wav_path, audio, 44100)) {
            std::fprintf(stderr, "guild-generate: cannot write %s\n", wav_path.c_str());
            return 1;
        }
        const double secs = (double) audio.size() / 44100.0;
        std::fprintf(stderr,
                     "guild-generate: %s: %.2f s of audio, %d frames (%s) in %.0f ms (%.2f frames/s, %.2fx real "
                     "time; tower %.0f ms, codec %.0f ms), prompt %d rows in %.0f ms\n",
                     wav_path.c_str(), secs, st.frames, st.finish_reason.c_str(), st.decode_ms,
                     st.decode_ms > 0 ? st.frames / (st.decode_ms / 1000.0) : 0.0,
                     st.decode_ms > 0 ? secs / (st.decode_ms / 1000.0) : 0.0, st.tower_ms, st.codec_ms, st.prompt_rows,
                     st.prompt_ms);
        return 0;
    }

    // Standard CLI generation mode
    GenerationRequest req;
    req.prompt = prompt;
    req.prompt_tokens = prompt_tokens;
    req.max_new_tokens = max_new;
    req.temperature = greedy ? 0.0f : temperature;
    req.top_p = top_p;
    req.top_k = top_k;
    req.seed = seed;

    RuntimeTelemetry telem;
    std::string gen_err;
    auto on_tok = [](const TokenEvent& ev) -> bool {
        std::cout << ev.text << std::flush;
        return true;
    };
    auto on_pref = [](const PrefillProgressEvent& pe) {
        std::cerr << "Prefill chunk " << pe.pos << "/" << pe.total << " (" << pe.tok_s << " tok/s)\n";
    };

    bool ok = session->generate(req, on_tok, telem, gen_err, on_pref);
    std::cout << "\n";
    if (!ok) {
        std::fprintf(stderr, "guild-generate: generation failed: %s\n", gen_err.c_str());
        return 1;
    }

    std::fprintf(stderr, "\nCompleted %d tokens in %.2f ms (%.2f tok/s), prompt %d tokens in %.2f ms (%.2f tok/s)\n",
                 telem.completion_tokens, telem.decode_ms, telem.decode_tok_s, telem.prompt_tokens, telem.prompt_ms,
                 telem.prompt_tok_s);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "guild-generate: %s\n", e.what());
    return 2;
}
