// src/program/generate.cpp - Thin CLI / process wrapper around GuildModel and GuildSession
#include "guild/program/model_options.hpp"
#include "guild/runtime/model.hpp"
#include "guild/runtime/session.hpp"
#include "guild/model/model_descriptor.hpp"
#include "guild/model/archetype.hpp"
#include "guild/memory/planner.hpp"
#include "guild/cli/hardware.hpp"
#include "guild/artifact/gguf_reader.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
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
#include <iterator>
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
              << guild::program::model_options_usage() << "  --prompt <string>          Input prompt text\n"
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
              << "  --ref-wav <path>           ... and its audio (a WAV file, any rate), or\n"
              << "  --ref-codes <path>         ... its codes, one frame of 10 per line\n"
              << "  --save-codes <path>        Write the reference's codes (from --ref-wav) for --ref-codes\n"
              << "  (sampling defaults to the model card's: --temperature 1.0 --top-p 0.9 --top-k 30)\n";
}

std::string b64_encode(const unsigned char* p, size_t n) {
    static const char* k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v =
            (uint32_t) p[i] << 16 | (i + 1 < n ? (uint32_t) p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        out += k[v >> 18];
        out += k[(v >> 12) & 63];
        out += i + 1 < n ? k[(v >> 6) & 63] : '=';
        out += i + 2 < n ? k[v & 63] : '=';
    }
    return out;
}

bool b64_decode(const std::string& in, std::string& out) {
    out.clear();
    uint32_t v = 0;
    int bits = 0;
    for (char c : in) {
        int d;
        if (c >= 'A' && c <= 'Z')
            d = c - 'A';
        else if (c >= 'a' && c <= 'z')
            d = c - 'a' + 26;
        else if (c >= '0' && c <= '9')
            d = c - '0' + 52;
        else if (c == '+' || c == '-')
            d = 62;
        else if (c == '/' || c == '_')
            d = 63;
        else if (c == '=')
            break;
        else
            return false;
        v = v << 6 | (uint32_t) d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += (char) ((v >> bits) & 0xFF);
        }
    }
    return true;
}

// SAY [max_new=N] [temperature=T] [top_p=P] [top_k=K] [seed=S] [chunk=F] text=<base64 UTF-8>
//     [ref=<base64 transcript>:<comma-separated codes, [frames][10]>]...
bool parse_say_line(const std::string& rest, SpeechRequest& req, std::string& err) {
    std::istringstream iss(rest);
    std::string part, raw;
    bool have_text = false;
    try {
        while (iss >> part) {
            const size_t eq = part.find('=');
            if (eq == std::string::npos) {
                err = "SAY: expected key=value, got '" + part + "'";
                return false;
            }
            const std::string key = part.substr(0, eq), value = part.substr(eq + 1);
            if (key == "max_new")
                req.max_new_frames = std::stoi(value);
            else if (key == "temperature")
                req.temperature = std::stof(value);
            else if (key == "top_p")
                req.top_p = std::stof(value);
            else if (key == "top_k")
                req.top_k = std::stoi(value);
            else if (key == "seed")
                req.seed = std::stoull(value);
            else if (key == "chunk")
                req.chunk_frames = std::stoi(value);
            else if (key == "text") {
                if (!b64_decode(value, req.text)) {
                    err = "SAY: text is not base64";
                    return false;
                }
                have_text = true;
            } else if (key == "ref") {
                const size_t colon = value.find(':');
                SpeechReference r;
                if (colon == std::string::npos || !b64_decode(value.substr(0, colon), r.text)) {
                    err = "SAY: ref=<base64 transcript>:<codes>";
                    return false;
                }
                std::string codes = value.substr(colon + 1);
                std::replace(codes.begin(), codes.end(), ',', ' ');
                std::istringstream cs(codes);
                int32_t c;
                while (cs >> c)
                    r.codes.push_back(c);
                req.references.push_back(std::move(r));
            } else {
                err = "SAY: unknown key " + key;
                return false;
            }
        }
    } catch (const std::exception&) {
        err = "SAY: bad value in '" + part + "'";
        return false;
    }
    if (!have_text) {
        err = "SAY needs text=<base64>";
        return false;
    }
    return true;
}

// a WAV file's samples, mono (channels averaged) in [-1, 1]: PCM 16 / 24 / 32-bit or IEEE float 32-bit
bool read_wav(const std::string& path, std::vector<float>& out, int& rate, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto u16 = [&](size_t at) { return (uint32_t) b[at] | ((uint32_t) b[at + 1] << 8); };
    auto u32 = [&](size_t at) { return u16(at) | (u16(at + 2) << 16); };
    if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) || std::memcmp(b.data() + 8, "WAVE", 4)) {
        err = path + " is not a WAV file";
        return false;
    }
    int fmt = 0, channels = 0, bits = 0;
    for (size_t at = 12; at + 8 <= b.size();) {
        uint32_t size = u32(at + 4);
        const size_t body = at + 8;
        if (body + size > b.size()) {
            if (std::memcmp(b.data() + at, "data", 4))
                break;
            size = (uint32_t) (b.size() - body); // a streamed WAV's open-ended data chunk (size 0xFFFFFFFF): the rest
        }
        if (!std::memcmp(b.data() + at, "fmt ", 4) && size >= 16) {
            fmt = (int) u16(body);
            channels = (int) u16(body + 2);
            rate = (int) u32(body + 4);
            bits = (int) u16(body + 14);
            if (fmt == 0xFFFE && size >= 26)
                fmt = (int) u16(body + 24); // WAVE_FORMAT_EXTENSIBLE: the subformat's first two bytes
        } else if (!std::memcmp(b.data() + at, "data", 4)) {
            const int bytes = bits / 8;
            if (channels < 1 || !((fmt == 1 && (bits == 16 || bits == 24 || bits == 32)) || (fmt == 3 && bits == 32))) {
                err = path + ": only 16/24/32-bit PCM and 32-bit float WAV are read";
                return false;
            }
            const size_t frames = size / (size_t) (bytes * channels);
            out.assign(frames, 0.0f);
            for (size_t i = 0; i < frames; ++i) {
                double acc = 0.0;
                for (int c = 0; c < channels; ++c) {
                    const size_t p = body + (i * channels + c) * bytes;
                    if (fmt == 3) {
                        float v;
                        std::memcpy(&v, &b[p], 4);
                        acc += v;
                    } else if (bits == 16) {
                        acc += (int16_t) u16(p) / 32768.0;
                    } else if (bits == 24) {
                        const int32_t v = (int32_t) ((u16(p) | ((uint32_t) b[p + 2] << 16)) << 8) >> 8;
                        acc += v / 8388608.0;
                    } else {
                        acc += (int32_t) u32(p) / 2147483648.0;
                    }
                }
                out[i] = (float) (acc / channels);
            }
            return true;
        }
        at = body + size + (size & 1);
    }
    err = path + ": no audio data";
    return false;
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

    guild::program::ModelOptions mo;
    std::string prompt;
    std::vector<int32_t> prompt_tokens;
    int max_new = 512;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    uint64_t seed = 0;
    bool greedy = true;
    bool serve_mode = false;
    // the flags given: the speech path keeps the model card's sampling (1.0 / 0.9 / 30) and 2048 frames for the rest
    bool temperature_given = false, top_p_given = false, top_k_given = false, max_new_given = false;
    std::string say, wav_path = "speech.wav", ref_text, ref_codes_path, ref_wav_path, save_codes_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires an argument\n", what);
                std::exit(2);
            }
            return argv[++i];
        };

        if (const int r = guild::program::parse_model_option(argc, argv, i, mo, "guild-generate"); r == 2) {
            return 2;
        } else if (r == 1) {
            continue;
        }
        if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a == "--say") {
            say = next("--say");
        } else if (a == "--wav") {
            wav_path = next("--wav");
        } else if (a == "--ref-text") {
            ref_text = next("--ref-text");
        } else if (a == "--ref-codes") {
            ref_codes_path = next("--ref-codes");
        } else if (a == "--ref-wav") {
            ref_wav_path = next("--ref-wav");
        } else if (a == "--save-codes") {
            save_codes_path = next("--save-codes");
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
            max_new_given = true;
        } else if (a == "--temperature") {
            temperature = std::stof(next("--temperature"));
            temperature_given = true;
            if (temperature > 0.0f)
                greedy = false;
        } else if (a == "--top-p") {
            top_p = std::stof(next("--top-p"));
            top_p_given = true;
        } else if (a == "--top-k") {
            top_k = std::stoi(next("--top-k"));
            top_k_given = true;
        } else if (a == "--seed") {
            seed = std::stoull(next("--seed")); // a 64-bit seed, as a GEN line's
        } else if (a == "--greedy") {
            greedy = true;
            temperature_given = true;
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

    guild::program::LoadedModel loaded;
    if (const int rc = guild::program::load_model(mo, "guild-generate", loaded); rc != 0)
        return rc;
    auto& model = loaded.model;
    auto& session = loaded.session;
    const int64_t context = loaded.context;

    if (serve_mode) {
        std::cout << "READY " << context << " stop" << (model->is_speech_model() ? " speech" : "") << "\n"
                  << std::flush;

        ServeInput input;
        std::string line;
        while (input.next(line)) {
            if (line.empty())
                continue;
            if (line == "QUIT") {
                break;
            }
            // Text to speech (a --drama3 model).  VOICE <sample rate> <base64 float32 samples> -> CODES <frames>
            // <codes>; SAY ... -> A <base64 16-bit PCM> per chunk, then SDONE <frames> <prompt rows> <prompt ms>
            // <decode ms> <finish reason> <sample rate>
            if (line.rfind("VOICE ", 0) == 0 || line.rfind("SAY ", 0) == 0) {
                if (!model->is_speech_model()) {
                    std::cout << "ERR this model does not speak (start it with --drama3)\n" << std::flush;
                    continue;
                }
                if (line.rfind("VOICE ", 0) == 0) {
                    std::istringstream vs(line.substr(6));
                    int rate = 0;
                    std::string b64, bytes, verr;
                    std::vector<int32_t> codes;
                    if (!(vs >> rate >> b64) || !b64_decode(b64, bytes) || bytes.size() % 4) {
                        std::cout << "ERR VOICE <rate> <base64 float32 samples>\n" << std::flush;
                        continue;
                    }
                    std::vector<float> pcm(bytes.size() / 4);
                    std::memcpy(pcm.data(), bytes.data(), bytes.size());
                    if (!model->encode_voice(pcm.data(), pcm.size(), rate, codes, verr)) {
                        std::cout << "ERR " << verr << "\n" << std::flush;
                        continue;
                    }
                    std::cout << "CODES " << codes.size() / 10 << " ";
                    for (size_t i = 0; i < codes.size(); ++i)
                        std::cout << (i ? "," : "") << codes[i];
                    std::cout << "\n" << std::flush;
                    continue;
                }
                SpeechRequest sreq;
                std::string perr;
                if (!parse_say_line(line.substr(4), sreq, perr)) {
                    std::cout << "ERR " << perr << "\n" << std::flush;
                    continue;
                }
                sreq.cancel_flag = std::make_shared<std::atomic<bool>>(false);
                SpeechTelemetry st;
                std::string serr;
                int rate = 44100;
                std::vector<int16_t> pcm16;
                input.begin(sreq.cancel_flag);
                const bool ok = model->synthesize(
                    sreq,
                    [&](const SpeechChunk& c) {
                        rate = c.sample_rate;
                        if (c.samples == 0)
                            return true;
                        pcm16.resize(c.samples);
                        for (size_t i = 0; i < c.samples; ++i)
                            pcm16[i] = (int16_t) std::lrint(std::max(-1.0f, std::min(1.0f, c.pcm[i])) * 32767.0f);
                        std::cout << "A "
                                  << b64_encode(reinterpret_cast<const unsigned char*>(pcm16.data()), c.samples * 2)
                                  << "\n"
                                  << std::flush;
                        return true;
                    },
                    st, serr);
                input.end();
                if (!ok)
                    std::cout << "ERR " << serr << "\n" << std::flush;
                else
                    std::cout << "SDONE " << st.frames << " " << st.prompt_rows << " " << st.prompt_ms << " "
                              << st.decode_ms << " " << st.finish_reason << " " << rate << "\n"
                              << std::flush;
                continue;
            }
            if (model->is_speech_model() && (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0)) {
                std::cout << "ERR this is a text-to-speech model: send SAY (/v1/audio/speech), not text generation\n"
                          << std::flush;
                continue;
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
        if (max_new_given)
            sreq.max_new_frames = max_new;
        if (temperature_given)
            sreq.temperature = greedy ? 0.0f : temperature;
        if (top_p_given)
            sreq.top_p = top_p;
        if (top_k_given)
            sreq.top_k = top_k;
        sreq.seed = seed;
        if (!ref_codes_path.empty() || !ref_wav_path.empty()) {
            SpeechReference r;
            r.text = ref_text;
            if (!ref_codes_path.empty()) {
                std::ifstream in(ref_codes_path);
                int32_t v = 0;
                while (in >> v)
                    r.codes.push_back(v);
            } else {
                std::vector<float> audio;
                int rate = 0;
                std::string werr;
                if (!read_wav(ref_wav_path, audio, rate, werr)) {
                    std::fprintf(stderr, "guild-generate: %s\n", werr.c_str());
                    return 1;
                }
                const auto t0 = std::chrono::steady_clock::now();
                if (!model->encode_voice(audio.data(), audio.size(), rate, r.codes, werr)) {
                    std::fprintf(stderr, "guild-generate: encoding the reference failed: %s\n", werr.c_str());
                    return 1;
                }
                std::fprintf(stderr, "guild-generate: %s: %.2f s at %d Hz -> %zu frames in %.0f ms\n",
                             ref_wav_path.c_str(), (double) audio.size() / rate, rate, r.codes.size() / 10,
                             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                if (!save_codes_path.empty()) {
                    std::ofstream out(save_codes_path);
                    for (size_t i = 0; i < r.codes.size(); ++i)
                        out << r.codes[i] << ((i + 1) % 10 ? ' ' : '\n');
                }
            }
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
