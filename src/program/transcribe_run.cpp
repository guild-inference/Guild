// src/program/transcribe_run.cpp - Nemotron 3.5 Transcribe: offline transcription of WAV clips, greedy, on the GTX
// 1070 + dual Xeon class of machine (the audio model on the GPU, the language model on the GPU and the CPU).
//
//   transcribe_run --model <LM GGUF> --audio <runtime dir> --tokenizer <tokenizer.json>
//                  (--clips clips.jsonl | file.wav ...) [--out out.jsonl] [--mtp K] [--no-mtp] [--max-new N]
//                  [--repeat N] [--threads N] [--max-context N] [--mtp-rms]
//
// The request is the checkpoint's default transcription request (verbatim, thinking off): the runtime dir's
// prompt.json holds its token ids around the audio, one audio row per 80 ms (tools/nemotron_audio_rt.py).  Decoding
// is greedy (temperature 0) up to an end-of-turn token or --max-new tokens; --mtp K drafts K tokens a step with the
// MTP block and verifies them (the output is the greedy output: a verify window's rows are one-token steps).
// --no-mtp loads no MTP block.  One JSON line per clip and run: ids, text, and the time of each stage.
#include "guild/audio/encoder.hpp"
#include "guild/audio/frontend.hpp"
#include "guild/core/nemotron_h.hpp"
#include "guild/server/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using guild::server::json::JsonValue;

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

/// A byte-level BPE vocabulary (tokenizer.json): token ids to text, special tokens skipped
class Vocab {
public:
    bool load(const std::string& path, std::string& err) {
        std::string text;
        JsonValue j;
        if (!read_file(path, text) || !JsonValue::parse(text, j, err)) {
            err = "cannot read " + path + (err.empty() ? "" : ": " + err);
            return false;
        }
        // GPT-2's bytes_to_unicode, inverted
        std::vector<int> bs;
        for (int b = '!'; b <= '~'; ++b)
            bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b)
            bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b)
            bs.push_back(b);
        std::vector<int> cs = bs;
        for (int b = 0, n = 0; b < 256; ++b)
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n++);
            }
        for (size_t i = 0; i < bs.size(); ++i)
            byte_of_[cs[i]] = bs[i];
        const JsonValue& vocab = j["model"]["vocab"];
        for (const auto& kv : vocab.obj_val) {
            const int64_t id = kv.second.as_int(-1);
            if (id < 0)
                continue;
            if ((size_t) id >= tok_.size())
                tok_.resize((size_t) id + 1);
            tok_[(size_t) id] = to_bytes(kv.first);
        }
        for (const JsonValue& t : j["added_tokens"].arr_val) {
            const int64_t id = t["id"].as_int(-1);
            if (id < 0)
                continue;
            if ((size_t) id >= tok_.size())
                tok_.resize((size_t) id + 1);
            tok_[(size_t) id] = t["content"].as_string("");
            if (t["special"].as_bool(false)) {
                if ((size_t) id >= special_.size())
                    special_.resize((size_t) id + 1, false);
                special_[(size_t) id] = true;
            }
        }
        return !tok_.empty();
    }
    std::string decode(const std::vector<int32_t>& ids) const {
        std::string s;
        for (int32_t id : ids)
            if (id >= 0 && (size_t) id < tok_.size() && !((size_t) id < special_.size() && special_[(size_t) id]))
                s += tok_[(size_t) id];
        return s;
    }

private:
    /// a vocabulary entry (UTF-8, one character per byte) -> its bytes
    std::string to_bytes(const std::string& u) const {
        std::string out;
        for (size_t i = 0; i < u.size();) {
            unsigned c = (unsigned char) u[i];
            int len = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            unsigned cp = len == 1 ? c : c & (0xFF >> (len + 1));
            for (int k = 1; k < len && i + k < u.size(); ++k)
                cp = cp << 6 | ((unsigned char) u[i + k] & 0x3F);
            i += len;
            auto it = byte_of_.find((int) cp);
            out += it != byte_of_.end() ? (char) it->second : '?';
        }
        return out;
    }
    std::map<int, int> byte_of_;
    std::vector<std::string> tok_;
    std::vector<bool> special_;
};

std::string json_escape(const std::string& s) {
    return JsonValue(s).dump();
}

struct Clip {
    std::string id, wav;
};

int argmax(const float* x, int64_t n) {
    return (int) (std::max_element(x, x + n) - x); // the first of equal maxima, as torch's argmax
}

} // namespace

int main(int argc, char** argv) {
    std::string model, audio_dir, tok_path, clips_path, out_path, rows_path;
    std::vector<std::string> wavs;
    int mtp_k = 0, max_new = 256, repeat = 1;
    guild::core::NemotronHOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model")
            model = next();
        else if (a == "--audio")
            audio_dir = next();
        else if (a == "--tokenizer")
            tok_path = next();
        else if (a == "--clips")
            clips_path = next();
        else if (a == "--out")
            out_path = next();
        else if (a == "--rows") // a debugging input: the whole prompt as float32 embedding rows, no audio
            rows_path = next();
        else if (a == "--mtp")
            mtp_k = std::atoi(next());
        else if (a == "--no-mtp")
            opt.mtp = false;
        else if (a == "--mtp-rms")
            opt.mtp_layer_norm = false;
        else if (a == "--max-new")
            max_new = std::atoi(next());
        else if (a == "--repeat")
            repeat = std::max(1, std::atoi(next()));
        else if (a == "--threads")
            opt.threads = std::atoi(next());
        else if (a == "--max-context")
            opt.max_context = std::atoll(next());
        else if (a.rfind("--", 0) == 0) {
            std::fprintf(stderr, "transcribe_run: unknown option %s\n", a.c_str());
            return 2;
        } else
            wavs.push_back(a);
    }
    if (model.empty() || audio_dir.empty() || tok_path.empty() ||
        (clips_path.empty() && wavs.empty() && rows_path.empty())) {
        std::fprintf(stderr, "usage: transcribe_run --model <LM GGUF> --audio <runtime dir> --tokenizer "
                             "<tokenizer.json> (--clips clips.jsonl | file.wav ...) [--out out.jsonl] [--mtp K] "
                             "[--no-mtp] [--max-new N] [--repeat N]\n");
        return 2;
    }
    std::vector<Clip> clips;
    std::string err;
    if (!clips_path.empty()) {
        std::ifstream f(clips_path);
        const std::string base = std::filesystem::path(clips_path).parent_path().string();
        for (std::string line; std::getline(f, line);) {
            JsonValue j;
            if (line.empty() || !JsonValue::parse(line, j, err))
                continue;
            clips.push_back({j["id"].as_string(""), base + "/" + j["wav"].as_string("")});
        }
    }
    for (const std::string& w : wavs)
        clips.push_back({std::filesystem::path(w).stem().string(), w});
    if (!rows_path.empty())
        clips.push_back({"rows", rows_path});

    // the request's ids around the audio
    std::vector<int32_t> prefix, suffix, eos;
    {
        std::string text;
        JsonValue j;
        if (!read_file(audio_dir + "/prompt.json", text) || !JsonValue::parse(text, j, err)) {
            std::fprintf(stderr, "transcribe_run: cannot read %s/prompt.json\n", audio_dir.c_str());
            return 1;
        }
        for (const JsonValue& v : j["prefix_ids"].arr_val)
            prefix.push_back((int32_t) v.as_int());
        for (const JsonValue& v : j["suffix_ids"].arr_val)
            suffix.push_back((int32_t) v.as_int());
        for (const JsonValue& v : j["eos_ids"].arr_val)
            eos.push_back((int32_t) v.as_int());
    }
    Vocab vocab;
    if (!vocab.load(tok_path, err)) {
        std::fprintf(stderr, "transcribe_run: %s\n", err.c_str());
        return 1;
    }

    auto t0 = std::chrono::steady_clock::now();
    guild::audio::AudioEncoder enc;
    if (!enc.load(audio_dir, err)) {
        std::fprintf(stderr, "transcribe_run: %s\n", err.c_str());
        return 1;
    }
    guild::core::NemotronHEngine lm;
    if (!lm.load(model, opt, err)) {
        std::fprintf(stderr, "transcribe_run: %s\n", err.c_str());
        return 1;
    }
    // the audio model's float32 matrices kept, when the language model leaves room for them
    const bool wide = enc.keep_wide(uint64_t(512) << 20);
    std::fprintf(stderr, "transcribe_run: loaded in %.1f s; audio model %.0f MiB%s, %s", ms_since(t0) / 1000.0,
                 enc.vram_bytes() / 1048576.0, wide ? " (float32 matrices kept)" : "", lm.placement().c_str());
    if (mtp_k > 0 && !lm.has_mtp()) {
        std::fprintf(stderr, "transcribe_run: --mtp needs the MTP block\n");
        return 1;
    }
    mtp_k = std::min(mtp_k, lm.max_drafts());
    const int64_t V = lm.geometry().n_vocab, N = lm.geometry().n_embd;
    std::vector<float> logits((size_t) (lm.max_drafts() + 1) * V);
    std::FILE* out = out_path.empty() ? nullptr : std::fopen(out_path.c_str(), "w");
    const auto& cfg = enc.mel_config();
    for (const Clip& c : clips)
        for (int r = 0; r < repeat; ++r) {
            const auto c0 = std::chrono::steady_clock::now();
            guild::audio::Wave w;
            guild::audio::Features f;
            std::vector<int32_t> ids;
            std::vector<float> rows;
            double seconds = 0, fe_ms = 0, enc_ms = 0;
            int n_audio = 0;
            if (c.id == "rows") {
                std::string bytes;
                if (!read_file(c.wav, bytes)) {
                    std::fprintf(stderr, "transcribe_run: cannot read %s\n", c.wav.c_str());
                    return 1;
                }
                rows.resize(bytes.size() / 4);
                std::memcpy(rows.data(), bytes.data(), rows.size() * 4);
                ids.assign(rows.size() / (size_t) N, -1);
            } else {
                if (!guild::audio::read_wav(c.wav, w, err)) {
                    std::fprintf(stderr, "transcribe_run: %s\n", err.c_str());
                    return 1;
                }
                seconds = double(w.samples.size()) / w.rate;
                if (w.rate != cfg.rate)
                    w.samples = guild::audio::resample(w.samples, w.rate, cfg.rate);
                if (!guild::audio::log_mel(w.samples.data(), (int64_t) w.samples.size(), cfg, f, err)) {
                    std::fprintf(stderr, "transcribe_run: %s\n", err.c_str());
                    return 1;
                }
                fe_ms = ms_since(c0);
                std::vector<float> emb;
                if (!enc.encode(f, emb, err)) {
                    std::fprintf(stderr, "transcribe_run: %s: %s\n", c.id.c_str(), err.c_str());
                    return 1;
                }
                enc_ms = ms_since(c0) - fe_ms;
                // the prompt: prefix ids, one row per audio embedding, suffix ids
                n_audio = (int) (emb.size() / (size_t) N);
                ids = prefix;
                ids.insert(ids.end(), (size_t) n_audio, -1);
                ids.insert(ids.end(), suffix.begin(), suffix.end());
                rows.assign(ids.size() * (size_t) N, 0.f);
                std::memcpy(rows.data() + prefix.size() * (size_t) N, emb.data(), emb.size() * sizeof(float));
            }
            lm.reset();
            const auto p0 = std::chrono::steady_clock::now();
            if (!lm.forward(ids.data(), rows.data(), (int) ids.size(), 0, logits.data(), err)) {
                std::fprintf(stderr, "transcribe_run: %s: %s\n", c.id.c_str(), err.c_str());
                return 1;
            }
            const double prefill_ms = ms_since(p0);
            if (std::getenv("TRANSCRIBE_TOP5")) {
                std::vector<int> idx((size_t) V);
                for (int64_t i = 0; i < V; ++i)
                    idx[(size_t) i] = (int) i;
                std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                                  [&](int x, int y) { return logits[(size_t) x] > logits[(size_t) y]; });
                std::fprintf(stderr, "first step top5:");
                for (int i = 0; i < 5; ++i)
                    std::fprintf(stderr, " %d:%.4f", idx[(size_t) i], logits[(size_t) idx[(size_t) i]]);
                std::fprintf(stderr, "\n");
            }
            // TRANSCRIBE_VERIFY_TEST="a b" (debugging, with GUILD_NEMOTRON_DUMP): verify([a, b]) after the prompt, then
            // the prompt again and step(a), step(b): the dump's windows compare row 1 of the verify with the step
            if (const char* vt = std::getenv("TRANSCRIBE_VERIFY_TEST")) {
                int32_t a = 0, b = 0;
                std::sscanf(vt, "%d %d", &a, &b);
                const int32_t win[2] = {a, b};
                std::vector<float> lg((size_t) 2 * V);
                const int64_t p0 = (int64_t) ids.size();
                if (!lm.verify(win, 2, p0, lg.data(), err) || !lm.accept(2, err))
                    return 1;
                lm.reset();
                if (!lm.forward(ids.data(), rows.data(), (int) ids.size(), 0, logits.data(), err) ||
                    !lm.step(a, p0, logits.data(), err) || !lm.step(b, p0 + 1, logits.data(), err))
                    return 1;
                float d = 0;
                for (int64_t i = 0; i < V; ++i)
                    d = std::max(d, std::fabs(lg[(size_t) V + (size_t) i] - logits[(size_t) i]));
                std::fprintf(stderr, "verify test: row 1 against step(b): max |d logit| %g\n", d);
                return 0;
            }
            // TRANSCRIBE_CHECK_SPEC=1 (debugging): a greedy pass first, its logits kept; then the drafted pass compares
            // every verified row whose prefix is greedy's with greedy's logits at that position
            std::vector<std::vector<float>> greedy_logits;
            std::vector<int32_t> greedy_ids;
            if (std::getenv("TRANSCRIBE_CHECK_SPEC") && mtp_k > 0) {
                std::vector<float> lg(logits.begin(), logits.begin() + V);
                int64_t gp = (int64_t) ids.size();
                for (int i = 0; i < max_new; ++i) {
                    greedy_logits.push_back(lg);
                    const int32_t t = argmax(lg.data(), V);
                    greedy_ids.push_back(t);
                    if (std::find(eos.begin(), eos.end(), t) != eos.end() || !lm.step(t, gp++, lg.data(), err))
                        break;
                }
                lm.reset();
                if (!lm.forward(ids.data(), rows.data(), (int) ids.size(), 0, logits.data(), err))
                    return 1;
                float d = 0;
                for (int64_t i = 0; i < V; ++i)
                    d = std::max(d, std::fabs(logits[(size_t) i] - greedy_logits[0][(size_t) i]));
                std::fprintf(stderr, "check: prompt logits again: max |d| %g\n", d);
            }
            // greedy decoding, with or without drafts
            const auto d0 = std::chrono::steady_clock::now();
            std::vector<int32_t> gen;
            int64_t pos = (int64_t) ids.size();
            int32_t next = argmax(logits.data(), V);
            int drafted = 0, accepted = 0, steps = 0;
            double t_draft = 0, t_verify = 0, t_accept = 0; // milliseconds in each call of the drafted path
            double t_cpu = 0; // the CPU experts' part of the decode steps or the verify windows
            // TRANSCRIBE_FORCE=FILE (debugging): feed these ids instead of the argmax and print, per step, the forced
            // token's logit minus the largest other one
            std::vector<int32_t> forced;
            if (const char* fp = std::getenv("TRANSCRIBE_FORCE")) {
                std::ifstream ff(fp);
                for (int32_t t; ff >> t;)
                    forced.push_back(t);
            }
            auto margin = [&](const float* lg, int32_t tok) {
                float best = -INFINITY;
                for (int64_t i = 0; i < V; ++i)
                    if (i != tok)
                        best = std::max(best, lg[i]);
                return lg[tok] - best;
            };
            if (!forced.empty()) {
                std::fprintf(stderr, "forced margins:");
                for (size_t i = 0; i < forced.size(); ++i) {
                    std::fprintf(stderr, " %.2f", margin(logits.data(), forced[i]));
                    if (i + 1 == forced.size() || !lm.step(forced[i], pos, logits.data(), err))
                        break;
                    ++pos;
                }
                std::fprintf(stderr, "\n");
                continue;
            }
            auto is_eos = [&](int32_t t) { return std::find(eos.begin(), eos.end(), t) != eos.end(); };
            while ((int) gen.size() < max_new) {
                gen.push_back(next);
                if (is_eos(next) || (int) gen.size() >= max_new)
                    break;
                ++steps;
                if (mtp_k > 0) {
                    int32_t d[8];
                    const int k = std::min(mtp_k, max_new - (int) gen.size());
                    auto q0 = std::chrono::steady_clock::now();
                    if (!lm.draft(next, pos, k, d, err)) {
                        std::fprintf(stderr, "transcribe_run: %s\n", err.c_str());
                        return 1;
                    }
                    std::vector<int32_t> win = {next};
                    win.insert(win.end(), d, d + k);
                    t_draft += ms_since(q0);
                    q0 = std::chrono::steady_clock::now();
                    if (!lm.verify(win.data(), (int) win.size(), pos, logits.data(), err)) {
                        std::fprintf(stderr, "transcribe_run: %s\n", err.c_str());
                        return 1;
                    }
                    t_cpu += lm.last_cpu_ms();
                    if (!greedy_logits.empty()) {
                        const size_t step0 = gen.size(); // the greedy step whose logits row 0 of the window gives
                        bool prefix_ok = true;
                        for (int r = 0; r <= k && step0 + (size_t) r < greedy_logits.size(); ++r) {
                            prefix_ok = prefix_ok && (r == 0 || greedy_ids[step0 + (size_t) r - 1] == win[(size_t) r]);
                            if (!prefix_ok)
                                break;
                            float d = 0;
                            for (int64_t i = 0; i < V; ++i)
                                d = std::max(d, std::fabs(logits[(size_t) r * V + (size_t) i] -
                                                          greedy_logits[step0 + (size_t) r][(size_t) i]));
                            if (d > 0)
                                std::fprintf(stderr, "check: step %zu (window row %d of %d): max |d logit| %g\n",
                                             step0 + (size_t) r, r, k + 1, d);
                        }
                    }
                    // keep the drafts the model agrees with, stopping at an end-of-turn token
                    int keep = 1;
                    int32_t nt = argmax(logits.data(), V);
                    while (keep <= k && nt == d[keep - 1] && !is_eos(nt) && (int) gen.size() < max_new) {
                        gen.push_back(nt);
                        nt = argmax(logits.data() + (size_t) keep * V, V);
                        ++keep;
                    }
                    drafted += k;
                    accepted += keep - 1;
                    t_verify += ms_since(q0);
                    q0 = std::chrono::steady_clock::now();
                    if (!lm.accept(keep, err)) {
                        std::fprintf(stderr, "transcribe_run: %s\n", err.c_str());
                        return 1;
                    }
                    t_accept += ms_since(q0);
                    pos += keep;
                    next = nt;
                } else {
                    if (!lm.step(next, pos, logits.data(), err)) {
                        std::fprintf(stderr, "transcribe_run: %s\n", err.c_str());
                        return 1;
                    }
                    t_cpu += lm.last_cpu_ms();
                    ++pos;
                    next = argmax(logits.data(), V);
                }
            }
            const double decode_ms = ms_since(d0);
            const std::string text = vocab.decode(gen);
            const double total_ms = ms_since(c0);
            char tail[512];
            std::snprintf(
                tail, sizeof tail,
                ",\"seconds\":%.3f,\"prompt_tokens\":%d,\"audio_tokens\":%d,\"tokens\":%d,\"features_ms\":%.1f,"
                "\"encode_ms\":%.1f,\"prefill_ms\":%.1f,\"decode_ms\":%.1f,\"total_ms\":%.1f,\"steps\":%d,"
                "\"drafted\":%d,\"accepted\":%d,\"mtp\":%d,\"draft_ms\":%.1f,\"verify_ms\":%.1f,\"accept_ms\":%.1f,"
                "\"step_cpu_ms\":%.1f,"
                "\"run\":%d}",
                seconds, (int) ids.size(), n_audio, (int) gen.size(), fe_ms, enc_ms, prefill_ms, decode_ms, total_ms,
                steps, drafted, accepted, mtp_k, t_draft, t_verify, t_accept, t_cpu, r);
            std::string line = "{\"id\":" + json_escape(c.id) + ",\"ids\":[";
            for (size_t i = 0; i < gen.size(); ++i)
                line += (i ? "," : "") + std::to_string(gen[i]);
            line += "],\"text\":" + json_escape(text) + tail;
            if (out) {
                std::fprintf(out, "%s\n", line.c_str());
                std::fflush(out);
            }
            std::fprintf(stderr, "%s: %.1f s, %d tokens in %.0f ms (prefill %.0f, decode %.0f); %s\n", c.id.c_str(),
                         seconds, (int) gen.size(), total_ms, prefill_ms, decode_ms, text.c_str());
        }
    if (out)
        std::fclose(out);
    lm.print_timing("run");
    return 0;
}
