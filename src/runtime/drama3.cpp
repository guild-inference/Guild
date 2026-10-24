// src/runtime/drama3.cpp - Fish Audio Drama 3 text to speech (GuildModel::synthesize).
//
// The prompt is fish_speech/models/text2semantic/fish_omni/generate.py's `build_drama3_prompt`
// (speech_template_version reference-bracket-v1): rows of [token, c0..c9], text rows with zero codes, a reference
// clip's frames as [<|semantic:c0|>, c0..c9].  Every literal is tokenized on its own, as the reference does
// (tokenizing the joined string changes ids at the seams):
//
//   <|im_start|>system\n  "convert the provided text to speech[ reference to the following:]"
//   per reference i:      "\n\n[Reference i]\nText: <|speaker:N|>transcript\nSpeech: "  + its frames
//   <|im_end|>\n  <|im_start|>user\n  "<|speaker:0|> text"  <|im_end|>\n  <|im_start|>assistant\n<|voice|>
//
// Decode is the reference's generate_drama3: the backbone's state at row p predicts frame p + 1 through the tower;
// over the prompt the tower is teacher-forced with the next row, after it each sampled frame is the next input.
#include "model_impl.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace guild::runtime {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

constexpr int kWidth = 11;           // [main, c0..c9]
constexpr int kMaxPromptRows = 8192; // the reference's DRAMA3_MAX_PROMPT_ROWS
constexpr int kMaxSpeaker = 15;

// leading white space (ASCII, NBSP, ideographic space, U+2000..U+200A)
size_t white_prefix(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        if (c == ' ' || (c >= 0x09 && c <= 0x0D)) {
            ++i;
        } else if (c == 0xC2 && i + 1 < s.size() &&
                   ((unsigned char) s[i + 1] == 0xA0 || (unsigned char) s[i + 1] == 0x85)) {
            i += 2;
        } else if (c == 0xE3 && i + 2 < s.size() && (unsigned char) s[i + 1] == 0x80 &&
                   (unsigned char) s[i + 2] == 0x80) {
            i += 3;
        } else if (c == 0xE2 && i + 2 < s.size() && (unsigned char) s[i + 1] == 0x80 &&
                   (unsigned char) s[i + 2] >= 0x80 && (unsigned char) s[i + 2] <= 0x8A) {
            i += 3;
        } else {
            break;
        }
    }
    return i;
}

// length of a canonical <|speaker:N|> (N 0..15) at `at`, else 0
size_t speaker_tag(const std::string& s, size_t at) {
    static const std::string head = "<|speaker:";
    if (s.compare(at, head.size(), head) != 0)
        return 0;
    size_t i = at + head.size(), j = i;
    while (j < s.size() && s[j] >= '0' && s[j] <= '9')
        ++j;
    if (j == i || s.compare(j, 2, "|>") != 0)
        return 0;
    const std::string digits = s.substr(i, j - i);
    if (digits.size() > 2 || (digits.size() > 1 && digits[0] == '0') || std::stoi(digits) > kMaxSpeaker)
        return 0;
    return j + 2 - at;
}

bool validate_text(const std::string& text, const std::string& where, std::string& err) {
    if (white_prefix(text) == text.size()) {
        err = where + ": text must not be empty";
        return false;
    }
    for (const char* t : {"<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"}) {
        if (text.find(t) != std::string::npos) {
            err = where + ": " + t + " is written by the prompt builder";
            return false;
        }
    }
    for (size_t at = text.find("<|speaker"); at != std::string::npos; at = text.find("<|speaker", at + 1)) {
        if (speaker_tag(text, at) == 0) {
            err = where + ": speaker tags must be <|speaker:0|> to <|speaker:15|>";
            return false;
        }
    }
    return true;
}

std::string with_speaker_tag(const std::string& text) {
    return speaker_tag(text, white_prefix(text)) ? text : "<|speaker:0|> " + text;
}

} // namespace

bool GuildModel::is_speech_model() const {
    return impl_ && impl_->tower != nullptr;
}

bool GuildModel::synthesize(const SpeechRequest& req, const SpeechCallback& on_audio, SpeechTelemetry& telemetry,
                            std::string& error_msg) {
    Impl& m = *impl_;
    if (!m.tower || !m.codec || !m.ver) {
        error_msg = "this model was not loaded as a speech model (--drama3)";
        return false;
    }
    const core::Drama3TowerGeometry& tg = m.tower->geometry();
    const int CS = tg.codebook_size;
    if (req.top_k < 1) {
        error_msg = "top_k must be at least 1";
        return false;
    }

    // ---- the prompt rows
    std::vector<int32_t> rows; // [n][kWidth]
    auto text_rows = [&](const std::string& s) -> bool {
        std::vector<int32_t> ids;
        if (!m.tokenizer->encode(s, ids, true, error_msg))
            return false;
        for (int32_t id : ids) {
            rows.push_back(id);
            rows.insert(rows.end(), kWidth - 1, 0);
        }
        return true;
    };
    if (!validate_text(req.text, "text", error_msg))
        return false;
    if (!text_rows("<|im_start|>system\n") ||
        !text_rows(req.references.empty() ? "convert the provided text to speech"
                                          : "convert the provided text to speech reference to the following:"))
        return false;
    for (size_t i = 0; i < req.references.size(); ++i) {
        const SpeechReference& r = req.references[i];
        const std::string where = "reference " + std::to_string(i + 1);
        if (!validate_text(r.text, where, error_msg))
            return false;
        const int nc = tg.n_codebooks;
        if (r.codes.empty() || r.codes.size() % (size_t) nc) {
            error_msg = where + ": codes must be [frames >= 1][" + std::to_string(nc) + "]";
            return false;
        }
        // the tag must follow "Text: " directly; leading white space moves behind it
        const std::string tagged = with_speaker_tag(r.text);
        const size_t lead = white_prefix(tagged), tag = speaker_tag(tagged, lead);
        const std::string transcript = tagged.substr(lead, tag) + tagged.substr(0, lead) + tagged.substr(lead + tag);
        if (!text_rows("\n\n[Reference " + std::to_string(i + 1) + "]\nText: " + transcript + "\nSpeech: "))
            return false;
        for (size_t f = 0; f < r.codes.size(); f += (size_t) nc) {
            for (int k = 0; k < nc; ++k) {
                if (r.codes[f + k] < 0 || r.codes[f + k] >= CS) {
                    error_msg = where + ": codes must lie in [0, " + std::to_string(CS) + ")";
                    return false;
                }
            }
            rows.push_back(tg.semantic_start + r.codes[f]);
            rows.insert(rows.end(), r.codes.begin() + (std::ptrdiff_t) f, r.codes.begin() + (std::ptrdiff_t)(f + nc));
        }
    }
    if (!text_rows("<|im_end|>\n") || !text_rows("<|im_start|>user\n") || !text_rows(with_speaker_tag(req.text)) ||
        !text_rows("<|im_end|>\n") || !text_rows("<|im_start|>assistant\n<|voice|>"))
        return false;
    const int R = (int) (rows.size() / kWidth);
    telemetry.prompt_rows = R;
    if (const char* dump = std::getenv("GUILD_DRAMA3_DUMP_PROMPT"); dump && *dump) { // int32 [rows][11], for tests
        if (std::FILE* f = std::fopen(dump, "wb")) {
            std::fwrite(rows.data(), 4, rows.size(), f);
            std::fclose(f);
        }
    }
    if (R > kMaxPromptRows) {
        error_msg = "the prompt has " + std::to_string(R) + " rows; at most " + std::to_string(kMaxPromptRows);
        return false;
    }
    if ((int64_t) R + req.max_new_frames > m.tower->max_frames() ||
        (int64_t) R + req.max_new_frames > m.ver->attention_capacity()) {
        error_msg = "the prompt (" + std::to_string(R) + " rows) and max_new_frames do not fit the context (" +
                    std::to_string(std::min<int64_t>(m.tower->max_frames(), m.ver->attention_capacity())) + ")";
        return false;
    }

    // ---- a fresh sequence
    m.clear_prefix_cache();
    if (!m.ver->wait_commit(error_msg))
        return false;
    core::session_zero(m.ss, m.g, nullptr, m.main_stream);
    if (cudaStreamSynchronize(m.main_stream) != cudaSuccess) {
        error_msg = "failed to reset inference state";
        return false;
    }
    m.codec->reset();
    m.tower->reset_sampling();
    core::Drama3Sampling sp;
    sp.temperature = req.temperature;
    sp.top_p = req.top_p;
    sp.top_k = req.top_k;
    sp.seed = req.seed;

    std::vector<int32_t> mains; // every row's main token so far (the repetition window)
    mains.reserve((size_t) R + (size_t) req.max_new_frames);
    for (int p = 0; p < R; ++p)
        mains.push_back(rows[(size_t) p * kWidth]);
    std::vector<int32_t> frame(kWidth), pending; // pending: generated frames not yet decoded, [n][10]
    std::vector<float> pcm;
    double codec_ms = 0.0, tower_ms = 0.0;
    auto cancelled = [&] { return req.cancel_flag && req.cancel_flag->load(); };
    auto flush = [&](bool last) -> bool {
        const int n = (int) (pending.size() / (size_t) tg.n_codebooks);
        if (n == 0 && !last)
            return true;
        const auto t0 = Clock::now();
        pcm.resize((size_t) n * (size_t) m.codec->frame_samples());
        if (n > 0 && !m.codec->decode(pending.data(), n, pcm.data(), error_msg))
            return false;
        codec_ms += ms_since(t0);
        pending.clear();
        SpeechChunk c;
        c.pcm = pcm.data();
        c.samples = pcm.size();
        c.sample_rate = m.codec->sample_rate();
        c.last = last;
        return on_audio(c);
    };
    auto tower_step = [&](const float* hidden, int64_t p, const int32_t* row, const int32_t* forced) -> bool {
        const auto t0 = Clock::now();
        const int n_hist = (int) std::min<size_t>(mains.size(), (size_t) p + 1);
        const bool ok = m.tower->step(hidden, p, row, forced, mains.data() + (n_hist - std::min(n_hist, 16)),
                                      std::min(n_hist, 16), sp, frame.data(), error_msg);
        tower_ms += ms_since(t0);
        return ok;
    };
    auto backbone = [&](const int32_t* tokens, const int32_t* code1, int T, int64_t p0) -> bool {
        std::vector<int32_t> out((size_t) T);
        m.ver->set_codes(code1, T);
        m.drive.d.layers = 0;
        m.drive.d.experts = 0;
        return m.ver->run(T, tokens, p0, drive_pool_multi, &m.drive, out.data(), error_msg) &&
               m.ver->commit(T, error_msg);
    };

    // Test hooks: GUILD_DRAMA3_FORCE (int32 [n][11]) teacher-forces the decode with those frames;
    // GUILD_DRAMA3_DUMP_HIDDEN writes the backbone's final-norm state of every position (float32 [rows][n_embd]).
    std::vector<int32_t> forced_frames;
    if (const char* f = std::getenv("GUILD_DRAMA3_FORCE"); f && *f) {
        if (std::FILE* in = std::fopen(f, "rb")) {
            int32_t v;
            while (std::fread(&v, 4, 1, in) == 1)
                forced_frames.push_back(v);
            std::fclose(in);
        }
    }
    std::FILE* hidden_dump = nullptr;
    if (const char* f = std::getenv("GUILD_DRAMA3_DUMP_HIDDEN"); f && *f)
        hidden_dump = std::fopen(f, "wb");
    std::vector<float> hidden_host((size_t) m.g.n_embd);
    auto dump_hidden = [&](const float* dev) {
        if (hidden_dump &&
            cudaMemcpy(hidden_host.data(), dev, hidden_host.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess)
            std::fwrite(hidden_host.data(), 4, hidden_host.size(), hidden_dump);
    };
    auto forced_at = [&](int64_t frame_index) -> const int32_t* {
        return (frame_index + 1) * kWidth <= (int64_t) forced_frames.size()
                   ? &forced_frames[(size_t) frame_index * kWidth]
                   : nullptr;
    };

    // ---- the prompt: windows of up to `spec` rows; the tower learns each row's next row
    const auto t_prompt = Clock::now();
    const int S = std::max(1, m.config.spec);
    std::vector<int32_t> tok((size_t) S), c1((size_t) S);
    for (int p0 = 0; p0 < R; p0 += S) {
        if (cancelled()) {
            telemetry.finish_reason = "cancel";
            return true;
        }
        const int T = std::min(S, R - p0);
        for (int t = 0; t < T; ++t) {
            tok[(size_t) t] = rows[(size_t) (p0 + t) * kWidth];
            c1[(size_t) t] = rows[(size_t) (p0 + t) * kWidth + 2];
        }
        if (!backbone(tok.data(), c1.data(), T, p0))
            return false;
        for (int t = 0; t < T; ++t) {
            const int p = p0 + t;
            const int32_t* forced = p + 1 < R ? &rows[(size_t) (p + 1) * kWidth] : forced_at(0);
            dump_hidden(m.ver->head_input(t));
            if (!tower_step(m.ver->head_input(t), p, &rows[(size_t) p * kWidth], forced))
                return false;
        }
    }
    telemetry.prompt_ms = ms_since(t_prompt);

    // ---- decode: `frame` holds the frame after row R - 1
    const auto t_decode = Clock::now();
    telemetry.finish_reason = "length";
    for (int step = 0; step < req.max_new_frames; ++step) {
        if (frame[0] == tg.eos) {
            telemetry.finish_reason = "eos";
            break;
        }
        if (cancelled()) {
            telemetry.finish_reason = "cancel";
            break;
        }
        ++telemetry.frames;
        mains.push_back(frame[0]);
        pending.insert(pending.end(), frame.begin() + 1, frame.end());
        if ((int) (pending.size() / (size_t) tg.n_codebooks) >= std::max(1, req.chunk_frames) && !flush(false)) {
            if (!error_msg.empty())
                return false;
            telemetry.finish_reason = "cancel";
            break;
        }
        if (step + 1 == req.max_new_frames)
            break;
        const int64_t p = (int64_t) R + step; // the new frame's row
        const std::vector<int32_t> row = frame;
        if (!backbone(&row[0], &row[2], 1, p))
            return false;
        dump_hidden(m.ver->head_input(0));
        if (!tower_step(m.ver->head_input(0), p, row.data(), forced_at(step + 1)))
            return false;
    }
    telemetry.decode_ms = ms_since(t_decode);
    if (hidden_dump)
        std::fclose(hidden_dump);
    if (!flush(true) && !error_msg.empty())
        return false;
    telemetry.tower_ms = tower_ms;
    telemetry.codec_ms = codec_ms;
    return true;
}

} // namespace guild::runtime
