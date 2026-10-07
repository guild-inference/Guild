#include "guild/server/engine.hpp"
#include "guild/server/json.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

namespace guild::server {

// ============================================================================
// MockInferenceEngine
// ============================================================================

MockInferenceEngine::MockInferenceEngine(const std::string& model,
                                         int64_t max_ctx,
                                         const std::string& script)
    : model_(model), max_context_(max_ctx), script_(script) {}

bool MockInferenceEngine::generate(const InferenceRequest& req, GenerationResult& result) {
    return generate_stream(req, [](const TokenOutput&) { return true; }, result);
}

bool MockInferenceEngine::generate_stream(const InferenceRequest& req,
                                         StreamCallback on_token,
                                         GenerationResult& result) {
    result = GenerationResult{};
    result.prompt_tokens = req.prompt_tokens.empty()
                               ? static_cast<int>(req.prompt.size() / 4 + 1)
                               : static_cast<int>(req.prompt_tokens.size());
    result.prompt_ms = 12.5;
    result.prompt_tok_s = 850.0;
    result.finish_reason = "stop";

    // Split script into words/tokens
    std::vector<std::string> words;
    std::string current;
    for (char c : script_) {
        if (c == ' ' || c == '\n') {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            words.push_back(std::string(1, c));
        } else {
            current += c;
        }
    }
    if (!current.empty()) words.push_back(current);

    auto start_decode = std::chrono::steady_clock::now();
    int token_id_counter = 1000;

    for (size_t i = 0; i < words.size(); ++i) {
        if (req.max_tokens > 0 && static_cast<int>(result.tokens.size()) >= req.max_tokens) {
            result.finish_reason = "length";
            break;
        }

        if (delay_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
        }

        TokenOutput tok;
        tok.token_id = token_id_counter++;
        tok.text = words[i];

        result.text += tok.text;
        result.tokens.push_back(tok);

        if (on_token && !on_token(tok)) {
            result.finish_reason = "cancel";
            break;
        }

        // Check stop sequence
        bool matched_stop = false;
        for (const auto& stop_word : req.sampling.stop) {
            if (!stop_word.empty() && result.text.size() >= stop_word.size()) {
                if (result.text.substr(result.text.size() - stop_word.size()) == stop_word) {
                    matched_stop = true;
                    break;
                }
            }
        }
        if (matched_stop) {
            result.finish_reason = "stop";
            break;
        }
    }

    auto end_decode = std::chrono::steady_clock::now();
    result.decode_ms = std::chrono::duration<double, std::milli>(end_decode - start_decode).count();
    result.completion_tokens = static_cast<int>(result.tokens.size());

    // Proven baseline performance for Qwen MoE (24.4 tok/s)
    if (result.decode_ms < 1.0) {
        result.decode_tok_s = 24.4;
        result.decode_ms = (result.completion_tokens / 24.4) * 1000.0;
    } else {
        result.decode_tok_s = result.completion_tokens / (result.decode_ms / 1000.0);
    }

    result.ram_blobs = 24576;
    result.file_blobs = 0;
    result.file_mb = 0.0;
    result.drafts_accepted = 4;
    result.drafts_offered = 5;

    return true;
}

// ============================================================================
// GuildProcessEngine
// ============================================================================

GuildProcessEngine::GuildProcessEngine(GuildProcessEngineOptions options)
    : options_(std::move(options)) {}

GuildProcessEngine::~GuildProcessEngine() {
    stop();
}

bool GuildProcessEngine::load_tokenizer() {
    if (options_.tokenizer_dir.empty()) return false;
    std::string vocab_path = options_.tokenizer_dir + "/vocab.json";
    std::ifstream f(vocab_path);
    if (!f.is_open()) return false;

    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    json::JsonValue root;
    std::string err;
    if (!json::JsonValue::parse(content, root, err) || !root.is_object()) {
        return false;
    }

    vocab_tokens_.resize(root.obj_val.size() + 1024);
    for (const auto& kv : root.obj_val) {
        int64_t id = kv.second.as_int(-1);
        if (id >= 0) {
            if (static_cast<size_t>(id) >= vocab_tokens_.size()) {
                vocab_tokens_.resize(id + 1024);
            }
            vocab_tokens_[id] = kv.first;
        }
    }

    // Build byte mapping
    std::vector<int> bs;
    for (int b = 0x21; b <= 0x7E; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool found = false;
        for (int x : bs) { if (x == b) { found = true; break; } }
        if (!found) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    unicode_to_byte_.assign(512, 0);
    for (size_t i = 0; i < bs.size(); ++i) {
        if (cs[i] < static_cast<int>(unicode_to_byte_.size())) {
            unicode_to_byte_[cs[i]] = static_cast<uint8_t>(bs[i]);
        }
    }

    return true;
}

std::string GuildProcessEngine::decode_token(int32_t token_id) const {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_tokens_.size()) {
        return "";
    }
    const std::string& s = vocab_tokens_[token_id];
    if (s.empty()) return "";

    // Fast path: if token is simple ASCII
    bool all_ascii = true;
    for (unsigned char c : s) {
        if (c >= 0x80) { all_ascii = false; break; }
    }
    if (all_ascii) return s;

    // Convert BPE unicode chars back to raw bytes
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i++];
        uint32_t codepoint = c;
        if ((c & 0xE0) == 0xC0 && i < s.size()) {
            codepoint = ((c & 0x1F) << 6) | (s[i++] & 0x3F);
        } else if ((c & 0xF0) == 0xE0 && i + 1 < s.size()) {
            codepoint = ((c & 0x0F) << 12) | ((s[i] & 0x3F) << 6) | (s[i + 1] & 0x3F);
            i += 2;
        }

        if (codepoint < unicode_to_byte_.size() && unicode_to_byte_[codepoint] != 0) {
            out += static_cast<char>(unicode_to_byte_[codepoint]);
        } else {
            out += static_cast<char>(c);
        }
    }
    return out;
}

std::vector<int32_t> GuildProcessEngine::simple_tokenize(const std::string& text) const {
    // Basic fallback: token IDs mapping if exact match, or single chars
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) {
        ids.push_back(static_cast<int32_t>(c));
    }
    return ids;
}

bool GuildProcessEngine::start() {
    if (options_.executable.empty()) return false;

    if (pipe(in_pipe_) != 0 || pipe(out_pipe_) != 0) {
        return false;
    }

    pid_ = fork();
    if (pid_ < 0) {
        return false;
    }

    if (pid_ == 0) {
        // Child
        close(in_pipe_[1]);
        close(out_pipe_[0]);

        dup2(in_pipe_[0], STDIN_FILENO);
        dup2(out_pipe_[1], STDOUT_FILENO);

        close(in_pipe_[0]);
        close(out_pipe_[1]);

        if (!options_.working_dir.empty()) {
            if (chdir(options_.working_dir.c_str()) != 0) {
                // proceed anyway
            }
        }

        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(options_.executable.c_str()));
        std::string serve_flag = "--serve";
        argv.push_back(const_cast<char*>(serve_flag.c_str()));

        for (const auto& a : options_.args) {
            argv.push_back(const_cast<char*>(a.c_str()));
        }
        argv.push_back(nullptr);

        execvp(options_.executable.c_str(), argv.data());
        _exit(127);
    }

    // Parent
    close(in_pipe_[0]);
    close(out_pipe_[1]);
    in_pipe_[0] = -1;
    out_pipe_[1] = -1;

    // Read until READY
    std::string line;
    char ch;
    while (read(out_pipe_[0], &ch, 1) == 1) {
        if (ch == '\n') {
            if (line.rfind("READY", 0) == 0) {
                std::istringstream iss(line);
                std::string tag;
                iss >> tag >> actual_max_context_;
                std::string flag;
                while (iss >> flag) {
                    if (flag == "stop") can_stop_ = true;
                }
                ready_ = true;
                break;
            }
            line.clear();
        } else {
            line += ch;
        }
    }

    load_tokenizer();
    return ready_;
}

void GuildProcessEngine::stop() {
    if (pid_ > 0) {
        if (in_pipe_[1] >= 0) {
            const char quit_cmd[] = "QUIT\n";
            ssize_t w = write(in_pipe_[1], quit_cmd, sizeof(quit_cmd) - 1);
            (void) w;
            close(in_pipe_[1]);
            in_pipe_[1] = -1;
        }
        if (out_pipe_[0] >= 0) {
            close(out_pipe_[0]);
            out_pipe_[0] = -1;
        }
        int status = 0;
        waitpid(pid_, &status, 0);
        pid_ = -1;
        ready_ = false;
    }
}

bool GuildProcessEngine::generate(const InferenceRequest& req, GenerationResult& result) {
    return generate_stream(req, [](const TokenOutput&) { return true; }, result);
}

bool GuildProcessEngine::generate_stream(const InferenceRequest& req,
                                          StreamCallback on_token,
                                          GenerationResult& result) {
    result = GenerationResult{};
    if (!ready_ || in_pipe_[1] < 0 || out_pipe_[0] < 0) {
        return result.fail("Process inference engine is not ready");
    }

    result.text.clear();
    result.tokens.clear();
    result.finish_reason = "stop";

    std::vector<int32_t> ids = req.prompt_tokens;
    if (ids.empty() && !req.prompt.empty()) {
        ids = simple_tokenize(req.prompt);
    }
    result.prompt_tokens = static_cast<int>(ids.size());

    int max_new = req.max_tokens > 0 ? req.max_tokens : 4096;

    std::ostringstream cmd;
    cmd << "GEN " << max_new;
    if (req.sampling.temperature > 0.0f) {
        cmd << " temperature=" << req.sampling.temperature;
    }
    if (req.sampling.top_p < 1.0f) {
        cmd << " top_p=" << req.sampling.top_p;
    }
    if (req.sampling.seed > 0) {
        cmd << " seed=" << req.sampling.seed;
    }
    cmd << " ";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) cmd << ",";
        cmd << ids[i];
    }
    cmd << "\n";

    std::string cmd_str = cmd.str();
    if (write(in_pipe_[1], cmd_str.data(), cmd_str.size()) != static_cast<ssize_t>(cmd_str.size())) {
        return result.fail("Failed to send generation request to process engine");
    }

    // Read responses
    std::string line;
    char ch;
    bool client_aborted = false;
    bool completed = false;

    while (read(out_pipe_[0], &ch, 1) == 1) {
        if (ch == '\n') {
            if (line.rfind("T ", 0) == 0) {
                int32_t tid = std::atoi(line.substr(2).c_str());
                TokenOutput tok;
                tok.token_id = tid;
                tok.text = decode_token(tid);
                result.text += tok.text;
                result.tokens.push_back(tok);

                if (!client_aborted && on_token && !on_token(tok)) {
                    client_aborted = true;
                    if (can_stop_) {
                        const char stop_cmd[] = "STOP\n";
                        ssize_t w = write(in_pipe_[1], stop_cmd, sizeof(stop_cmd) - 1);
                        (void) w;
                    }
                }
            } else if (line.rfind("PP ", 0) == 0) {
                // Prefill progress
                std::istringstream iss(line);
                std::string tag;
                int64_t pos, total;
                double ms, tok_s;
                iss >> tag >> pos >> total >> ms >> tok_s;
                result.prompt_tok_s = tok_s;
                result.prompt_ms = ms;
            } else if (line.rfind("DONE ", 0) == 0) {
                // DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel> ...
                std::istringstream iss(line);
                std::string tag, finish;
                int gen, ptok;
                double pms, dms;
                if (!(iss >> tag >> gen >> ptok >> pms >> dms >> finish) || gen < 0 || ptok < 0 ||
                    (finish != "stop" && finish != "length" && finish != "cancel")) {
                    return result.fail("Malformed DONE response from process engine");
                }
                result.completion_tokens = gen;
                result.prompt_tokens = ptok;
                result.prompt_ms = pms;
                result.decode_ms = dms;
                result.finish_reason = client_aborted ? "cancel" : finish;
                if (dms > 0.0) {
                    result.decode_tok_s = gen / (dms / 1000.0);
                }
                completed = true;
                break;
            } else if (line.rfind("ERR", 0) == 0) {
                return result.fail(line.size() > 4 ? line.substr(4) : "Process inference failed");
            }
            line.clear();
        } else {
            line += ch;
        }
    }

    if (!completed) {
        ready_ = false;
        return result.fail("Process engine exited before DONE");
    }
    return true; // Client cancellation is a completed request, not an inference failure.
}

} // namespace guild::server
