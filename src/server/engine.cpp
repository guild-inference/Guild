#include "guild/server/engine.hpp"
#include "guild/server/json.hpp"
#include "guild/runtime/tokenizer.hpp"

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

bool MockInferenceEngine::encode(const std::string& text, std::vector<int32_t>& tokens, std::string& err) {
    err.clear();
    tokens.clear();
    if (text.empty()) return true;
    size_t count = text.size() / 4 + 1;
    for (size_t i = 0; i < count; ++i) {
        tokens.push_back(static_cast<int32_t>(1000 + i));
    }
    return true;
}

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
    tokenizer_ = std::make_unique<runtime::Tokenizer>();
    std::string err;
    if (!tokenizer_->load(options_.tokenizer_dir, err)) {
        tokenizer_.reset();
        return false;
    }
    return true;
}

std::string GuildProcessEngine::decode_token(int32_t token_id) const {
    if (!tokenizer_) return "";
    return tokenizer_->decode(token_id, true);
}

bool GuildProcessEngine::encode(const std::string& text, std::vector<int32_t>& tokens, std::string& err) {
    if (!tokenizer_) {
        err = "Process engine has no loaded tokenizer";
        return false;
    }
    return tokenizer_->encode(text, tokens, true, err);
}

bool GuildProcessEngine::decode(const std::vector<int32_t>& tokens, std::string& text) const {
    if (!tokenizer_) return false;
    text = tokenizer_->decode(tokens, true);
    return true;
}

const runtime::ChatTemplate* GuildProcessEngine::chat_template() const {
    if (!tokenizer_) return nullptr;
    return &tokenizer_->chat_template();
}

const runtime::Tokenizer* GuildProcessEngine::tokenizer() const {
    return tokenizer_.get();
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
        std::string err;
        if (!encode(req.prompt, ids, err)) {
            return result.fail("Process engine tokenization failed: " + err);
        }
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
