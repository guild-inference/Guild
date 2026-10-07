#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace guild::server {

struct SamplingParams {
    float temperature = 0.0f; // 0 = greedy
    float top_p = 1.0f;
    int top_k = 0;
    int seed = 0;
    std::vector<std::string> stop;
};

struct InferenceRequest {
    std::string request_id;
    std::string model;
    std::string prompt;
    std::vector<int32_t> prompt_tokens;
    int max_tokens = 0;
    SamplingParams sampling;
};

struct TokenOutput {
    int32_t token_id = 0;
    std::string text;
    bool is_special = false;
};

struct GenerationResult {
    std::string text;
    std::vector<TokenOutput> tokens;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double prompt_ms = 0.0;
    double decode_ms = 0.0;
    double prompt_tok_s = 0.0;
    double decode_tok_s = 0.0;
    std::string finish_reason = "stop"; // "stop", "length", "cancel", "error"
    std::string error_message;
    int drafts_accepted = 0;
    int drafts_offered = 0;
    int reused_tokens = 0;
    int64_t ram_blobs = 0;
    int64_t file_blobs = 0;
    double file_mb = 0.0;
    int64_t gpu_cache_hits = 0;

    bool fail(const std::string& message) {
        text.clear();
        tokens.clear();
        completion_tokens = 0;
        finish_reason = "error";
        error_message = message.empty() ? "Inference failed" : message;
        return false;
    }
};

using StreamCallback = std::function<bool(const TokenOutput& token)>;

class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;
    virtual std::string model_name() const = 0;
    virtual int64_t max_context() const = 0;
    virtual bool is_ready() const = 0;

    virtual bool generate(const InferenceRequest& req, GenerationResult& result) = 0;
    virtual bool generate_stream(const InferenceRequest& req,
                                 StreamCallback on_token,
                                 GenerationResult& result) = 0;
    virtual void stop() = 0;
};

class MockInferenceEngine : public IInferenceEngine {
public:
    MockInferenceEngine(const std::string& model = "Qwen3.8-Flash-Next",
                        int64_t max_ctx = 262144,
                        const std::string& script = "Hello! I am Guild running natively.");

    std::string model_name() const override { return model_; }
    int64_t max_context() const override { return max_context_; }
    bool is_ready() const override { return true; }

    bool generate(const InferenceRequest& req, GenerationResult& result) override;
    bool generate_stream(const InferenceRequest& req,
                         StreamCallback on_token,
                         GenerationResult& result) override;
    void stop() override {}

    void set_script(const std::string& s) { script_ = s; }
    void set_delay_ms(int ms) { delay_ms_ = ms; }

private:
    std::string model_;
    int64_t max_context_;
    std::string script_;
    int delay_ms_ = 0;
};

struct GuildProcessEngineOptions {
    std::string executable;
    std::vector<std::string> args;
    std::string working_dir;
    std::string tokenizer_dir;
    std::string model_name = "Qwen3.8-Flash-Next";
    int64_t max_context = 262144;
};

class GuildProcessEngine : public IInferenceEngine {
public:
    explicit GuildProcessEngine(GuildProcessEngineOptions options);
    ~GuildProcessEngine() override;

    bool start();
    std::string model_name() const override { return options_.model_name; }
    int64_t max_context() const override { return actual_max_context_ > 0 ? actual_max_context_ : options_.max_context; }
    bool is_ready() const override { return ready_; }

    bool generate(const InferenceRequest& req, GenerationResult& result) override;
    bool generate_stream(const InferenceRequest& req,
                         StreamCallback on_token,
                         GenerationResult& result) override;
    void stop() override;

private:
    GuildProcessEngineOptions options_;
    int in_pipe_[2]{-1, -1};
    int out_pipe_[2]{-1, -1};
    int pid_ = -1;
    bool ready_ = false;
    int64_t actual_max_context_ = 0;
    bool can_stop_ = false;

    // Tokenizer vocabulary for decoding tokens back to UTF-8
    std::vector<std::string> vocab_tokens_;
    std::vector<uint8_t> unicode_to_byte_;

    bool load_tokenizer();
    std::string decode_token(int32_t token_id) const;
    std::vector<int32_t> simple_tokenize(const std::string& text) const;
};

} // namespace guild::server
