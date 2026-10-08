#include "guild/server/engine.hpp"
#include "guild/server/native_engine.hpp"
#include "guild/server/telemetry.hpp"
#include "guild/runtime/types.hpp"
#include "guild/runtime/tokenizer.hpp"
#include "guild/model/model_descriptor.hpp"
#include "guild/memory/planner.hpp"
#include "guild/cli/hardware.hpp"

#include "../check.hpp"
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "[parity_test] Running in-process inference engine and parity checks...\n";

    // 1. Test MockInferenceEngine stream & non-stream consistency
    {
        std::cout << "  Test 1: MockInferenceEngine request consistency\n";
        guild::server::MockInferenceEngine mock("Qwen3.8-Flash-Next", 262144, "The quick brown fox jumps over the lazy dog");

        guild::server::InferenceRequest req;
        req.request_id = "test-1";
        req.model = "Qwen3.8-Flash-Next";
        req.prompt = "Tell me about the fox";
        req.max_tokens = 50;

        guild::server::GenerationResult res1;
        CHECK(mock.generate(req, res1));
        CHECK(!res1.text.empty());
        CHECK(res1.completion_tokens > 0);
        CHECK(res1.finish_reason == "stop");

        guild::server::GenerationResult res2;
        std::string streamed_text;
        auto stream_cb = [&](const guild::server::TokenOutput& tok) -> bool {
            streamed_text += tok.text;
            return true;
        };
        bool s_ok = mock.generate_stream(req, stream_cb, res2);
        CHECK(s_ok);
        (void) s_ok;
        CHECK(streamed_text == res1.text);
        CHECK(res2.completion_tokens == res1.completion_tokens);
        std::cout << "  -> PASSED\n";
    }

    // 2. Test NativeInferenceEngine stub / initialization contract
    {
        std::cout << "  Test 2: NativeInferenceEngine lifecycle & interface contract\n";
        guild::server::NativeInferenceEngineOptions opts;
        opts.model_name = "Qwen3.8-Flash-Next";
        opts.desc.name = "Qwen3.8-Flash-Next";
        opts.desc.attn.context_length = 262144;
        opts.plan.context_length = 262144;

        guild::server::NativeInferenceEngine native(opts);
        CHECK(native.model_name() == "Qwen3.8-Flash-Next");
        CHECK(native.max_context() == 262144);

        // Before init, is_ready() must be false
        CHECK(!native.is_ready());

        std::string err;
        bool inited = native.init(err);
        // On CPU-only builds or without real weights files, init returns false cleanly
        CHECK(!inited);
        CHECK(!err.empty());
        CHECK(!native.is_ready());
        guild::server::InferenceRequest req;
        req.prompt = "Test";
        guild::server::GenerationResult res;
        res.text = "stale successful result";
        CHECK(!native.generate(req, res));
        CHECK(res.finish_reason == "error" && !res.error_message.empty());
        CHECK(res.text.empty() && res.completion_tokens == 0);
        std::cout << "  -> PASSED\n";
    }

    // 3. Missing tokenizer assets must not synthesize raw byte token IDs.
    {
        std::cout << "  Test 3: Unloaded tokenizer rejects encoding\n";
        guild::runtime::Tokenizer tok;
        std::string sample = "Hello, world!";
        std::vector<int32_t> ids = tok.tokenize(sample);
        CHECK(ids.empty());
        std::string err;
        CHECK(!tok.load("/nonexistent-guild-tokenizer", err));
        CHECK(!err.empty() && !tok.is_loaded());
        std::cout << "  -> PASSED\n";
    }

    // 4. Test Telemetry direct counter recording
    {
        std::cout << "  Test 4: Direct Telemetry counter accuracy\n";
        guild::server::Telemetry telem;

        guild::server::RequestMetrics m1;
        m1.method = "POST";
        m1.path = "/v1/chat/completions";
        m1.status_code = 200;
        m1.duration_s = 0.5;
        m1.prompt_tokens = 32;
        m1.completion_tokens = 12;
        m1.prompt_tok_s = 640.0;
        m1.decode_tok_s = 24.4;
        m1.ram_expert_hits = 120;
        m1.file_expert_reads = 0;
        m1.gpu_cache_hits = 10;
        m1.drafts_accepted = 9;
        m1.drafts_offered = 12;
        m1.context_tokens = 44;
        m1.streamed = true;
        m1.outcome = "ok";

        telem.record_request_start();
        telem.record_request_finish(m1);

        auto snap = telem.snapshot();
        CHECK(snap.active_requests == 0);
        CHECK(snap.completed_requests == 1);
        CHECK(snap.prompt_tokens_processed == 32);
        CHECK(snap.generated_tokens_produced == 12);
        CHECK(snap.ram_expert_hits == 120);
        CHECK(snap.file_expert_reads == 0);
        CHECK(snap.gpu_cache_hits == 10);
        CHECK(snap.drafts_accepted == 9);
        CHECK(snap.drafts_offered == 12);
        CHECK(snap.context_usage == 44);
        (void) snap;

        std::string jsonl = guild::server::Telemetry::format_jsonl(m1);
        CHECK(jsonl.find("\"ram_expert_hits\":120") != std::string::npos);
        CHECK(jsonl.find("\"drafts_accepted\":9") != std::string::npos);
        CHECK(jsonl.find("\"decode_tok_s\":24.4") != std::string::npos);
        std::cout << "  -> PASSED\n";
    }

    // 5. Test Multiple Concurrent Request Isolation
    {
        std::cout << "  Test 5: Multi-request isolation and reset\n";
        guild::server::MockInferenceEngine mock("Qwen3.8-Flash-Next", 262144, "Response 1");
        guild::server::InferenceRequest req1{"req-1", "Qwen3.8-Flash-Next", "P1", {}, 10, {}};
        guild::server::GenerationResult res1;
        CHECK(mock.generate(req1, res1));

        mock.set_script("Response 2");
        guild::server::InferenceRequest req2{"req-2", "Qwen3.8-Flash-Next", "P2", {}, 10, {}};
        guild::server::GenerationResult res2;
        CHECK(mock.generate(req2, res2));

        CHECK(res1.text != res2.text);
        CHECK(res2.text.find("Response 2") != std::string::npos);
        std::cout << "  -> PASSED\n";
    }

    std::cout << "\n[parity_test] ALL TESTS PASSED!\n";
    return 0;
}
