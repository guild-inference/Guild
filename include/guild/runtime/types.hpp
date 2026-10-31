#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace guild::runtime {

struct ModelPaths {
    std::string primary_model_path;
    std::vector<std::string> additional_shards;
    std::string pack_dir;
    std::string expert_profile_path;
    std::string mtp_dir;
    std::string tokenizer_dir;
    std::string ple_gguf;
    /// How the PLE table is read: "" or "direct" (unbuffered reads and a row cache), "mmap", or "ram" (mapped and
    /// locked in RAM at open: no disk read on the token path; for a model on a rotational disk).
    std::string ple_io;
    std::string gexec_path;
    /// Fish Audio Drama 3: the directory with `tower/` (tools/drama3_tower_rt.py) and `codec/`
    /// (tools/drama3_codec_rt.py).  Set, the model is a text-to-speech model: GuildModel::synthesize.
    std::string drama3_dir;
};

struct RuntimeConfig {
    int64_t max_context = 262144;
    std::string kv_format = "fp16";
    bool kv_host_only = true;
    int64_t kv_resident = 0;
    bool shared_early = true;
    uint64_t resident_budget_gib = 56;
    int spec = 4;
    float spec_min_p = 0.5f;
    std::string decode_strategy = "auto";
    int spec_max_draft = 0;
    int spec_lookup_window = 65536;
    int spec_cost_window = 16;
    bool spec_trace = false;
    int expert_cache_slots = 0; // -1 for auto, 0 for off, >0 explicit
    int64_t prefill_chunk = 512;
    int pool_workers = 0;
    int vram_reserve_mib = 512;
    double pcie_expert_fraction = 0.0;
    std::vector<int> pcie_layer_numerators;
    std::vector<int64_t> eos_token_ids = {151643, 151645, 151644}; // Qwen default EOS/im_end
};

struct GenerationRequest {
    std::string request_id;
    std::string prompt;
    std::vector<int32_t> prompt_tokens;
    int max_new_tokens = 512;
    float temperature = 0.0f; // 0.0 = greedy
    float top_p = 1.0f;
    float min_p = 0.0f;
    int top_k = 0;
    int penalty_last_n = 0;
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    uint64_t seed = 0;
    std::string decode_strategy; // Optional per-request override ("normal", "mtp", "lookup", "retrieval", "adaptive")
    int spec_max_draft = -1;     // Optional per-request override (-1 = use model config)
    std::vector<std::string> stop;
    std::shared_ptr<std::atomic<bool>> cancel_flag;
};

/// Scoring a prompt without generating (GuildSession::score): the next-token logits at its last position.
struct ScoreRequest {
    std::vector<int32_t> tokens;     ///< the prompt
    std::vector<int32_t> candidates; ///< the token ids whose logits come back (empty: the whole vocabulary)
    /// > 0: the state after this many tokens is also kept as a checkpoint, so later prompts that begin with them
    /// read only the rest.  At most tokens.size() - 1 (the last token is read by the window that gives the logits).
    int64_t keep_prefix = 0;
    /// Also return the last position's final hidden state: the output head's input, after the output norm.
    bool want_hidden = false;
};

struct ScoreResult {
    std::vector<float> logits; ///< one per candidate, in their order (or the vocabulary's)
    std::vector<float> hidden; ///< with want_hidden: n_embd floats
    int64_t cached_tokens = 0; ///< tokens taken from the live state or a checkpoint instead of being read
    int64_t read_tokens = 0;   ///< tokens read by this call (prefill and the last token's window)
    double prefill_ms = 0;     ///< reading the prompt up to its last token (and the checkpoint, if any)
    double window_ms = 0;      ///< the last token's window and the logits' copy
};

/// A voice to clone: its transcript, and its codes ([frames][10], GuildModel::encode_voice) or its audio (mono
/// samples in [-1, 1] at `sample_rate`; synthesize encodes it, resampled to 44.1 kHz as torchaudio does)
struct SpeechReference {
    std::string text;
    std::vector<int32_t> codes;
    std::vector<float> audio;
    int sample_rate = 44100;
};

/// One text-to-speech request (Drama 3).  `text` may carry <|speaker:N|> tags (0..15) and the model's inline tags
/// ([laughs], [whisper] ...); without a leading speaker tag it is speaker 0.
struct SpeechRequest {
    std::string text;
    std::vector<SpeechReference> references;
    int max_new_frames = 2048; ///< 21.5 frames per second of audio
    float temperature = 1.0f;
    float top_p = 0.9f;
    int top_k = 30;
    uint64_t seed = 0;
    int chunk_frames = 16; ///< audio is handed out every this many frames
    std::shared_ptr<std::atomic<bool>> cancel_flag;
};

struct SpeechChunk {
    const float* pcm = nullptr; ///< mono, [-1, 1]
    size_t samples = 0;
    int sample_rate = 44100;
    bool last = false;
};

/// Return false to stop the generation.
using SpeechCallback = std::function<bool(const SpeechChunk& chunk)>;

struct SpeechTelemetry {
    int prompt_rows = 0;
    int frames = 0;
    double prompt_ms = 0.0, decode_ms = 0.0, tower_ms = 0.0, codec_ms = 0.0;
    std::string finish_reason; ///< "eos", "length", "cancel"
};

struct TokenEvent {
    int32_t token_id = 0;
    std::string text;
    bool is_special = false;
};

using TokenStreamCallback = std::function<bool(const TokenEvent& token)>;

struct PrefillProgressEvent {
    int64_t pos = 0;
    int64_t total = 0;
    double ms = 0.0;
    double tok_s = 0.0;
};

using PrefillProgressCallback = std::function<void(const PrefillProgressEvent& ev)>;

struct RuntimeTelemetry {
    int prompt_tokens = 0;
    int cached_prompt_tokens = 0;
    int completion_tokens = 0;
    double prompt_ms = 0.0;
    double decode_ms = 0.0;
    double prompt_tok_s = 0.0;
    double decode_tok_s = 0.0;
    std::string finish_reason = "stop"; // "stop", "length", "cancel", "error"
    int drafts_accepted = 0;
    int drafts_offered = 0;
    int64_t ram_expert_hits = 0;
    int64_t file_expert_reads = 0;
    int64_t gpu_cache_hits = 0;
    int64_t gpu_cache_lookups = 0;
    int64_t offload_entries = 0;
    uint64_t file_read_bytes = 0;
    uint64_t kv_usage_bytes = 0;
    int64_t current_context_length = 0;
    int64_t prefill_chunk = 0;
    double request_latency_s = 0.0;

    // Detailed inference stage breakdown (Phase 4 performance telemetry)
    double ttft_ms = 0.0;
    double first_window_capture_ms = 0.0;
    double prefill_embed_ms = 0.0;
    double prefill_gpu_dense_ms = 0.0;
    double prefill_gpu_attn_ms = 0.0;
    double prefill_gpu_gdn_ms = 0.0;
    double prefill_host_wait_ms = 0.0;
    double prefill_cpu_group_ms = 0.0;
    double prefill_cpu_quant_ms = 0.0;
    double prefill_cpu_gu_ms = 0.0;
    double prefill_cpu_down_ms = 0.0;
    double prefill_cpu_combine_ms = 0.0;
    double decode_verify_ms = 0.0;
    double decode_mtp_draft_ms = 0.0;
    double decode_host_wait_ms = 0.0;
    double decode_cpu_pool_ms = 0.0;
    double decode_cpu_validation_ms = 0.0;
    double decode_cpu_quant_ms = 0.0;
    double decode_cpu_gu_ms = 0.0;
    double decode_cpu_down_ms = 0.0;
    double decode_ple_ms = 0.0;
    double decode_commit_ms = 0.0;
    double decode_logit_validation_ms = 0.0;
    int64_t decode_windows = 0;

    // Phase 6.6: Adaptive speculative decoding & cost-aware scheduler telemetry
    std::string decode_strategy_mode = "auto";
    std::string decode_strategy_initial = "normal";
    std::string decode_strategy_final = "normal";
    std::string spec_fallback_reason = "none";
    double decode_lookup_draft_ms = 0.0;
    double decode_retrieval_draft_ms = 0.0;
    double decode_scheduler_ms = 0.0;
    double spec_ms_per_committed_token = 0.0;
    double spec_estimated_normal_ms_per_token = 0.0;
    double spec_estimated_speedup = 1.0;
    double spec_avg_proposed_draft = 0.0;
    double spec_avg_accepted_prefix = 0.0;
    double spec_mean_abs_cost_pred_error_ms = 0.0;
    double spec_mean_abs_acc_pred_error = 0.0;
    double spec_effective_hysteresis_margin = 0.05;
    double spec_measured_verify_cv = 0.0;
    int64_t spec_strategy_switches = 0;
    int64_t spec_fallback_count = 0;
    int64_t spec_normal_windows = 0;
    int64_t spec_mtp_windows = 0;
    int64_t spec_lookup_windows = 0;
    int64_t spec_retrieval_windows = 0;
    uint64_t spec_drafter_memory_bytes = 0;
};

struct StartupProfile {
    std::string container_format = "gguf+pack";
    double package_load_ms = 0.0;
    double model_open_parse_ms = 0.0;
    double runner_and_plan_ms = 0.0;
    double cuda_init_ms = 0.0;
    double kernel_provider_resolve_ms = 0.0;
    double quant_layout_prep_ms = 0.0;
    double embed_load_ms = 0.0;
    double dense_index_and_pack_ms = 0.0;
    double native_dense_load_ms = 0.0;
    double native_head_load_ms = 0.0;
    double expert_source_open_ms = 0.0;
    double expert_calibration_pre_ms = 0.0;
    double expert_cache_init_ms = 0.0;
    double numa_expert_pin_ms = 0.0;
    double kv_session_init_ms = 0.0;
    double ple_bind_ms = 0.0;
    double mtp_init_ms = 0.0;
    double verifier_init_ms = 0.0;
    double prefill_init_ms = 0.0;
    double expert_calibration_post_ms = 0.0;
    double graph_capture_startup_ms = 0.0;
    double tokenizer_load_ms = 0.0;
    double total_load_ms = 0.0;

    int64_t container_opens = 0;
    int64_t index_parses = 0;
    int64_t dense_device_allocs = 0;
    int64_t expert_copy_calls = 0;
    uint64_t dense_vram_bytes = 0;
    uint64_t expert_vram_bytes = 0;
    uint64_t expert_ram_bytes = 0;
    uint64_t kv_host_bytes = 0;
    uint64_t kv_device_bytes = 0;
    bool expert_zero_copy = false;
    bool expert_numa_interleaved = false;
    bool expert_page_locked = false;

    // Phase 5 GEXEC ahead-of-time optimization telemetry
    bool gexec_active = false;
    std::string gexec_path;
    std::string gexec_cache_key;
    std::string gexec_cpu_expert_impl;
    std::string gexec_expert_layout = "PortableContiguous";
    std::string gexec_fallback_reason;
};

} // namespace guild::runtime
