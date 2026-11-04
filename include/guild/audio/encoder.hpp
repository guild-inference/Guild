// include/guild/audio/encoder.hpp - Nemotron 3.5 Transcribe's audio model on the GPU: log-mel features in, one
// language-model embedding per 8 frames out (tools/nemotron_audio_rt.py writes its files and describes the layers).
//
// Float32 throughout, as the reference runs it: the BF16 matrices are widened to float32 a layer at a time and
// multiplied with cuBLAS SGEMM; norms, softmax, GELU and the speaker term in float32.  Clips up to `diar_chunk`
// encoder frames (500: 40 s), which the speech and speaker encoders each read in one chunk; longer audio is
// refused (the reference's chunked reading with context and its speaker cache are not implemented).
#pragma once

#include "guild/audio/frontend.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace guild::audio {

struct AudioGeometry {
    int sample_rate = 16000, n_fft = 512, hop = 160, win = 400, n_mels = 128, stack = 8;
    float preemphasis = 0.97f, log_guard = 5.960464477539063e-08f, ln_eps = 1e-5f;
    int asr_layers = 32, asr_dim = 1280, asr_heads = 10, asr_ff = 5120;
    float asr_rope_fraction = 0.5f, asr_rope_base = 10000.f;
    int diar_layers = 31, diar_dim = 512, diar_heads = 8, diar_ff = 2048, diar_hidden = 192, speakers = 8;
    float diar_rope_base = 10000.f, speaker_threshold = 0.5f;
    int diar_chunk = 500, out_dim = 2688;
};

/// What encode() can also hand back, for the parity tools
struct AudioDetail {
    std::vector<float> asr;            ///< the speech encoder's output before asr_norm: tokens x asr_dim
    std::vector<float> speaker_logits; ///< the speaker model's logits: (tokens * stack) x speakers
    std::vector<float> audio;          ///< asr_norm output + the speaker term: tokens x asr_dim
};

class AudioEncoder {
public:
    AudioEncoder();
    ~AudioEncoder();
    AudioEncoder(const AudioEncoder&) = delete;
    AudioEncoder& operator=(const AudioEncoder&) = delete;

    /// `dir` holds audio.txt, audio_index.txt and audio.bin
    bool load(const std::string& dir, std::string& err);
    const AudioGeometry& geometry() const;
    /// The front end's configuration, with the stored window and mel filter bank
    const MelConfig& mel_config() const;
    uint64_t vram_bytes() const;
    /// Keeps a float32 copy of every BF16 matrix on the device, so that encode() skips the widening (the same
    /// products: 2x Xeon Gold 6130 + GTX 1070, about 35 ms less a clip), if the device still has `keep_free` bytes
    /// free after (twice the BF16 bytes: 2.9 GB for this model); false, and the widening a layer at a time, if not.
    bool keep_wide(uint64_t keep_free);

    /// The embeddings of `f` (from log_mel with mel_config()): one row of out_dim floats per audio token,
    /// (f.frames + stack - 1) / stack of them (host).  AudioDetail holds every token the model reads (f.padded /
    /// stack).
    bool encode(const Features& f, std::vector<float>& embeds, std::string& err, AudioDetail* detail = nullptr);
    /// Milliseconds of the last encode (GPU work and copies)
    double last_ms() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace guild::audio
