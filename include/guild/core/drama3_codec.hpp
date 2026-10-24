// include/guild/core/drama3_codec.hpp - Fish Audio Drama 3's audio codec, decode path, on the GPU.
//
// The codec (fish-speech's modded DAC, shared with S2-Pro) turns frames of 10 codes into 44.1 kHz audio, 2048
// samples per frame; tools/drama3_codec_rt.py describes the network and writes its runtime files.  Every layer is
// causal, so the decode streams: `decode` takes the next frames of an utterance and returns their samples, carrying
// each convolution's left context and the transformer's last window - 127 frames of K/V per layer - between calls.
// A stream cut into chunks therefore gives the samples of one whole-utterance decode (to float rounding: the GEMMs
// see other shapes).  The transformer's RoPE positions count from the stream's start; attention only sees their
// differences, so this matches the reference's per-call positions.
//
// FP32 throughout (cuBLAS SGEMM per convolution tap: no im2col buffer).  Codes are clamped as the reference does:
// codebook 0 to 4095, the rest to 1023 (the tower's heads are 4096-way).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace guild::core {

class Drama3Codec {
public:
    Drama3Codec() = default;
    ~Drama3Codec();
    Drama3Codec(const Drama3Codec&) = delete;
    Drama3Codec& operator=(const Drama3Codec&) = delete;

    /// Loads `dir` (tools/drama3_codec_rt.py) onto the current device; one `decode` takes up to `max_chunk` frames.
    bool load(const std::string& dir, int max_chunk, std::string& err);
    int sample_rate() const {
        return sample_rate_;
    }
    int frame_samples() const {
        return frame_samples_;
    }
    uint64_t vram_bytes() const {
        return vram_;
    }
    /// Starts a new utterance (clears the carried context).
    void reset();
    /// Decodes the next `n` frames (host, [n][10] codes) of the utterance into `pcm` (host, n * frame_samples floats
    /// in [-1, 1]).  n > max_chunk is decoded in pieces.
    bool decode(const int32_t* codes, int n, float* pcm, std::string& err);

    struct Impl;

private:
    Impl* impl_ = nullptr;
    int sample_rate_ = 44100, frame_samples_ = 2048;
    uint64_t vram_ = 0;
};

} // namespace guild::core
