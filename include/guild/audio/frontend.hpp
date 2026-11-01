// include/guild/audio/frontend.hpp - audio in, log-mel features out: the speech models' front end on the CPU.
//
// The steps and their arithmetic follow the Nemotron 3.5 Transcribe reference (transformers'
// NemotronAsrStreamingFeatureExtractor, whose audio transformers' load_audio reads with librosa):
//   read_wav   a RIFF WAV file as float32 samples, as soundfile reads it (PCM / 32768 ...), channels averaged
//   resample   to the model's rate (a Kaiser-windowed sinc; the reference uses libsoxr's HQ resampler, which this
//              approximates: docs and the frontend test give the measured difference)
//   log_mel    pre-emphasis, a centered STFT (zero padded) with the stored window, power spectrum, the stored mel
//              filter bank, log(x + guard); frames past the audio zeroed, padded to a multiple of `stack` frames
//   normalize  each mel bin to zero mean and unit (unbiased) standard deviation over the valid frames
// Every single-precision step rounds where the reference's float32 tensors round; the sums (FFT, filter bank,
// statistics) are taken in double and rounded once, so they differ from the reference's float32 sums by about one
// rounding.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace guild::audio {

struct Wave {
    std::vector<float> samples; ///< mono
    int rate = 0;               ///< samples a second
};

/// Reads a RIFF/WAVE file: PCM 8/16/24/32-bit or IEEE float 32/64, any channel count (averaged to mono).
bool read_wav(const std::string& path, Wave& out, std::string& err);
/// Writes mono float32 samples as a 16-bit PCM WAV file (tests and tools).
bool write_wav_pcm16(const std::string& path, const std::vector<float>& samples, int rate, std::string& err);

/// `x` at `from` samples a second resampled to `to`: ceil(n * to / from) samples, time-aligned (output sample i is at
/// time i / to), a linear-phase low-pass at min(from, to) / 2 (Kaiser window, passband to 0.913 of it, about 125 dB
/// stop band).  from == to returns x unchanged.
std::vector<float> resample(const std::vector<float>& x, int from, int to);

struct MelConfig {
    int rate = 16000, n_fft = 512, hop = 160, win = 400, n_mels = 128, stack = 8;
    float preemphasis = 0.97f;
    float guard = 5.960464477539063e-08f; ///< 2^-24, added before the log
    std::vector<float> window;            ///< `win` values (centered in the n_fft frame)
    std::vector<float> filters;           ///< n_mels x (n_fft / 2 + 1), row-major
};

struct Features {
    int frames = 0; ///< valid frames: floor(samples / hop)
    int padded = 0; ///< frames held: the STFT's frames (frames + 1) rounded up to a multiple of MelConfig::stack
    int n_mels = 0;
    std::vector<float> mel; ///< padded x n_mels, frame-major; zero past `frames`
};

/// The log-mel features of `n` samples at cfg.rate.
bool log_mel(const float* x, int64_t n, const MelConfig& cfg, Features& out, std::string& err);
/// Per-feature normalization over the valid frames (padded frames stay zero): padded x n_mels, in the reference's
/// float32 order (PyTorch's cascade sums over every frame held).
/// The audio tokens: one per `stack` valid frames, the last one partial: (frames + stack - 1) / stack.
std::vector<float> normalize_per_feature(const Features& f);

} // namespace guild::audio
