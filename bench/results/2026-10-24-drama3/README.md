# Fish Audio Drama 3: text to speech on the dual Xeon + GTX 1070

Measured 2026-10-24 on this PC: Ubuntu, two Xeon Gold 6130 (2 NUMA nodes), 376 GiB RAM, GTX 1070 8 GiB (sm_61),
driver 580.178.04, CUDA 12 Release build of `guild-generate` on branch `feat/drama3`
(`-DGUILD_EXPERIMENTAL_SM60=ON`, `guild-kernels-cuda` checkout at `perf/sm61-decode`). The model is
[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3), revision `b2b5ee99`.

## What Drama 3 is, and how Guild runs it

Drama 3 is a Qwen3.5-MoE backbone with Ornith's topology (40 layers, 30 Gated DeltaNet and 10 gated full attention,
256 experts top 8 with a gated shared expert, hidden 2048, FP8 experts), whose vocabulary has 4,096 more tokens: the
audio band `<|semantic:0|>` to `<|semantic:4095|>`.  Each step emits one audio frame of 10 codes, 21.5 frames per
second of audio:

1. The backbone's head picks the frame's main token: end of speech, or `<|semantic:c0|>`.  Only those 4,097 rows of
   its 252,288-row head are needed, so the engine skips the head (`Verifier::set_no_head`) and the tower runs them.
2. The codebook tower (13 transformer layers over the frames, about 1.2 B parameters) predicts codes 1 to 9, one
   layer after the other.
3. The codec (fish-speech's modded DAC, about 184 M parameters to decode) turns the frames into 44.1 kHz audio,
   2048 samples per frame.
4. The next backbone input is `(embed(main) + codebook-1 embedding) / sqrt(2)` (`Verifier::set_audio_embed`).

| Part | Where it runs | Format | VRAM |
| --- | --- | --- | ---: |
| backbone | the qwen35moe path: experts on the CPU pool, dense on the GPU | GGUF Q4_K_M, 21.2 GB | |
| tower | GPU, `Drama3Tower` | Q8_0 projections, BF16 embeddings, FP16 K/V | 1525 MiB for 10240 frames of K/V |
| codec | GPU, `Drama3Codec`, cuBLAS SGEMM per convolution tap | FP32 | 0.75 GiB weights + activations |

Whole engine, peak while speaking: 5163 MiB used on the 8 GiB card (5023 MiB before the batched prompt of
2026-10-25, whose buffers and wider decode window take the difference).

## Checks against the reference

fish-speech's own modules (commit `5415669`, "Update for Drama-3"), run in float32 on the CPU, as the reference.

| What | Result |
| --- | --- |
| prompt rows for the test sentence (`build_drama3_prompt`) | identical (40 rows) |
| tower, fed the reference's hidden states, teacher-forced with its 40 frames | every layer's output cosine >= 0.9995; heads' distributions: mean KL 0.0015 nats, worst row 0.0066 (`tests/core/drama3_tower_test.cpp`) |
| backbone Q4_K_M, teacher-forced with the reference's frames | final-norm hidden state cosine 0.993 on average, 0.985 or more on audio rows, 0.954 at worst (a prompt row) |
| codec decode, 60 frames of random codes, streamed in chunks of 1, 7, 16 or 32 frames | 110 to 111 dB SNR against the reference's one-call decode (`tests/core/drama3_codec_test.cpp`) |
| codec encode, 450 frames of speech at 44.1 kHz | all 4,500 codes equal |
| codec encode of the model card's 24 kHz clip, resampled by the engine | all 850 codes equal |

The engine's frames are sampled with its own random stream, so they equal the reference's only in distribution.
Whisper-small (`openai/whisper-small`) transcribes every output below as written; it writes "Ninety-seven" as "97" in
some runs.

## Speed

`guild-generate --drama3`, the Q4_K_M model, `--max-context 10240 --kv fp16 --kv-host-only --resident-budget-gib 24`,
the model card's sampling (T 1.0, top-p 0.9, top-k 30), a 290-character passage, three seeds each:

| Voice | Audio | Frames/s | Real time | Prompt |
| --- | --- | ---: | ---: | --- |
| none | 22.24 / 19.78 / 22.06 s | 37.70 / 38.00 / 38.01 | 1.75x / 1.76x / 1.77x | 86 rows, 0.70 / 0.69 / 0.68 s |
| the model card's 3.9 s clip | 18.67 / 17.14 / 16.49 s | 37.60 / 37.23 / 37.57 | 1.75x / 1.73x / 1.74x | 206 rows, 1.57 / 1.58 / 1.59 s |

Measured 2026-10-25, after the prompt was batched (below); the same seeds give the same audio as before.  The first
measurement, 2026-10-24, had the same frame rates and prompts of 1.42 / 1.38 / 1.44 s (86 rows) and
3.31 / 3.22 / 3.23 s (206 rows).

The prompt: the backbone runs it through decode windows of 8 rows (4 before; `kVerifyMaxT`), and the tower takes
all its rows but the last in one batched pass (`Drama3Tower::prefill`, 64 columns per projection call) instead of a
step per row.  The batched pass writes K/V bit-identical to the row-by-row steps (`drama3_tower_test` checks every
layer and position).

Per frame, about 26.5 ms: the tower 9.7 ms (12.1 with the 206-row voice prompt: more K/V to attend), the codec
2.6 ms, the backbone the rest.  Encoding a reference
clip took 330 ms for 3.9 s of audio (the encoder's weights are uploaded for the call and freed after).

Through the server (`serve/server.py`, the same engine): a sentence in the `en0` voice from `/v1/audio/speech` in
4.5 s (2.2 s on 2026-10-25 for "The storm passed before midnight.", streamed or not), a streamed sentence (`stream_format: audio`) in 2.0 s, and a `/v1/tts` request bringing its own reference clip
in 4.2 s, end to end.

## Not done yet

- The backbone still takes the prompt 8 rows at a time: 206 rows take 1.6 s before the first frame, most of it
  the backbone.  A prefill path for its audio rows would cut the time to the first audio again.
- The HIP build is not tested on an AMD card; the tower's and codec's kernels assume 32-wide warps in places.
- The tower's 10 samplings per frame read their logits back to the host; on the GPU, with the tower in a CUDA graph,
  each frame would save the launches and the round trips.
- `guild pull` / setup.py do not know Drama 3 yet; the model is converted by hand (see the README).
- The web app has no speech page; the API serves it.
