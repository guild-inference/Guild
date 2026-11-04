# Nemotron 3.5 Transcribe on the GTX 1070 + dual Xeon: offline transcription

Measured 2026-10-31/11-01 on this PC: Ubuntu (kernel 6.8), two Xeon Gold 6130 (2 NUMA nodes, 32 cores / 64 threads,
AVX-512 without VNNI or BF16), 376 GiB RAM, GTX 1070 8 GiB (sm_61), driver 580.178.04, CUDA 12.0.140, GCC 13.3.0.
Guild branch `feat/nemotron-transcribe` (on `feat/startlux-decision`), Release, `-DGUILD_EXPERIMENTAL_SM60=ON`, with
`GUILD_CUDA_SOURCE_DIR` on guild-kernels-cuda `perf/sm61-decode` `b511c94` and `GUILD_CPU_SOURCE_DIR` on
guild-kernels-cpu `5147f69` (the lock's).

[NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
(OpenMDW-1.1) is a speech-to-text model: an audio model (a speech encoder and a speaker model) feeds a Nemotron-H
MoE language model, which writes the transcript.  Scope here: offline transcription of clips up to 30 s (40 s
fit one chunk), the default verbatim request, thinking off, greedy (temperature 0).  No diarization output,
translation, word timestamps or long-form audio.

## The pieces

| Piece | What it is | Reused | New |
| --- | --- | --- | --- |
| Audio in | 16 kHz WAV, resampled when not | | `src/audio/frontend.cpp`: RIFF WAV as soundfile reads it, a Kaiser-windowed sinc resampler |
| Features | pre-emphasis 0.97, centered STFT (n_fft 512, hop 160, window 400), power, 128 mel bins, log(x + 2^-24), per-feature normalization | | the same file: the window and mel bank the checkpoint stores (BF16), the reference's float32 rounding points and its PyTorch cascade sums |
| Speech encoder | 8 frames stacked -> 1,280, 32 pre-norm blocks (q/k LayerNorm, RoPE on half of each head, GELU MLP), float32 | cuBLAS (Guild already links it) | `src/audio/encoder.cu`: LayerNorm, partial RoPE, softmax with a padding mask, epilogues |
| Speaker model | the same kind of encoder, 31 blocks of 512, a sub-pixel upsampler, a classifier for 8 speakers; thresholded activities -> LayerNorm -> a sinusoidal table, added to the speech encoder's output | | the same file |
| Projector | Linear 1,280 -> 2,688 | | the same file |
| Language model | Nemotron-H: 23 Mamba-2 layers, 23 MoE layers (128 relu^2 experts, top 6, sigmoid router with a selection bias, a shared expert), 6 grouped-query attention layers without positions; an MTP block | the GLM-5.3-Flash engine's design and its pieces: the CPU expert pool and phases and the NUMA split (moved to `cpu_moe`), the product dispatch (MMVQ, the staged one/two-token kernels, MMQ: moved to `dense_mm`), the sigmoid router, RMSNorm, f32 matvec, argmax; the GDN/KDA scans as the template of the Mamba-2 scan | `src/core/nemotron_h.cpp` / `.cu`: the Mamba-2 convolution and selective scan, the gated group norm, relu^2, the attention, the engine; relu^2 experts in `cpu_moe` |
| Weights | a GGUF | llama.cpp's own Nemotron-H converter and `llama-quantize` | `tools/nemotron_lm_gguf.py` (feeds the converter the checkpoint's NeMo-named tensors as lazy views), `tools/nemotron_audio_rt.py` (the audio model's runtime files) |
| Speculative decoding | the MTP block (attention + MoE after `eh_proj([enorm(e) \| hnorm(h)])`, one layer used repeatedly) drafts, the model verifies | GLM's draft / verify / accept scheme | the per-token snapshots of the recurrent states for the rollback |
| Runner | `transcribe_run` | | it, the prompt ids from `prompt.json` (Guild's tokenizer has no rule for this model's pre-tokenization) |

The MTP block ends in a LayerNorm, as llama.cpp's nemotron-h-moe graph has it (vLLM's NemotronHMTP has an RMSNorm
there; `--mtp-rms` switches): one draft a step is kept 95% of the time over the gate's clips.

The speaker model is needed: with its output replaced by "speaker 0 active on every frame", the reference's own bf16
run changes 11 of the 60 gate transcripts (capitals and punctuation; one clip loses both), where bf16 against fp32
changes 3.

Prepare the model once:

```bash
python tools/nemotron_lm_gguf.py --model NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --llama-cpp <llama.cpp> \
    --out LM-BF16.gguf --outtype bf16                                                   # 66 GB, MTP block included
llama-quantize --token-embedding-type bf16 LM-BF16.gguf LM-Q8_0.gguf Q8_0               # 35.3 GB
python tools/nemotron_audio_rt.py --model NVIDIA-Nemotron-3.5-Transcribe-30B-A4B --out audio   # 1.4 GB + prompt.json
transcribe_run --model LM-Q8_0.gguf --audio audio --tokenizer NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/tokenizer.json \
    --mtp 2 clip.wav
```

## The audio model against the reference

`tools/nemotron_audio_parity.py`: `audio_encode` against the reference's dumps (`tools/transcribe_reference.py
--dump`; the audio model is float32 in the reference's bf16 and fp32 runs alike), all 60 gate clips, the valid
frames and tokens:

| Stage | Max \|diff\| | Relative to the stage's largest value | Median of the clips' max |
| --- | ---: | ---: | ---: |
| log-mel | 4.4e-4 | 2.7e-5 | 1.0e-4 |
| speech encoder | 3.3e-4 | 1.7e-5 | 6.2e-6 |
| speaker logits | 1.2e-4 | 3.9e-6 | 3.3e-5 |
| audio model output | 4.2e-4 | 2.1e-5 | 8.7e-6 |
| embeddings (into the language model) | 3.2e-5 | 1.3e-5 | 3.4e-6 |

Speaker activities: 0 of 57,600 (token, speaker) values differ.  The log-mel differences are the reference's own
float32 rounding: on the 28 s clip its features are 1.9e-4 from the same steps in float64, Guild's 7.9e-5.  Two
things had to be the reference's to the bit:
- the per-feature normalization's sums: a mel bin that hardly varies over a clip (VoxPopuli has them) is divided
  by a deviation near 1e-5, and the reference sums in float32 with PyTorch's cascade, so a mean one unit in the last
  place apart moved the speech encoder's output by up to 9.9 on those clips;
- the padding: the reference pads the STFT's frame count (one more than the valid frames) to whole groups of 8,
  so clips whose frames fill whole groups get a group of padding, masked out of attention but read by the speaker
  model's upsampling convolution.

Resampling (not exercised by the gate, whose clips are 16 kHz): the reference's loader (librosa) uses libsoxr's
HQ resampler; Guild's sinc resampler reproduces sines at 8, 22.05, 44.1 and 48 kHz to 3e-8 and stops a 9.5 kHz tone
from 48 kHz to 3e-8 (`audio_frontend_test`).

## The gate

`tests/data/nemotron_transcribe`: 60 clips (2-28 s, 9.6 minutes): 50 LibriSpeech test-clean, 5 test-other,
5 VoxPopuli accented English.  The reference is the checkpoint's transformers implementation on the CPU, greedy,
in fp32; the bounds come from its own bf16 run (the task's rule): bf16 gives 57 of 60 transcripts identical to fp32
and the same WER, so the bounds are 57 identical and the task's 0.2-point WER floor.

| Run | Identical to fp32 | WER against the datasets' text | Drafted ids equal greedy's |
| --- | ---: | ---: | ---: |
| the reference in bf16 | 57 / 60 | 10.27% (fp32: 10.27%) | |
| Guild Q8_0, greedy | 50 / 60 | 10.34% | |
| Guild Q8_0, 1 / 2 / 3 MTP drafts | 50 / 60 | 10.34% | 60 / 60 each |
| llama.cpp `11fe02151`, the same Q8_0 GGUF, the reference's audio embeddings | 48 / 60 | 13.55% | |

And the reference itself with its language model's matrices rounded to Q8_0 (fp32 activations, the embedding and
the router left as the GGUF has them; `tools/transcribe_reference.py --weights-q8`): 55 / 60 identical to fp32, WER
10.34%.  The Q8_0 weights alone fall short of the bf16 bound; the 8-bit activations Guild's and llama.cpp's
products quantize to take Guild to 50 and llama.cpp to 48.

- **The bound on identical transcripts is not met**, and Q8_0 weights cannot meet it with exact fp32 arithmetic
  either (55).  Every one of Guild's ten differences is a choice fp32 makes
  narrowly: the two largest logits 0.05 to 2.8 apart at the first different token (a first word's capital in five,
  a final period or comma in four, "neighbours" for "neighbors").  llama.cpp on the same file differs on 12, and
  agrees with Guild on only 44 transcripts: Q8_0's error flips near-ties that bf16's does not.
- **One VoxPopuli clip** (23 s, `20101124-...-235051_6`) gets an empty transcript from the reference in fp32 and
  bf16 (the end of the turn first, fp32's margin 6.0), and from Guild; llama.cpp writes 92 words of something else
  there, which is most of its WER.
- **WER** is within 0.07 points of fp32's.  (The WER normalization is simple - lower case, punctuation but
  apostrophes removed - so numbers, "Mr." and spellings count as errors against LibriSpeech's upper-case text, and it
  reads higher than the model card's; the gate compares runs under the same normalization.)
- **Speculative decoding** gives the greedy output on every clip with 1, 2 and 3 drafts.  A verify window's rows
  are decode steps bit for bit: the router runs one token at a time, and the CPU experts sum each token's outputs
  in that token's own routing order (in a window's group order, an expert shared by two tokens changed the sum's
  rounding, which this model's router near-ties turned into other tokens).

## Speed

`transcribe_run --repeat 6` on two LibriSpeech test-clean utterances (`perf_clips.jsonl`: 1580-141084-0011,
5.000 s, and 4970-29093-0006, 29.6 s, the closest to 5 and 30 s), the Q8_0 GGUF; the first run of each process is a
warm-up and is dropped, so mean and standard deviation of 5.  Every run of a clip, in every mode, gives the same ids.
End to end is the WAV read to the last token; the real-time factor is that over the audio's length; decode tokens/s
counts the tokens after the first (which the prompt's logits give) over the decode time.

| Clip | Mode | End to end | Real-time factor | Front end | Audio model | Prompt (tokens) | Decode | Decode tokens/s | Drafts kept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.0 s, 21 tokens | greedy | 1001 ms (sd 3) | 0.200 | 9 ms | 57 ms | 534 ms (114) | 400 ms | 50.1 (sd 0.1) | |
| | 1 MTP draft | 950 ms (sd 12) | 0.190 | 9 ms | 57 ms | 563 ms | 319 ms | 62.6 (sd 0.6) | 10 / 10 |
| | 2 MTP drafts | 920 ms (sd 3) | 0.184 | 9 ms | 57 ms | 555 ms | 297 ms | 67.3 (sd 0.1) | 13 / 14 |
| | 3 MTP drafts | 892 ms (sd 10) | 0.178 | 10 ms | 58 ms | 560 ms | 263 ms | 75.9 (sd 0.2) | 15 / 15 |
| 29.6 s, 96 tokens | greedy | 4050 ms (sd 10) | 0.137 | 25 ms | 167 ms | 1884 ms (421) | 1972 ms | 48.2 (sd 0.1) | |
| | 1 MTP draft | 3750 ms (sd 10) | 0.127 | 23 ms | 167 ms | 1968 ms | 1590 ms | 59.7 (sd 0.1) | 46 / 49 |
| | 2 MTP drafts | 3610 ms (sd 11) | 0.122 | 30 ms | 169 ms | 1979 ms | 1429 ms | 66.5 (sd 0.5) | 62 / 66 |
| | 3 MTP drafts | 3562 ms (sd 7) | 0.120 | 24 ms | 169 ms | 1965 ms | 1401 ms | 67.8 (sd 0.1) | 69 / 78 |

- **Decode** reads the dense weights (2.2 GB, GPU) and six experts a layer (1.5 GB a token in RAM) per step: 20 ms
  a token greedy.  The MTP block adds the drafts' passes (its head reads all 131,072 rows) and the catch-up over
  the kept tokens; three drafts are the fastest on both clips (by 1.3 tokens/s over two on the 30 s one, where
  88% of them are kept against two's 94%).
- **The prompt** (the request's 51 text tokens and one row per 80 ms of audio) is about half of a 30 s clip's time
  (223 tokens/s): its products take MMQ on the GPU, and the CPU experts (each token's six, 23 layers) take most of
  it (2.3 of 2.6 s on a 28 s gate clip with the machine busy, `GUILD_MOE_TIMING`); the MTP block's catch-up over
  the prompt adds ~80 ms.  See the known gaps.
- **The audio model** takes 57 ms for 5 s and 167 ms for 30 s (32 + 31 encoder layers in float32 on the GTX 1070);
  the front end 9-30 ms (frames on 16 threads).
- Over the 60 gate clips (9.6 minutes of audio, 1,686 tokens, one process, the machine busy with the reference
  runs at the time): 43.4 decode tokens/s greedy, 59.0 with two drafts (1,052 of 1,148 drafts kept).

Files: `perf_clips.jsonl`, `perf_{greedy,mtp1,mtp2,mtp3}.jsonl` (`transcribe_run --out`, every run), `run_perf.sh`.

## Nothing else moved

The branch (`237706b`, this README aside) against `build-cuda12` (`feat/glm5` `196527e`), one machine lock,
`bench/results/2026-10-31-startlux-decision/regress.sh`, 2026-11-01 01:10-02:25 UTC.  The GLM engine now runs its
CPU experts and its products through the code this branch moved out of it (`cpu_moe`, `dense_mm`):
- **ctest:** 127 tests, 3 failures:
  - `ple_parity`: its fixtures are not on this machine, as on `feat/glm5`.
  - `decision_gate_test` and `transcribe_gate_test`: the two gates, failing as they should with the same results
    as above (StartLux-Decision's: the same flips, 60 over 0.01, the same 95th percentile).
- **Qwen3.8-Flash-Next:** qwen-bench, server flags, base / branch / branch / base, tok/s:
  - prose: 34.26 / 34.10 / 34.16 / 34.11;
  - code: 44.78 / 44.59 / 44.61 / 44.62.
  - All texts identical.
- **GLM-5.3-Flash:**
  - glm5_run, base / branch alternated, tok/s (CPU experts a step):
    - base 21.47 (70.1 ms), 21.71 (69.0), 21.82 (68.7);
    - branch 21.66 (69.5), 21.75 (69.4), 21.71 (69.0).
    - The same ids in all six.
  - guild-generate (the runtime's GLM path): the same text, 21.55 against 21.62 tok/s.
  - `session_score_test` on GLM: as before this branch (reading a prompt in two parts differs from one read by
    max \|dlogit\| 3.03; see the StartLux-Decision README).

## Known gaps

- **The gate's identical-transcript bound** (57 of 60) is not met at Q8_0 (50); see above.
- **Prompt reading** spends most of its time in the CPU experts (all of a 30 s clip's ~420 prompt tokens through
  23 MoE layers: about 0.55 trillion multiply-adds).  A batched expert path with an AVX-512 int8 kernel (the
  qwen35moe `BatchedPrefill` path's, which hard-codes SwiGLU and rows of at most 1,024 hidden values) is not
  reused: noted, not done.
- **Audio over 40 s** is refused: the reference reads longer audio in chunks of 500 encoder frames with context, and
  the speaker model keeps a speaker cache across them.
- **Requests other than the default verbatim one** need their token ids (Guild's tokenizer has no rule for this
  model's pre-tokenization, `pixtral`); `transcribe_run` reads the default's from `prompt.json`.
- **Resampling** is not libsoxr's: a high-quality sinc, not bit for bit (above).
- **No diarization output, word timestamps, translation or reasoning.**
