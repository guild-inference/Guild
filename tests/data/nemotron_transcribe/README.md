# Nemotron 3.5 Transcribe gate fixture

- `clips.jsonl`: 60 clips of 2-28 s, 9.6 minutes:
  - 50 LibriSpeech test-clean utterances (at most 2 a speaker; 3 of 25-30 s, 5 more of 20 s or longer);
  - 5 LibriSpeech test-other utterances (5 speakers);
  - 5 VoxPopuli accented-English test utterances (German, French, Polish, Finnish and Spanish speakers).

  `text` is the dataset's own transcript.  The audio is not here: `make_clips.py <data dir> <out dir>` writes the
  same clips (16 kHz PCM16 WAV) and the same `clips.jsonl` from the public archives it names.
- `reference_fp32.jsonl`, `reference_bf16.jsonl`: the checkpoint's transformers implementation on the CPU
  (`tools/transcribe_reference.py`), greedy with the default verbatim request and thinking off, the language
  model in fp32 and in bf16 (the audio model is fp32 in both, as the checkpoint keeps it): the prompt ids, the
  generated ids and text, and each step's margin between the two largest logits.
- `gate.json`: the bounds, from the bf16 run against the fp32 one: 57 of 60 transcripts identical, and the WER
  0.00 points apart, so the task's 0.2-point floor applies.
- `gate.cmake`: `transcribe_gate_test` (opt-in: `GUILD_NEMOTRON_MODEL`, `GUILD_NEMOTRON_AUDIO`,
  `GUILD_NEMOTRON_TOKENIZER`, `GUILD_NEMOTRON_CLIPS`) runs `transcribe_run` greedy and with two MTP drafts and
  `tools/transcribe_gate.py`: the greedy transcripts against fp32 with those bounds, and the drafted ids against
  greedy's on every clip.

The WER here normalizes both sides the same simple way (lower case, punctuation but apostrophes removed): numbers,
"Mr." and spelling variants count as errors against LibriSpeech's upper-case text, so it reads higher than the
model card's.  The gate compares runs under the same normalization.
