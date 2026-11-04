# Decode, tenth round: GLM-5.3-Flash verify windows of up to four tokens

GLM-5.3-Flash's verify window took at most two tokens: its KDA layers kept one rollback point, the recurrent state
after the window's first token. So the DFlash drafter's block of seven drafts was cut to one. The window now takes up
to four tokens, with the same texts as plain greedy decoding:

- **The KDA layers' rollback.** A window's first group reads the recurrent state from `kda_state` and writes it to a
  second buffer, `kda_alt`; the second group goes on in that buffer. `accept` then either swaps the two buffers (every
  token kept) or replays the kept tokens from the window's start, still in `kda_state`, with the convolved q|k|v rows,
  gates and betas each layer kept for the window. The short convolution's state is small, so it keeps a copy after
  each token instead (`kda_conv`'s `mid`, `nmid`). The VRAM: one more state a layer, as before, plus 18 MiB.
- **Exactness.** With two tokens in a pipelined group the texts parted from plain greedy decoding. The GPU kernels
  were not the cause: the GEMVs (the routers', the indexer's), `q8_0_heads`, the hyper-connection pre-mix and the
  DSA attention each give a 2-token group the bits of two 1-token calls (`glm5_kernels_test`, new). The CPU experts'
  multi-token kernels did not: they sum a token's rows in an order that follows the window's experts. A group of two
  or more tokens now runs them in `single` mode, each (token, row) as a decode step computes it.
- **The window rule.** DFlash's drafts are verified while their cumulative probability under the drafter stays at or
  above 0.7 (`GUILD_GLM_DFLASH_CUM`; `GUILD_GLM_DFLASH_K` fixes the count), up to three. NextN still drafts one token.

PC: 2x Xeon Gold 6130 (16 cores each, AVX-512 BW without VNNI, 2 NUMA nodes), GTX 1070 8 GB (sm_61, 15 SMs, 151 W
cap), CUDA 12. Branch `feat/nemotron-transcribe`. GLM-5.3-Flash UD-IQ4_XS (Unsloth), its attention re-quantized to
Q6_K at load; incoai's GLM-5.3-Flash DFlash 2 checkpoint (CC BY-NC-ND 4.0; opt-in, setup does not fetch it).

## Results

`glm5_run --spec --prompts`: the four chat prompts of round 9 (thinking closed), 256 tokens each from a reset
context, a 262,144-token context, `GUILD_GLM_SHEXP_SPLIT=0` (the verify equals greedy decoding). Separate loads,
alternated over 2 rounds, tok/s:

| Prompt | NextN (one draft a step) | DFlash, cumulative probability >= 0.7 | DFlash's steps with 0 / 1 / 2 / 3 drafts |
| --- | --- | --- | --- |
| prose | 21.07 / 20.98 | 21.05 / 21.39 | 18 / 38 / 34 / 27 |
| code | 21.38 / 21.49 | 22.68 / 23.89 | 18 / 25 / 15 / 41 |
| story | 19.99 / 19.77 | 17.78 / 18.60 | 47 / 55 / 30 / 16 |
| sql | 22.51 / 22.26 | 25.83 / 26.20 | 5 / 11 / 8 / 56 |

- **Texts.** Each arm's texts are the same in both rounds. NextN's equal round 9's, to the token. DFlash's equal plain
  greedy decoding's in the same process for all four prompts (`--plain`). The two arms' texts differ from each other:
  the drafter's 1017 MiB moves 36 shared experts to their CPU copies, which round differently.
- **DFlash at one draft a step** (round 9's rule; the same texts, earlier in the round): prose 20.19, code 21.20,
  story 17.27, sql 22.03.
- **Against NextN**: prose +0.9%, code +8.6%, sql +16.2%, story -8.5%. NextN stays GLM's drafter: it comes with the
  GGUF, and on the story prompt DFlash's first draft is kept at 59% of the steps against NextN's 71%.
- **The server** (`guild-generate --serve`, the same flags): for each prompt, 128 tokens greedy and sampled
  (temperature 0.8, seed 42, repeat penalty 1.1, last 64) with the drafter's windows equal those of
  `decode_strategy=normal` in the same process. Greedy: prose 19.3, code 23.1, story 18.2, sql 24.6 tok/s, against
  13.9-14.9 without drafts.

## The window size

`glm5_run --spec --prompts` as in Results, the DFlash arm, one process a setting. Every setting decodes the same
texts (plain greedy decoding's at this placement), so the columns compare the same tokens:

| Drafts a step | prose | code | story | sql |
| --- | --- | --- | --- | --- |
| 1 (before) | 20.19 | 21.20 | 17.27 | 22.03 |
| 2 | 19.07 | 23.27 | 15.73 | 21.71 |
| 3 | 18.58 | 23.34 | 14.16 | 22.39 |
| cumulative probability >= 0.3 | 19.88 | 23.32 | 16.42 | 24.86 |
| cumulative probability >= 0.5 | 20.50 | 23.08 | 17.33 | 25.91 |
| cumulative probability >= 0.7 | 21.15 | 23.39 | 17.32 | 26.16 |
| cumulative probability >= 0.85 | 20.90 | 23.35 | 16.86 | 25.35 |

(tok/s; these runs had the CPU experts in `single` mode for one-token groups too, which costs a few ms a step; Results
has the final binary.)

A step's median time by window, the same runs: about 84-89 ms for two tokens, 113-124 for three, 130-144 for four,
against 68-69 ms for plain decoding. The CPU's experts take most of it: 68-75 ms of a two-token step, 94-102 of a
three-token one and 112-128 of a four-token one: a group of two tokens computes both tokens' experts.

## Kept for the record

| What | Result |
| --- | --- |
| Three or four tokens with the CPU experts' multi-token kernels | 3 drafts a step: prose 18.85, code 22.83, story 14.54, sql 24.27 tok/s, but every text parted from plain greedy decoding (at tokens 18-187) |

## Open

- **Chained NextN drafts.** NextN keeps its single draft; a second draft from the NextN block's own hidden state is not
  built. Its first draft is accepted 71-93% of the time.

## Tools

- `glm5_kernels_test`: `test_window_invariance` (a group of two tokens against two one-token calls, bit for bit) and
  the KDA rollback checks (`kda_conv` snapshots after each token, the two-buffer window, the replay).
- `glm5_run`: `windows <name> <arm>` lines (steps by drafts offered and kept) and each draft's probability.
