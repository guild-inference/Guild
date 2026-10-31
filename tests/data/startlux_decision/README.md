# StartLux-Decision-v1-35B-A3B gate fixture

- `requests.jsonl`: 244 /v1/systemone requests, 255 questions:
  - the 231 public JevBench items (internlm/Intern-Decision, `benchmarks/accuracy-v1/jevbench`, MIT; `expected` is
    their gold answer);
  - the model card's example and edge cases of the prompt rendering;
  - four long inputs (8.9K to 31K tokens of Guild's own docs and C++, and a 20K-token JSON state).
  `make_requests.py` built it.
- `reference_fp32.jsonl`, `reference_bf16.jsonl`: the checkpoint's own package (`startlux_decision/`) on the CPU
  (`tools/decision_reference.py`), each question's prompt tokens (fp32 file), option letters' logits,
  temperature and probabilities, and the answers `decide()` returns.
- `gate.cmake`: `decision_gate_test` (opt-in: `GUILD_STARTLUX_MODEL`, `GUILD_STARTLUX_PACK`) runs `decision_run` on
  the requests and `tools/decision_gate.py` against the fp32 reference.
