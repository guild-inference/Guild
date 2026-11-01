# transcribe_gate_test: transcribe_run (RUN) on the fixture's clips (CLIPS: make_clips.py's output directory) with
# MODEL, AUDIO and TOKENIZER, greedy and with two MTP drafts, then tools/transcribe_gate.py against the reference's
# fp32 transcripts with the bounds in gate.json (derived from the reference's own bf16 run: at least as many
# transcripts identical to fp32's, the WER within 0.2 points), and the drafted run's ids equal to greedy's on every
# clip.  CMakeLists.txt registers it when GUILD_NEMOTRON_MODEL, _AUDIO, _TOKENIZER and _CLIPS are set.
set(D ${SRC}/tests/data/nemotron_transcribe)
foreach(mode greedy mtp2)
  if(mode STREQUAL "greedy")
    set(flags --no-mtp)
  else()
    set(flags --mtp 2)
  endif()
  execute_process(COMMAND ${RUN} --model ${MODEL} --audio ${AUDIO} --tokenizer ${TOKENIZER} ${flags}
                          --clips ${CLIPS}/clips.jsonl --out ${OUT}/transcribe_${mode}.jsonl
                  RESULT_VARIABLE run_rc)
  if(NOT run_rc EQUAL 0)
    message(FATAL_ERROR "transcribe_run (${mode}) failed: ${run_rc}")
  endif()
endforeach()
execute_process(COMMAND ${PYTHON} ${SRC}/tools/transcribe_gate.py --ref ${D}/reference_fp32.jsonl
                        --got ${OUT}/transcribe_greedy.jsonl --clips ${D}/clips.jsonl --thresholds ${D}/gate.json
                RESULT_VARIABLE gate_rc)
execute_process(COMMAND ${PYTHON} ${SRC}/tools/transcribe_gate.py --ref ${D}/reference_fp32.jsonl
                        --got ${OUT}/transcribe_mtp2.jsonl --clips ${D}/clips.jsonl --same-as ${OUT}/transcribe_greedy.jsonl
                RESULT_VARIABLE spec_rc)
if(NOT gate_rc EQUAL 0 OR NOT spec_rc EQUAL 0)
  message(FATAL_ERROR "the gate failed (transcripts ${gate_rc}, drafted against greedy ${spec_rc})")
endif()
