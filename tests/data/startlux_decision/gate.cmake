# decision_gate_test: decision_run (RUN) on the fixture's requests with MODEL and PACK, then tools/decision_gate.py
# against the reference package's fp32 answers with the thresholds in gate.json (derived from the package's own
# bf16 run: a top-answer flip only where fp32's top two are within 0.05; no more questions over 0.01, and no larger
# 95th percentile of max|dp|, than the bf16 run).  CMakeLists.txt registers it when GUILD_STARTLUX_MODEL and
# GUILD_STARTLUX_PACK are set.
set(D ${SRC}/tests/data/startlux_decision)
execute_process(COMMAND ${RUN} --native ${MODEL} --pack ${PACK} --resident-budget-gib 40 --requests ${D}/requests.jsonl
                        --out ${OUT}/decision_gate.jsonl --detail --fresh
                RESULT_VARIABLE run_rc)
if(NOT run_rc EQUAL 0)
  message(FATAL_ERROR "decision_run failed: ${run_rc}")
endif()
execute_process(COMMAND ${PYTHON} ${SRC}/tools/decision_gate.py --ref ${D}/reference_fp32.jsonl
                        --got ${OUT}/decision_gate.jsonl --thresholds ${D}/gate.json --requests ${D}/requests.jsonl
                RESULT_VARIABLE gate_rc)
if(NOT gate_rc EQUAL 0)
  message(FATAL_ERROR "the gate failed")
endif()
