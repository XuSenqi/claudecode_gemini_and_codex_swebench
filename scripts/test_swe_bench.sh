cd /data/claudecode_gemini_and_codex_swebench
source .venv/bin/activate

export CODE_SWE_MAX_STEPS=250
export CODE_SWE_MAX_REPEATS=10
export CODE_SWE_INSTANCE_TIMEOUT=7200          # 2 hours
export CODE_SWE_CODEX_IDLE_TIMEOUT_MS=120000   # 2 min with no tokens → retry the sampling request
export CODE_SWE_CODEX_STREAM_RETRIES=10        # extra SSE/idle-timeout attempts
export CODE_SWE_CODEX_REQUEST_RETRIES=10       # extra attempts on 5xx / connect failures

#nohup python swe_bench.py run \
#  --dataset princeton-nlp/SWE-bench_Verified \
#  --limit 500 --backend codex --workers 3 \
#  -o runs/verified-full >> test.log &

nohup python swe_bench.py run \
  --dataset SWE-bench/SWE-bench_Verified \
  --limit 3 --backend codex --workers 3 \
  -o runs/verified-full-test >> test.log &
