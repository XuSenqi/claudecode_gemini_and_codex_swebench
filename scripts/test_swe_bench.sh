cd /data/claudecode_gemini_and_codex_swebench
source .venv/bin/activate

#nohup python swe_bench.py run \
#  --dataset princeton-nlp/SWE-bench_Verified \
#  --limit 500 --backend codex --workers 3 \
#  -o runs/verified-full >> test.log &

nohup python swe_bench.py run \
  --dataset SWE-bench/SWE-bench_Verified \
  --limit 3 --backend codex --workers 3 \
  -o runs/verified-full-test >> test.log &
