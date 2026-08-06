#!/bin/bash
set -o pipefail; CODE=/workspace/AICC/code; cd "$CODE"; PY=/venv/qwen35/bin/python
if [ "${_Q35:-}" != "1" ]; then : > run_qwen35_auto.log; export _Q35=1
  nohup setsid bash "$0" "$@" </dev/null >> run_qwen35_auto.log 2>&1 & disown
  echo "✅ qwen35 VQA detached. tail -f run_qwen35_auto.log"; exit 0; fi
log(){ echo "[$(date '+%F %T')] $*"; }
export HF_HOME=/workspace/hf_cache QWEN35_NUM_IMAGES=3 AICC26_QWEN35_QUANT=bf16
log "==== Qwen3.5 VQA FULL train ===="
$PY -u run_qwen35_vqa.py train 2>&1 | grep -avE "FutureWarning|torch._check|Loading checkpoint|^[0-9 ]+%\||fast path|flash|fla-org|causal-conv|UserWarning|warnings.warn" | tail -15
log "train rc=$?"
log "==== Qwen3.5 VQA EVAL per-qtype ===="
$PY -u run_qwen35_vqa.py eval 2>&1 | grep -aiE "orientation|position|TOTAL|n="
log "================ XONG Qwen35-VQA (Qwen2.5: orient39/pos_ped67/pos_veh61/TOTAL81.1) ================"
