#!/bin/bash
# ctx_test_pipeline.sh — Predict test với adapter ctx-only, route 3 spatial (ori+pos_ped+pos_veh)
# qua Viterbi lên nền WSD2 -> SUBMIT_Q35_CTX. Chỉ deploy phần đã gate PASS split-stable (+0.27 val).
set -o pipefail; CODE=/workspace/AICC/code; cd "$CODE"; PY=/venv/qwen35/bin/python
if [ "${_CT:-}" != "1" ]; then : > ctx_test_pipeline.log; export _CT=1
  nohup setsid bash "$0" </dev/null >> ctx_test_pipeline.log 2>&1 & disown
  echo "✅ ctx TEST pipeline launched. tail -f ctx_test_pipeline.log"; exit 0; fi
log(){ echo "[$(date '+%F %T')] $*"; }
export HF_HOME=/workspace/hf_cache
export AICC26_DATA_DIR=$CODE/output_qwen35/processed_ctx
export AICC26_QWEN35_ADAPTER=$CODE/output_qwen35/vqa_lora_ctx
W=$CODE/output_qwen35; D=$W/processed_ctx; ROUTE="orientation,position_ped,position_veh"

f=0; while [ $f -lt 2 ]; do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && f=$((f+1)) || f=0; sleep 20; done

log "==== PREDICT TEST ctx (3 spatial types only, ~1.5h) ===="
ok=0
for try in 1 2 3; do
  QWEN35_BATCH=4 QWEN35_SAVE_PROBS=$W/ctx_test_probs.json $PY -u run_qwen35_vqa.py predict \
    --samples $D/vqa_test_ctx.json --out $W/ctx_test_preds.json --qtypes "$ROUTE" 2>&1 | tail -2
  [ $? -eq 0 ] && [ -s $W/ctx_test_probs.json ] && { ok=1; break; }
  log "predict attempt $try fail"; sleep 30
done
[ $ok -eq 1 ] || { log "❌ predict fail"; exit 1; }

log "==== ROUTE Viterbi 3 spatial lên nền WSD2 -> SUBMIT_Q35_CTX ===="
$PY -u wsd.py decode --samples test_root_work_full/processed/vqa_val.json --probs $W/ctx_test_probs.json \
  --base-answers /workspace/AICC/SUBMIT_Q35_WSD2/subtask2_vqa.json --out $W/subtask2_vqa_ctx.json --types "$ROUTE"
DEST=/workspace/AICC/SUBMIT_Q35_CTX; mkdir -p "$DEST"
cp /workspace/AICC/SUBMIT_Q35_BESTMIX/subtask1_captioning.json "$DEST"/    # caption tốt nhất (29.96)
cp $W/subtask2_vqa_ctx.json "$DEST"/subtask2_vqa.json
log "SUBMIT_Q35_CTX sẵn sàng (caption 29.96 + VQA ctx-routed 3 spatial trên nền WSD2)"
log "================ XONG CTX-TEST ================"
