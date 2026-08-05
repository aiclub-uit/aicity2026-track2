#!/bin/bash
# ctx_pipeline.sh — Retrain VQA ctx+bbox (chống Sim2Real: bbox-dropout 30%, ctx scheduled-sampling 25%)
# [1] full train (adapter vqa_lora_ctx, KHÔNG đụng vqa_lora best)
# [2] predict val ×3 config (both/ctx/bbox) + save probs
# [3] gate: gate_ctx.py per-qtype vs baseline + Viterbi compound + split-half
set -o pipefail; CODE=/workspace/AICC/code; cd "$CODE"; PY=/venv/qwen35/bin/python
if [ "${_CX:-}" != "1" ]; then : > ctx_pipeline.log; export _CX=1
  nohup setsid bash "$0" </dev/null >> ctx_pipeline.log 2>&1 & disown
  echo "✅ ctx pipeline launched. tail -f ctx_pipeline.log"; exit 0; fi
log(){ echo "[$(date '+%F %T')] $*"; }
export HF_HOME=/workspace/hf_cache
export AICC26_DATA_DIR=$CODE/output_qwen35/processed_ctx
export AICC26_QWEN35_ADAPTER=$CODE/output_qwen35/vqa_lora_ctx
W=$CODE/output_qwen35; D=$W/processed_ctx

log "==== [1] FULL TRAIN (7641 samples, 1 epoch) ===="
$PY -u run_qwen35_vqa.py train 2>&1 | grep -aE "loss|epoch|samples|saved|Error" | tail -20
[ -s "$AICC26_QWEN35_ADAPTER/adapter_model.safetensors" ] || { log "❌ train fail"; exit 1; }
log "train xong -> $AICC26_QWEN35_ADAPTER"

log "==== [2] PREDICT VAL x3 config ===="
for cfg in both ctx bbox; do
  ok=0
  for try in 1 2 3; do
    QWEN35_BATCH=4 QWEN35_SAVE_PROBS=$W/ctx_val_probs_$cfg.json $PY -u run_qwen35_vqa.py predict \
      --samples $D/vqa_val_$cfg.json --out $W/ctx_val_preds_$cfg.json 2>&1 | tail -2
    [ $? -eq 0 ] && [ -s $W/ctx_val_probs_$cfg.json ] && { ok=1; break; }
    log "predict $cfg attempt $try fail"; sleep 30
  done
  [ $ok -eq 1 ] || { log "❌ predict $cfg fail"; exit 1; }
  log "predict $cfg xong"
done

log "==== [3] GATE ===="
$PY -u gate_ctx.py 2>&1 | tee $W/ctx_gate.txt
log "================ XONG CTX ================"
