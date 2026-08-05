#!/bin/bash
# WSD-v2: Viterbi cho 5 qtype mới (direction/speed/attention/distance/action).
# Chạy SAU D4. Val predict probs (full) -> gate per-type -> chỉ decode test các type Δ>=+1.0
# -> SUBMIT_Q35_WSD2 = caption V4 + VQA (WSD spatial + WSD2 types mới).
set -o pipefail; CODE=/workspace/AICC/code; cd "$CODE"; PY=/venv/qwen35/bin/python
if [ "${_W2:-}" != "1" ]; then : > wsd_v2.log; export _W2=1
  nohup setsid bash "$0" </dev/null >> wsd_v2.log 2>&1 & disown
  echo "✅ WSD-v2 queued (sau D4). tail -f wsd_v2.log"; exit 0; fi
log(){ echo "[$(date '+%F %T')] $*"; }
export HF_HOME=/workspace/hf_cache
W=$CODE/output_qwen35; T5="direction,speed,attention,distance,action"

log "Chờ D4 xong + GPU free..."
while ! grep -qa "XONG D4-CBTW" d4_cbtw.log 2>/dev/null; do sleep 180; done
f=0; while [ $f -lt 2 ]; do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && f=$((f+1)) || f=0; sleep 30; done

log "==== [1] VAL predict FULL + probs ===="
ok=0
for try in 1 2 3; do
  QWEN35_BATCH=4 QWEN35_SAVE_PROBS=$W/wsd2_val_probs.json $PY -u run_qwen35_vqa.py predict \
    --samples output_qwen7b/processed/vqa_val.json --out $W/wsd2_val_preds.json 2>&1 | tail -2
  [ $? -eq 0 ] && [ -s $W/wsd2_val_probs.json ] && { ok=1; break; }
  log "attempt $try fail"; sleep 30
done
[ $ok -eq 1 ] || { log "❌ val predict fail"; exit 1; }

log "==== [2] GATE 5 qtype mới ===="
$PY -u wsd.py gate --samples output_qwen7b/processed/vqa_val.json --probs $W/wsd2_val_probs.json --types "$T5" | tee $W/wsd2_gate.txt
PASS=$($PY - <<'PYEOF'
import re
txt = open("/workspace/AICC/code/output_qwen35/wsd2_gate.txt").read()
ok = []
for m in re.finditer(r"^\s+(\w+)\s+:\s+[\d.]+% ->\s+[\d.]+%\s+\(Δ([+-][\d.]+),", txt, re.M):
    t, d = m.group(1), float(m.group(2))
    if t != "TOTAL" and d >= 1.0: ok.append(t)
print(",".join(ok))
PYEOF
)
log "types PASS (Δ>=+1.0): '$PASS'"
[ -n "$PASS" ] || { log "không type nào pass -> giữ nguyên. XONG WSD2 (no-op)"; exit 0; }

log "==== [3] TEST predict FULL + probs ===="
ok=0
for try in 1 2 3; do
  QWEN35_BATCH=4 QWEN35_SAVE_PROBS=$W/wsd2_test_probs.json $PY -u run_qwen35_vqa.py predict \
    --samples test_root_work_full/processed/vqa_val.json --out $W/wsd2_test_preds.json 2>&1 | tail -2
  [ $? -eq 0 ] && [ -s $W/wsd2_test_probs.json ] && { ok=1; break; }
  log "attempt $try fail"; sleep 30
done
[ $ok -eq 1 ] || { log "❌ test predict fail"; exit 1; }

log "==== [4] decode types PASS trên nền WSD -> SUBMIT_Q35_WSD2 ===="
# GUARD: đừng ghi đè bản đã nộp điểm cao (WSD2 = VQA 84.5717). Nếu đã có -> ghi vào dir mới có timestamp.
DEST=/workspace/AICC/SUBMIT_Q35_WSD2
if [ -s "$DEST/subtask2_vqa.json" ]; then
  DEST=/workspace/AICC/SUBMIT_Q35_WSD2_rerun_$(date +%Y%m%d_%H%M%S)
  log "⚠️ SUBMIT_Q35_WSD2 đã tồn tại (bản đã chấm) -> ghi vào $DEST thay vì đè"
fi
$PY -u wsd.py decode --samples test_root_work_full/processed/vqa_val.json --probs $W/wsd2_test_probs.json \
  --base-answers /workspace/AICC/SUBMIT_Q35_WSD/subtask2_vqa.json --out $W/subtask2_vqa_wsd2.json --types "$PASS"
mkdir -p "$DEST"
cp /workspace/AICC/SUBMIT_Q35_V4/subtask1_captioning.json "$DEST"/
cp $W/subtask2_vqa_wsd2.json "$DEST"/subtask2_vqa.json
log "SUBMIT_Q35_WSD2 sẵn sàng (caption V4 + VQA WSD+WSD2[$PASS])"
log "================ XONG WSD2 ================"
