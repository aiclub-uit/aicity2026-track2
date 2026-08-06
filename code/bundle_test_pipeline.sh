#!/bin/bash
set -o pipefail; CODE=/workspace/AICC/code; cd "$CODE"; PY=/venv/qwen35/bin/python
if [ "${_BT:-}" != "1" ]; then : > bundle_test.log; export _BT=1
  nohup setsid bash "$0" </dev/null >> bundle_test.log 2>&1 & disown
  echo "✅ bundle TEST launched. tail -f bundle_test.log"; exit 0; fi
log(){ echo "[$(date '+%F %T')] $*"; }
export HF_HOME=/workspace/hf_cache
export AICC26_QWEN35_QUANT=bf16
export AICC26_QWEN35_ADAPTER=$CODE/output_qwen35/vqa_lora_bundle
W=$CODE/output_qwen35

f=0; while [ $f -lt 2 ]; do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && f=$((f+1)) || f=0; sleep 20; done

log "==== [1] PREDICT test full (19.6k, bf16 batch 8, ~2.6h) ===="
ok=0
for try in 1 2 3; do
  QWEN35_BATCH=8 QWEN35_SAVE_PROBS=$W/bundle_test_probs.json $PY -u run_qwen35_vqa.py predict \
    --samples test_root_work_full/processed/vqa_val.json --out $W/bundle_test_preds.json 2>&1 | tail -1
  [ $? -eq 0 ] && [ -s $W/bundle_test_probs.json ] && { ok=1; break; }
  log "attempt $try fail"; sleep 30
done
[ $ok -eq 1 ] || { log "❌ predict fail"; exit 1; }

log "==== [2] MIX -> Viterbi -> update 5 qtype -> SUBMIT_Q35_BUNDLE ===="
$PY - <<'PYEOF' || { log "❌ build fail"; exit 1; }
import json, sys
sys.path.insert(0,'/workspace/AICC/code')
from wsd import _qt, _decode_all, LETTERS
SP={'orientation','position_ped','position_veh'}; VIT=SP|{'action'}
TGT={'direction','speed','attention','distance','action'}
samples=json.load(open('/workspace/AICC/code/test_root_work_full/processed/vqa_val.json'))
base=json.load(open('/workspace/AICC/code/output_qwen35/wsd2_test_probs.json'))
ctx=json.load(open('/workspace/AICC/code/output_qwen35/ctx_test_probs.json'))
bu=json.load(open('/workspace/AICC/code/output_qwen35/bundle_test_probs.json'))
EPS=1e-5
mixed={}
for s in samples:
    qid=s['id']; p=base.get(qid)
    if not p: continue
    t=_qt(s['question']) or 'other'
    f=p
    if t in SP and qid in ctx: f=ctx[qid]
    elif t in TGT and qid in bu: f=bu[qid]
    tt=sum(max(x,EPS) for x in f); mixed[qid]=[max(x,EPS)/tt for x in f]
v={s['id']:LETTERS[max(range(4),key=lambda i:mixed[s['id']][i])] for s in samples if s['id'] in mixed}
v.update(_decode_all(samples, mixed, VIT))
dep=json.load(open('/workspace/AICC/SUBMIT_Q35_CTX/subtask2_vqa.json'))
ans={x['id']:x['correct'] for x in dep}
nch=0
for s in samples:
    if (_qt(s['question']) or 'other') in TGT and s['id'] in v:
        if ans.get(s['id'])!=v[s['id']]: nch+=1
        ans[s['id']]=v[s['id']]
out=[{'id':k,'correct':val} for k,val in ans.items()]
json.dump(out, open('/workspace/AICC/code/output_qwen35/subtask2_vqa_bundle.json','w'))
print(f'update 5 qtype: {nch} câu ĐỔI so với deployed CTX')
PYEOF
DEST=/workspace/AICC/SUBMIT_Q35_BUNDLE; mkdir -p "$DEST"
cp /workspace/AICC/Q35_HIGHEST_CUR/subtask1_captioning-2.json "$DEST"/subtask1_captioning.json
cp $W/subtask2_vqa_bundle.json "$DEST"/subtask2_vqa.json
md5sum "$DEST"/* | awk '{print substr($1,1,12), $2}'
log "SUBMIT_Q35_BUNDLE sẵn sàng (caption 30.17 + VQA bundle-no-other)"
log "================ XONG BUNDLE-TEST ================"
