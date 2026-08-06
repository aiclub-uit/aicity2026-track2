#!/bin/bash
set -o pipefail; CODE=/workspace/AICC/code; cd "$CODE"; PY=/venv/qwen35/bin/python
CPY=/workspace/cosmos25/.venv/bin/python
if [ "${_CN:-}" != "1" ]; then : > cosmos_night.log; export _CN=1
  nohup setsid bash "$0" </dev/null >> cosmos_night.log 2>&1 & disown
  echo "✅ cosmos night chain launched. tail -f cosmos_night.log"; exit 0; fi
log(){ echo "[$(date '+%F %T')] $*"; }
export HF_HOME=/workspace/hf_cache PATH="$HOME/.local/bin:$PATH"
W=$CODE/output_qwen35; CW=$CODE/cosmos_work

f=0; while [ $f -lt 2 ]; do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && f=$((f+1)) || f=0; sleep 20; done

log "==== [1] RESTYLE 1447 frames (distilled edge, ~4h) ===="
ls $CW/specs/*.json > $CW/spec_list.txt
$CPY /workspace/cosmos25/examples/inference.py \
  -i $(tr '\n' ' ' < $CW/spec_list.txt) \
  -o $CW/restyled --disable-guardrails 2>&1 | grep -aE "SUCCESS|Error|Traceback" | tail -20
NR=$(ls $CW/restyled/*.jpg 2>/dev/null | grep -avc control_edge)
log "restyled: $NR/1447"
[ "$NR" -ge 1300 ] || { log "❌ restyle quá ít ($NR) -> dừng"; exit 1; }

log "==== [2] REBUILD processed_cosmos (bbox + local re-crop) ===="
AICC26_DATA_ROOT=/workspace/AICC/synwts/data $PY cosmos_restyle_prep.py rebuild 2>&1 | tail -3 || { log "❌ rebuild fail"; exit 1; }

log "==== [3] BUNDLE: DR đè lên cosmos tree ===="
$PY - <<'PYEOF' || { log "❌ bundle build fail"; exit 1; }
import json, os, sys, hashlib, random
sys.path.insert(0,'/workspace/AICC/code')
from multiprocessing import Pool
from dr_prep import augment, _rng
from PIL import Image
SRC_JSON='/workspace/AICC/code/output_qwen7b/processed_cosmos/vqa_train.json'
DR_ROOT='/workspace/AICC/code/output_qwen7b/frames_dr_bundle'
OUT_DIR='/workspace/AICC/code/output_qwen35/processed_bundle'
d=json.load(open(SRC_JSON))
paths=sorted({p for s in d for p in s['images']})
def drp(p):
    key=hashlib.md5(p.encode()).hexdigest()[:16]
    return f'{DR_ROOT}/{key}.jpg'
def work(p):
    out=drp(p)
    if os.path.exists(out): return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im=Image.open(p).convert('RGB')
    augment(im,_rng(p)).save(out,'JPEG',quality=92)
    return 1
with Pool(16) as pool:
    made=sum(pool.map(work, paths, chunksize=32))
print(f'DR variants: {made} mới / {len(paths)} paths')
na=nt=0
for s in d:
    imgs=[]
    for p in s['images']:
        nt+=1
        if _rng('bundle:'+s['id']+p).random()<0.6:
            imgs.append(drp(p)); na+=1
        else:
            imgs.append(p)
    s['images']=imgs
os.makedirs(OUT_DIR, exist_ok=True)
json.dump(d, open(f'{OUT_DIR}/vqa_train.json','w'))
print(f'bundle train json: {na}/{nt} ảnh DR ({100*na/nt:.0f}%)')
PYEOF

log "==== [4] TRAIN bundle (bf16, ~6h) ===="
export AICC26_QWEN35_QUANT=bf16
export AICC26_DATA_DIR=$W/processed_bundle
export AICC26_QWEN35_ADAPTER=$W/vqa_lora_bundle
$PY -u run_qwen35_vqa.py train 2>&1 | grep -aE "'loss'|trainable|samples|saved|Error" | tail -12
[ -s "$AICC26_QWEN35_ADAPTER/adapter_model.safetensors" ] || { log "❌ train fail"; exit 1; }

log "==== [5] PREDICT val + GATE not-degrade ===="
ok=0
for try in 1 2 3; do
  QWEN35_BATCH=8 QWEN35_SAVE_PROBS=$W/bundle_val_probs.json $PY -u run_qwen35_vqa.py predict \
    --samples output_qwen7b/processed/vqa_val.json --out $W/bundle_val_preds.json 2>&1 | tail -1
  [ $? -eq 0 ] && [ -s $W/bundle_val_probs.json ] && { ok=1; break; }
  log "attempt $try fail"; sleep 30
done
[ $ok -eq 1 ] || { log "❌ predict fail"; exit 1; }
$PY - <<'PYEOF' 2>&1 | tee $W/bundle_gate.txt
import json, sys
sys.path.insert(0,'/workspace/AICC/code')
from wsd import _qt, _gt_letter, _decode_all, LETTERS
SP={'orientation','position_ped','position_veh'}
VIT=SP|{'action'}
QALL=['orientation','position_ped','position_veh','direction','speed','attention','distance','action','other']
samples=json.load(open('/workspace/AICC/code/output_qwen7b/processed/vqa_val.json'))
b8=json.load(open('/workspace/AICC/code/output_qwen35/wsd2_val_probs.json'))
c8=json.load(open('/workspace/AICC/code/output_qwen35/ctx_val_probs_ctx.json'))
bu=json.load(open('/workspace/AICC/code/output_qwen35/bundle_val_probs.json'))
EPS=1e-5
def mix(bd, use_ctx=True):
    out={}
    for s in samples:
        qid=s['id']; p=bd.get(qid)
        if not p: continue
        t=_qt(s['question']) or 'other'
        f=c8[qid] if (use_ctx and t in SP and qid in c8) else p
        tt=sum(max(x,EPS) for x in f); out[qid]=[max(x,EPS)/tt for x in f]
    return out
def score(probs):
    v={s['id']:LETTERS[max(range(4),key=lambda i:probs[s['id']][i])] for s in samples if s['id'] in probs}
    v.update(_decode_all(samples,probs,VIT))
    def acc(qt=None,ss=None):
        c=n=0
        for s in samples:
            if s['id'] not in v: continue
            t=_qt(s['question']) or 'other'
            if qt and t!=qt: continue
            if ss and s['scenario'] not in ss: continue
            c+=(v[s['id']]==_gt_letter(s)); n+=1
        return (100*c/n if n else 0), n
    return acc
scens=sorted({s['scenario'] for s in samples}); half=set(scens[::2])
accb=score(mix(b8)); b_tot=accb()[0]
accu=score(mix(bu)); accu2=score(mix(bu,use_ctx=False))
print(f"deploy: TOTAL {b_tot:.2f}")
for name,ac in (('BUNDLE+ctx',accu),('BUNDLE-pure',accu2)):
    t=ac()[0]; dA=ac(ss=half)[0]-accb(ss=half)[0]; dB=ac(ss=set(scens)-half)[0]-accb(ss=set(scens)-half)[0]
    print(f"{name:<12}: {t:.2f} (Δ{t-b_tot:+.2f} | A {dA:+.2f} B {dB:+.2f})")
print(f"\n{'qtype':<14} {'deploy':>7} {'BUNDLE':>7} {'Δ':>6} {'n':>5}")
worst=0
for q in QALL:
    bq,n=accb(qt=q); nq,_=accu(qt=q)
    worst=min(worst,nq-bq)
    print(f"{q:<14} {bq:>7.1f} {nq:>7.1f} {nq-bq:>+6.1f} {n:>5}")
t=accu()[0]
verdict="PASS not-degrade" if (t-b_tot>=-0.3 and worst>=-1.5) else "FAIL degrade"
print(f"\nVERDICT: {verdict} (Δtotal {t-b_tot:+.2f}, worst qtype {worst:+.1f})")
PYEOF
log "================ XONG COSMOS NIGHT CHAIN ================"
