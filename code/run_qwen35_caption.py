#!/usr/bin/env python
from __future__ import annotations
import argparse, json, os, sys, statistics, collections, re
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-9B"
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
WORK = Path("/workspace/AICC/code/output_qwen35")
LORA_OUT = Path(os.environ.get("AICC26_QWEN35_CAP_ADAPTER", str(WORK / "caption_lora")))
DATA = Path("/workspace/AICC/code/output_qwen7b_caption/processed")
STERHINT = os.environ.get("AICC26_STERHINT", "")
NUM_IMAGES = int(os.environ.get("QWEN35_NUM_IMAGES", "4"))
sys.path.insert(0, "/workspace/AICC/src/src")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "out_proj", "gate_proj", "up_proj", "down_proj"]


QUANT = os.environ.get("AICC26_QWEN35_QUANT", "8bit")


def _load(for_train=False):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    dm = {"": 0} if for_train else "cuda"
    print(f"[q35cap] quant={QUANT}")
    if QUANT == "bf16":
        kw = dict(dtype=torch.bfloat16, device_map=dm)
    elif QUANT == "8bit":
        kw = dict(dtype=torch.bfloat16, device_map=dm, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    else:
        kw = dict(dtype=torch.bfloat16, device_map=dm, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kw)
    return proc, model


def _imgs(sample):
    ims = [str(p) for p in sample["images"]]
    return ims[:NUM_IMAGES]


def _prompt(sample):
    return sample["conversations"][0]["value"].replace("<image>", "").strip()


def build_ds(split, limit=None):
    from datasets import Dataset
    raw = json.load(open(DATA / f"caption_{split}{STERHINT}.json"))
    wmap = {}
    if os.environ.get("AICC26_LOSS_WEIGHTS"):
        wmap = json.load(open(os.environ["AICC26_LOSS_WEIGHTS"]))
        print(f"[q35cap] weighted-loss ON ({len(wmap)} weights)")
    rows = [{"images": _imgs(s), "q": _prompt(s), "ans": s["conversations"][1]["value"].strip(),
             "w": float(wmap.get(str(s.get("id")), 1.0))} for s in raw]
    if limit:
        rows = rows[:limit]
    print(f"[q35cap] {split}: {len(rows)} samples")
    return Dataset.from_list(rows)


class Collator:
    def __init__(self, proc): self.proc = proc

    def __call__(self, batch):
        import copy
        from qwen_vl_utils import process_vision_info
        ex = batch[0]
        uc = [{"type": "image", "image": p} for p in ex["images"]] + [{"type": "text", "text": ex["q"]}]
        um = [{"role": "user", "content": uc}]
        fm = um + [{"role": "assistant", "content": [{"type": "text", "text": ex["ans"]}]}]
        imgs, _ = process_vision_info(copy.deepcopy(um))
        ftext = self.proc.apply_chat_template(fm, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        ptext = self.proc.apply_chat_template(um, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = self.proc(text=[ftext], images=imgs, return_tensors="pt", padding=True)
        nprompt = self.proc(text=[ptext], images=imgs, return_tensors="pt", padding=True)["input_ids"].shape[1]
        labels = full["input_ids"].clone(); labels[:, :nprompt] = -100
        full["labels"] = labels
        out = dict(full)
        if os.environ.get("AICC26_LOSS_WEIGHTS"):
            import torch
            out["w"] = torch.tensor(float(ex.get("w", 1.0)))
        return out


def cmd_train(limit):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments
    proc, model = _load(for_train=True)
    if QUANT in ("4bit", "8bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    model.print_trainable_parameters()
    ds = build_ds("train", limit)
    args = TrainingArguments(
        output_dir=str(LORA_OUT), per_device_train_batch_size=1, gradient_accumulation_steps=16,
        num_train_epochs=float(os.environ.get("CAPTION_EPOCHS", "3")), learning_rate=1e-4, warmup_ratio=0.05, bf16=True,
        logging_steps=10, save_strategy="no", report_to=[], remove_unused_columns=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    TrainerCls = Trainer
    if os.environ.get("AICC26_LOSS_WEIGHTS"):
        class WTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                w = inputs.pop("w", None)
                out = model(**inputs)
                loss = out.loss if w is None else out.loss * w.to(out.loss.device).mean()
                return (loss, out) if return_outputs else loss
        TrainerCls = WTrainer
    TrainerCls(model=model, args=args, train_dataset=ds, data_collator=Collator(proc)).train()
    LORA_OUT.mkdir(parents=True, exist_ok=True); model.save_pretrained(str(LORA_OUT))
    print(f"[q35cap] saved -> {LORA_OUT}")


def cmd_eval(limit):
    import torch, copy
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from eval.local_eval import evaluate_captions
    proc, model = _load(for_train=False)
    if os.environ.get("AICC26_EVAL_ADAPTER"):
        adapter = Path(os.environ["AICC26_EVAL_ADAPTER"])
    else:
        adapter = (WORK / "caption_dpo_lora") if (WORK / "caption_dpo_lora").exists() else LORA_OUT
    if os.environ.get("AICC26_CAP_PRED_ADAPTER"):
        adapter = Path(os.environ["AICC26_CAP_PRED_ADAPTER"])
    if adapter.exists():
        model = PeftModel.from_pretrained(model, str(adapter)); print(f"[q35cap] adapter {adapter}")
    model.eval()
    raw = json.load(open(DATA / f"caption_val{STERHINT}.json"))
    if limit:
        raw = raw[:limit]
    NCAND = int(os.environ.get("QWEN35_EVAL_NCAND", "5"))
    MAXNEW = int(os.environ.get("QWEN35_EVAL_MAXNEW", "256"))
    print(f"[q35cap] eval ncand={NCAND} max_new={MAXNEW}")
    from tqdm import tqdm
    gen = {}
    for s in tqdm(raw, desc="q35cap-gen"):
        uc = [{"type": "image", "image": p} for p in _imgs(s)] + [{"type": "text", "text": _prompt(s)}]
        um = [{"role": "user", "content": uc}]
        imgs, _ = process_vision_info(copy.deepcopy(um))
        text = proc.apply_chat_template(um, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = proc(text=[text], images=imgs, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            if NCAND <= 1:
                out = model.generate(**inp, max_new_tokens=MAXNEW, do_sample=False)
            else:
                out = model.generate(**inp, max_new_tokens=MAXNEW, min_new_tokens=40, do_sample=True,
                                     temperature=0.7, top_p=0.9, num_return_sequences=NCAND)
        cands = [c.strip() for c in proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
        pred = _mbr_select(cands)
        gen[s["id"]] = (s, pred)
    json.dump({sid: pred for sid, (s, pred) in gen.items()}, open(WORK / "caption_val_preds.json", "w"))
    print(f"[q35cap] saved preds -> {WORK / 'caption_val_preds.json'}")
    tg, tp = {}, {}
    for sid, (s, pred) in gen.items():
        key = (s["scenario"], str(s["phase_label"]))
        tg.setdefault(key, {})[s["target"]] = s["conversations"][1]["value"].strip()
        tp.setdefault(key, {})[s["target"]] = pred
    R, P = {}, {}
    for (sc, lab), g in tg.items():
        R.setdefault(sc, []).append({"labels": [lab], "caption_pedestrian": g.get("pedestrian", ""), "caption_vehicle": g.get("vehicle", "")})
        pp = tp[(sc, lab)]
        P.setdefault(sc, []).append({"labels": [lab], "caption_pedestrian": pp.get("pedestrian", ""), "caption_vehicle": pp.get("vehicle", "")})
    o = evaluate_captions(P, R)
    print(f"=== Qwen3.5 caption (fair): S1={o['S1']:.2f}  ped={o['caption_score']['pedestrian']:.2f}  veh={o['caption_score']['vehicle']:.2f}  (Qwen2.5 local_eval=26.68 | greedy=27.38) ===")


DPO_PAIRS = Path(os.environ.get("AICC26_DPO_PAIRS", str(WORK / "dpo_pairs.json")))
DPO_OUT = Path(os.environ.get("AICC26_DPO_OUT", str(WORK / "caption_dpo_lora")))
SFT_SRC = Path(os.environ.get("AICC26_SFT_SRC", str(WORK / "caption_lora")))
DPO_SUBSET = int(os.environ.get("QWEN35_DPO_SUBSET", "1000"))
DPO_NCAND = int(os.environ.get("QWEN35_DPO_NCAND", "4"))


def _sent_score(pred, ref):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    p, r = pred.split(), ref.split()
    if not p:
        return 0.0
    sm = SmoothingFunction().method1
    try: b = sentence_bleu([r], p, smoothing_function=sm)
    except Exception: b = 0.0
    try: m = meteor_score([r], p)
    except Exception: m = 0.0
    n, mm = len(p), len(r); dp = [0]*(mm+1)
    for i in range(1, n+1):
        prev = 0
        for j in range(1, mm+1):
            t = dp[j]; dp[j] = prev+1 if p[i-1] == r[j-1] else max(dp[j], dp[j-1]); prev = t
    l = dp[mm]; rg = 0.0 if not (l and n and mm) else 2*(l/n)*(l/mm)/((l/n)+(l/mm))
    return (b + m + rg) / 3.0


def _mbr_select(cands):
    cs = [c for c in cands if c.strip()] or cands
    if len(cs) == 1:
        return cs[0]
    best, best_u = cs[0], -1.0
    for i, ci in enumerate(cs):
        u = sum(_sent_score(ci, cj) for j, cj in enumerate(cs) if j != i)
        if u > best_u:
            best_u, best = u, ci
    return best


def _composite_scores(records):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    gts_d, res_d = {}, {}
    for i, r in enumerate(records):
        for k, c in enumerate(r["cands"]):
            cid = f"{i}_{k}"
            gts_d[cid] = [{"caption": r["gt"]}]
            res_d[cid] = [{"caption": c or " "}]
    tok = PTBTokenizer()
    gts_t, res_t = tok.tokenize(gts_d), tok.tokenize(res_d)
    keys = list(gts_t.keys())
    _, bl = Bleu(4).compute_score(gts_t, res_t)
    _, me = Meteor().compute_score(gts_t, res_t)
    _, ro = Rouge().compute_score(gts_t, res_t)
    _, ci = Cider().compute_score(gts_t, res_t)
    comp = {keys[n]: (float(bl[3][n]) + float(me[n]) + float(ro[n]) + 0.1 * float(ci[n])) / 4 for n in range(len(keys))}
    out = []
    for i, r in enumerate(records):
        out.append({c: comp.get(f"{i}_{k}", 0.0) for k, c in enumerate(r["cands"])})
    return out


def cmd_build_dpo(limit):
    import torch, copy, random
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    proc, model = _load(for_train=False)
    sft = SFT_SRC
    model = PeftModel.from_pretrained(model, str(sft)); model.eval()
    print(f"[dpo] SFT adapter: {sft} | reward=COMPOSITE(BLEU+MET+ROUGE+0.1CIDEr)")
    raw = json.load(open(DATA / f"caption_train{STERHINT}.json"))
    random.Random(43).shuffle(raw)
    raw = raw[:(limit or DPO_SUBSET)]
    records = []
    for s in tqdm(raw, desc="dpo-gen"):
        gt = s["conversations"][1]["value"].strip()
        uc = [{"type": "image", "image": p} for p in _imgs(s)] + [{"type": "text", "text": _prompt(s)}]
        um = [{"role": "user", "content": uc}]
        imgs, _ = process_vision_info(copy.deepcopy(um))
        text = proc.apply_chat_template(um, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = proc(text=[text], images=imgs, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=256, min_new_tokens=80, do_sample=True,
                                  temperature=1.0, top_p=0.95, num_return_sequences=DPO_NCAND)
        cands = list({c.strip() for c in proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True) if c.strip()})
        if len(cands) >= 2:
            records.append({"images": _imgs(s), "q": _prompt(s), "gt": gt, "cands": cands})
    print(f"[dpo] sinh xong {len(records)} records -> chấm composite (CPU)...")
    scores = _composite_scores(records)
    pairs = []
    for r, sc in zip(records, scores):
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        if ranked[0][1] - ranked[-1][1] < 0.01 or ranked[0][0] == ranked[-1][0]:
            continue
        pairs.append({"images": r["images"], "q": r["q"], "chosen": ranked[0][0], "rejected": ranked[-1][0]})
    DPO_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(pairs, open(DPO_PAIRS, "w"))
    print(f"[dpo] {len(pairs)} pairs -> {DPO_PAIRS}")


def cmd_train_dpo():
    import torch
    from PIL import Image
    from peft import PeftModel
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset, Sequence
    from datasets import Image as HFImage
    # TRL's per-token entropy is a logging-only metric, but materializing it costs
    # ~4.3 GB fp32 on this vocab and OOMs 32 GB cards — zero it out; the DPO loss
    # itself is untouched
    import trl.trainer.dpo_trainer as _dt
    _dt.entropy_from_logits = lambda logits, *a, **k: logits.new_zeros(logits.shape[:-1])
    proc, model = _load(for_train=True)
    model = PeftModel.from_pretrained(model, str(SFT_SRC), is_trainable=True)
    pairs = json.load(open(DPO_PAIRS))
    def to_row(p):
        return {
            "images": p["images"],
            "prompt": [{"role": "user", "content": [{"type": "image"} for _ in p["images"]] + [{"type": "text", "text": p["q"]}]}],
            "chosen": [{"role": "assistant", "content": [{"type": "text", "text": p["chosen"]}]}],
            "rejected": [{"role": "assistant", "content": [{"type": "text", "text": p["rejected"]}]}],
        }
    # image paths + lazy Image feature: eager PIL rows overflow arrow's 2 GB
    # binary-column offsets at this pair count (997 pairs / 3,004 images)
    ds = Dataset.from_list([to_row(p) for p in pairs]).cast_column("images", Sequence(HFImage()))
    cfg = DPOConfig(output_dir=str(DPO_OUT), per_device_train_batch_size=1, gradient_accumulation_steps=8,
                    num_train_epochs=1.0, learning_rate=5e-6,
                    beta=float(os.environ.get("AICC26_DPO_BETA", "0.1")), bf16=True, logging_steps=10,
                    save_strategy="no", report_to=[], remove_unused_columns=False, max_length=8192,
                    truncation_mode="keep_start",
                    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds, processing_class=proc).train()
    DPO_OUT.mkdir(parents=True, exist_ok=True); model.save_pretrained(str(DPO_OUT))
    print(f"[dpo] saved -> {DPO_OUT}")


def cmd_predict_submission(testjson, out, limit=None):
    import torch, copy
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    proc, model = _load(for_train=False)
    adapter = (WORK / "caption_dpo_lora") if (WORK / "caption_dpo_lora").exists() else LORA_OUT
    if os.environ.get("AICC26_CAP_PRED_ADAPTER"):
        adapter = Path(os.environ["AICC26_CAP_PRED_ADAPTER"])
    if adapter.exists():
        model = PeftModel.from_pretrained(model, str(adapter)); print(f"[q35cap] adapter {adapter}")
    model.eval()
    raw = json.load(open(testjson))
    if limit:
        raw = raw[:limit]
    MAXNEW = int(os.environ.get("QWEN35_EVAL_MAXNEW", "256"))
    BATCH = int(os.environ.get("QWEN35_BATCH", "8"))
    NCAND = int(os.environ.get("QWEN35_PRED_NCAND", "1"))
    proc.tokenizer.padding_side = "left"
    uniq = {}
    for s in raw:
        k = f"{s['scenario']}|{s['phase_label']}|{s['target']}"
        uniq.setdefault(k, s)
    items = sorted(uniq.items(), key=lambda kv: (len(_imgs(kv[1])), kv[1]["target"]))
    mode = "greedy" if NCAND <= 1 else f"MBR best-of-{NCAND} (utility=BLEU+MET+ROUGE)"
    print(f"[q35cap] predict-test unique={len(items)} (raw {len(raw)}) {mode} max_new={MAXNEW} batch={BATCH}")
    from tqdm import tqdm
    preds = {}
    for i in tqdm(range(0, len(items), BATCH), desc="q35cap-test"):
        chunk = items[i:i + BATCH]
        texts, all_imgs = [], []
        for k, s in chunk:
            uc = [{"type": "image", "image": p} for p in _imgs(s)] + [{"type": "text", "text": _prompt(s)}]
            um = [{"role": "user", "content": uc}]
            ims, _ = process_vision_info(copy.deepcopy(um))
            texts.append(proc.apply_chat_template(um, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            all_imgs.extend(ims)
        inp = proc(text=texts, images=all_imgs, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            if NCAND <= 1:
                o = model.generate(**inp, max_new_tokens=MAXNEW, do_sample=False)
            else:
                o = model.generate(**inp, max_new_tokens=MAXNEW, min_new_tokens=40, do_sample=True,
                                   temperature=0.7, top_p=0.9, num_return_sequences=NCAND)
        outs = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        for gi, (k, s) in enumerate(chunk):
            if NCAND <= 1:
                preds[k] = outs[gi].strip()
            else:
                cands = [outs[gi * NCAND + j].strip() for j in range(NCAND)]
                preds[k] = _mbr_select(cands)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(preds, open(out, "w"))
    print(f"[q35cap] saved {len(preds)} test preds -> {out}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--limit", type=int, default=0)
    e = sub.add_parser("eval"); e.add_argument("--limit", type=int, default=0)
    b = sub.add_parser("build_dpo"); b.add_argument("--limit", type=int, default=0)
    sub.add_parser("train_dpo")
    p = sub.add_parser("predict_submission"); p.add_argument("--testjson", required=True)
    p.add_argument("--out", required=True); p.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "train": cmd_train(a.limit or None)
    elif a.cmd == "eval": cmd_eval(a.limit or None)
    elif a.cmd == "build_dpo": cmd_build_dpo(a.limit or None)
    elif a.cmd == "train_dpo": cmd_train_dpo()
    elif a.cmd == "predict_submission": cmd_predict_submission(a.testjson, a.out, a.limit or None)


if __name__ == "__main__":
    main()
