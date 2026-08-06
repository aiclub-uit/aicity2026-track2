#!/usr/bin/env python
from __future__ import annotations
import argparse, json, re, os
from pathlib import Path

MODEL_ID = os.environ.get("AICC26_QWEN35_MODEL", "Qwen/Qwen3.5-9B")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
WORK = Path("/workspace/AICC/code/output_qwen35")
LORA_OUT = Path(os.environ.get("AICC26_QWEN35_ADAPTER", str(WORK / "vqa_lora")))
DATA = Path(os.environ.get("AICC26_DATA_DIR", "/workspace/AICC/code/output_qwen7b/processed"))
OPTION_LABELS = ["a", "b", "c", "d"]
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "out_proj", "gate_proj", "up_proj", "down_proj"]
if os.environ.get("AICC26_LORA_REGEX"):
    LORA_TARGETS = os.environ["AICC26_LORA_REGEX"]
NUM_IMAGES = int(os.environ.get("QWEN35_NUM_IMAGES", "3"))


QUANT = os.environ.get("AICC26_QWEN35_QUANT", "8bit")


def _quant_kw(for_train):
    import torch
    from transformers import BitsAndBytesConfig
    dm = {"": 0} if for_train else "cuda"
    if QUANT == "bf16":
        return dict(dtype=torch.bfloat16, device_map=dm)
    if QUANT == "8bit":
        return dict(dtype=torch.bfloat16, device_map=dm,
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    return dict(dtype=torch.bfloat16, device_map=dm,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))


def _load(for_train=False):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    pkw = {}
    if os.environ.get("QWEN35_MAX_PIXELS"):
        pkw["max_pixels"] = int(os.environ["QWEN35_MAX_PIXELS"])
    if os.environ.get("QWEN35_MIN_PIXELS"):
        pkw["min_pixels"] = int(os.environ["QWEN35_MIN_PIXELS"])
    proc = AutoProcessor.from_pretrained(MODEL_ID, **pkw)
    if pkw:
        print(f"[q35] pixel-cap {pkw}")
    print(f"[q35] quant={QUANT}")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **_quant_kw(for_train))
    return proc, model


def _qtype(q):
    q = q.lower()
    for n, p in [('orientation', 'orientation of the'), ('position_ped', 'position of the pedestrian'),
                 ('position_veh', 'position of the vehicle')]:
        if re.search(p, q):
            return n
    return 'other'


def _imgs(sample):
    ims = [str(p) for p in sample["images"]]
    if os.environ.get("QWEN35_IMG_ORDER") == "local_first":
        l = [p for p in ims if "frames_local" in p]
        return (l + [p for p in ims if p not in l])[:NUM_IMAGES]
    if NUM_IMAGES >= len(ims):
        return ims
    g = [p for p in ims if "frames_global" in p]
    return (g + [p for p in ims if p not in g])[:NUM_IMAGES]


def _user_content(sample):
    c = [{"type": "image", "image": p} for p in _imgs(sample)]
    txt = sample["conversations"][0]["value"].replace("<image>", "").strip()
    txt += "\nAnswer with a single letter a, b, c, or d."
    return c + [{"type": "text", "text": txt}]


def _gt(sample):
    return (sample["conversations"][1]["value"].strip()[:1] or "a").lower()


def _letter_ids(proc):
    tok = proc.tokenizer
    ids = {}
    for L in OPTION_LABELS:
        for cand in (L, " " + L):
            e = tok.encode(cand, add_special_tokens=False)
            if len(e) == 1:
                ids[L] = e[-1]; break
        else:
            raise SystemExit(f"letter {L} không 1 token")
    return ids


def cmd_inspect():
    proc, model = _load(for_train=False)
    print("processor:", type(proc).__name__, "| model:", type(model).__name__)
    print("LETTER_IDS:", _letter_ids(proc))


class Collator:
    def __init__(self, proc):
        self.proc = proc

    def __call__(self, batch):
        import torch, copy
        from qwen_vl_utils import process_vision_info
        feats = []
        for ex in batch:
            uc = [{"type": "image", "image": p} for p in ex["images"]] + [{"type": "text", "text": ex["q"]}]
            um = [{"role": "user", "content": uc}]
            fm = um + [{"role": "assistant", "content": [{"type": "text", "text": ex["ans"]}]}]
            imgs, _ = process_vision_info(copy.deepcopy(um))
            ftext = self.proc.apply_chat_template(fm, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            ptext = self.proc.apply_chat_template(um, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            full = self.proc(text=[ftext], images=imgs, return_tensors="pt", padding=True)
            nprompt = self.proc(text=[ptext], images=imgs, return_tensors="pt", padding=True)["input_ids"].shape[1]
            labels = full["input_ids"].clone()
            labels[:, :nprompt] = -100
            full["labels"] = labels
            feats.append(full)
        return {k: (v if not hasattr(v, "shape") else v) for k, v in feats[0].items()}


def build_ds(split, limit=None):
    from datasets import Dataset
    raw = json.load(open(DATA / f"vqa_{split}.json"))
    rows = []
    for s in raw:
        txt = s["conversations"][0]["value"].replace("<image>", "").strip() + "\nAnswer with a single letter a, b, c, or d."
        rows.append({"images": _imgs(s), "q": txt, "ans": _gt(s)})
    if limit:
        rows = rows[:limit]
    print(f"[q35] {split}: {len(rows)} samples")
    return Dataset.from_list(rows)


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
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    ds = build_ds("train", limit)
    args = TrainingArguments(
        output_dir=str(LORA_OUT), per_device_train_batch_size=1, gradient_accumulation_steps=16,
        num_train_epochs=1.0, learning_rate=1e-4, warmup_ratio=0.05, bf16=True,
        logging_steps=10, save_strategy="no", report_to=[], remove_unused_columns=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    Trainer(model=model, args=args, train_dataset=ds, data_collator=Collator(proc)).train()
    LORA_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(LORA_OUT))
    print(f"[q35] saved adapter -> {LORA_OUT}")


def _predict_letters(proc, model, samples):
    import torch, copy
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    LID = _letter_ids(proc)
    BATCH = int(os.environ.get("QWEN35_BATCH", "1"))
    if BATCH > 1:
        proc.tokenizer.padding_side = "left"
    res = {}
    for i in tqdm(range(0, len(samples), BATCH), desc="q35-pred"):
        chunk = samples[i:i + BATCH]
        texts, all_imgs = [], []
        for s in chunk:
            um = [{"role": "user", "content": _user_content(s)}]
            ims, _ = process_vision_info(copy.deepcopy(um))
            texts.append(proc.apply_chat_template(um, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            all_imgs.extend(ims)
        inp = proc(text=texts, images=all_imgs, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            lg = model(**inp, return_dict=True).logits[:, -1, :]
        for j, s in enumerate(chunk):
            row = lg[j]
            ol = torch.tensor([row[LID[L]] for L in OPTION_LABELS]).float()
            pred = OPTION_LABELS[int(ol.argmax())]
            res[s.get("id", id(s))] = pred
            if os.environ.get("QWEN35_SAVE_PROBS"):
                _PROBS[s.get("id", id(s))] = [round(float(x), 5) for x in torch.softmax(ol, 0)]
    if os.environ.get("QWEN35_SAVE_PROBS"):
        json.dump(_PROBS, open(os.environ["QWEN35_SAVE_PROBS"], "w"))
        print(f"[q35] saved probs {len(_PROBS)} -> {os.environ['QWEN35_SAVE_PROBS']}")
    return res


_PROBS = {}


def cmd_eval(limit):
    import collections
    from peft import PeftModel
    proc, model = _load(for_train=False)
    if LORA_OUT.exists():
        model = PeftModel.from_pretrained(model, str(LORA_OUT)); print(f"[q35] adapter {LORA_OUT}")
    model.eval()
    raw = json.load(open(DATA / "vqa_val.json"))
    if limit:
        raw = raw[:limit]
    preds = _predict_letters(proc, model, raw)
    agg = collections.defaultdict(lambda: [0, 0])
    for s in raw:
        t = _qtype(s["question"]); agg[t][1] += 1; agg[t][0] += int(preds[s["id"]] == _gt(s))
    tot = [0, 0]
    print("=== Qwen3.5 VQA val per-qtype (Qwen2.5: orient 39 / pos_ped 67 / pos_veh 61 / TOTAL 81.1) ===")
    for t, (ok, n) in sorted(agg.items(), key=lambda x: -x[1][1]):
        print(f"  {t:<13}: {100*ok/n:.1f}% (n={n})"); tot[0] += ok; tot[1] += n
    print(f"  TOTAL: {100*tot[0]/tot[1]:.1f}% (n={tot[1]})")


def cmd_predict(samples_path, out_path, qtypes):
    from peft import PeftModel
    proc, model = _load(for_train=False)
    if LORA_OUT.exists():
        model = PeftModel.from_pretrained(model, str(LORA_OUT))
    model.eval()
    raw = json.load(open(samples_path))
    want = set(qtypes.split(",")) if qtypes else None
    sel = [s for s in raw if want is None or _qtype(s["question"]) in want]
    print(f"[q35] predict {len(sel)}/{len(raw)} (qtypes={qtypes})")
    preds = _predict_letters(proc, model, sel)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({s["id"]: preds[s["id"]] for s in sel}, open(out_path, "w"))
    print(f"[q35] saved {len(sel)} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect")
    t = sub.add_parser("train"); t.add_argument("--limit", type=int, default=0)
    e = sub.add_parser("eval"); e.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("predict"); p.add_argument("--samples", required=True); p.add_argument("--out", required=True)
    p.add_argument("--qtypes", default="")
    a = ap.parse_args()
    if a.cmd == "inspect": cmd_inspect()
    elif a.cmd == "train": cmd_train(a.limit or None)
    elif a.cmd == "eval": cmd_eval(a.limit or None)
    elif a.cmd == "predict": cmd_predict(a.samples, a.out, a.qtypes)


if __name__ == "__main__":
    main()
