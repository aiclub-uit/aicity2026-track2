#!/usr/bin/env python
from __future__ import annotations
import argparse, json, re, collections, shutil, sys
from pathlib import Path

CODE = Path("/workspace/AICC/code")
WORK = CODE / "output_qwen35"
sys.path.insert(0, str(CODE))

Q_PATTERNS = [("age", r"\bage\b"), ("clothing", r"wearing on upper body"),
              ("weather", r"\bweather\b"), ("height", r"\bheight\b")]

CLOTH_NOUNS = r"t-?shirt|shirt|jacket|coat|hoodie|sweater|parka|blouse"
WEATHER_MAP = {"clear": "clear", "cloudy": "cloudy", "rain": "rainy", "rainy": "rainy", "snow": "snowy"}


def _opt_text(sample, letter):
    opts = sample["options"]
    if isinstance(opts, dict):
        return str(opts.get(letter, ""))
    idx = "abcd".find(letter)
    return str(opts[idx]) if 0 <= idx < len(opts) else ""


def _canon(qtype, opt_text):
    t = opt_text.lower().strip()
    if qtype == "age":
        m = re.search(r"(\d0)s", t); return m.group(1) if m else None
    if qtype == "height":
        m = re.search(r"(\d{2,3})\s*cm", t); return m.group(1) if m else None
    if qtype == "weather":
        for k, v in WEATHER_MAP.items():
            if k in t: return v
        return None
    if qtype == "clothing":
        m = re.search(CLOTH_NOUNS, t)
        if not m: return None
        item = m.group(0)
        return "T-shirt" if item.replace("-", "").lower() == "tshirt" else item.lower()
    return None


def build_oracle(vqa_samples_path, answers):
    samples = json.load(open(vqa_samples_path))
    votes = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for s in samples:
        ql = s["question"].lower()
        qtype = next((n for n, p in Q_PATTERNS if re.search(p, ql)), None)
        if not qtype: continue
        letter = answers.get(s["id"])
        if not letter: continue
        v = _canon(qtype, _opt_text(s, letter))
        if v: votes[s["scenario"]][qtype][v] += 1
    oracle = {}
    for scn, d in votes.items():
        o = {}
        for qtype, cnt in d.items():
            top = cnt.most_common(2)
            if len(top) == 1 or top[0][1] > top[1][1]:
                o[qtype] = top[0][0]
        oracle[scn] = o
    return oracle


def stitch(text, o, target):
    n = 0
    if target == "pedestrian":
        if o.get("age"):
            new = re.sub(r"(in (?:his|her|their) )\d0s", lambda m: m.group(1) + o["age"] + "s", text)
            if new != text: n += 1; text = new
        if o.get("height"):
            new = re.sub(r"\b\d{2,3}(\s*cm\b)", lambda m: o["height"] + m.group(1), text)
            if new != text: n += 1; text = new
        if o.get("clothing"):
            new = re.sub(rf"\b({CLOTH_NOUNS})\b", o["clothing"], text, count=1, flags=re.I)
            if new != text: n += 1; text = new
    if o.get("weather"):
        out_sents = []
        changed = False
        for sent in re.split(r"(?<=[.;])", text):
            if re.search(r"\bweather|\bday\b|\bsky\b|environment condition|\bbright", sent, re.I):
                new = re.sub(r"\b(cloudy|sunny|rainy|overcast|clear|snowy)\b", o["weather"], sent, flags=re.I)
                if new != sent: changed = True; sent = new
            out_sents.append(sent)
        if changed: n += 1; text = "".join(out_sents)
    return text, n


def cmd_val():
    from harmonize_attrs import _score
    answers = json.load(open(WORK / "vqa_val_preds_full.json"))
    oracle = build_oracle(CODE / "output_qwen7b/processed/vqa_val.json", answers)
    print(f"[stitch] oracle val: {len(oracle)} scenarios; ví dụ:", dict(list(oracle.items())[:1]))
    preds = json.load(open(WORK / "caption_val_preds.json"))
    meta = {s["id"]: s for s in json.load(open(CODE / "output_qwen7b_caption/processed/caption_val.json"))}
    gen = {sid: (meta[sid], p) for sid, p in preds.items() if sid in meta}
    fixed, nfix = {}, 0
    for sid, (s, p) in gen.items():
        t, n = stitch(p, oracle.get(s["scenario"], {}), s["target"])
        fixed[sid] = (s, t); nfix += n
    print(f"[stitch] val: {nfix} thay đổi trên {len(gen)} captions")
    labels = json.load(open(WORK / "fact_labels_val.json"))
    for tag, g in (("BEFORE", gen), ("AFTER ", fixed)):
        ok = tot = 0
        for sid, (s, p) in g.items():
            if s["target"] != "pedestrian": continue
            gt = labels.get(s["scenario"], {}).get("age")
            m = re.search(r"in (?:his|her|their) (\d0)s", p.lower())
            if gt and m: tot += 1; ok += (m.group(1) == gt)
        print(f"[gate-attr] age {tag}: {100*ok/max(1,tot):.1f}% (n={tot})")
    print("\n=== BEFORE (per-metric) ==="); print(json.dumps(_score(gen)["mean_metrics"], indent=1))
    ob = _score(gen); oa = _score(fixed)
    print("=== AFTER (per-metric) ===");  print(json.dumps(oa["mean_metrics"], indent=1))
    print(f"\n>>> S1: {ob['S1']:.2f} -> {oa['S1']:.2f}  (Δ={oa['S1']-ob['S1']:+.3f})")


def cmd_test(base, out):
    sub_ans = json.load(open(Path(base) / "subtask2_vqa.json"))
    answers = {x["id"]: x["correct"] for x in sub_ans}
    oracle = build_oracle(CODE / "test_root_work_full/processed/vqa_val.json", answers)
    print(f"[stitch] oracle test: {len(oracle)} scenarios")
    cap = json.load(open(Path(base) / "subtask1_captioning.json"))
    nfix = 0; per = collections.Counter()
    for scn, entries in cap.items():
        o = oracle.get(scn, {})
        if not o: per["no_oracle"] += 1; continue
        for e in entries:
            for ch, tgt in (("caption_pedestrian", "pedestrian"), ("caption_vehicle", "vehicle")):
                if e.get(ch):
                    e[ch], n = stitch(e[ch], o, tgt); nfix += n
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    json.dump(cap, open(outp / "subtask1_captioning.json", "w"))
    shutil.copy(Path(base) / "subtask2_vqa.json", outp / "subtask2_vqa.json")
    print(f"[stitch] test: {nfix} thay đổi | scenario thiếu oracle: {per['no_oracle']} -> {outp}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("val")
    t = sub.add_parser("test"); t.add_argument("--base", required=True); t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "val": cmd_val()
    else: cmd_test(a.base, a.out)


if __name__ == "__main__":
    main()
