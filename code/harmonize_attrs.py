#!/usr/bin/env python
"""harmonize_attrs.py — Cross-phase attribute harmonization (age/height/weather).

Cơ chế: trong 1 scenario, pedestrian là CÙNG một người + weather là CÙNG một scene
-> majority-vote giá trị qua các phase (voters DE-correlated: cùng người, khác frame),
rewrite caption lệch về giá trị majority. Phục hồi fact distinctive bị MBR sampling lật
(audit: age-inconsistent 70% scenario trên test MBR vs 0% trong GT) -> CIDEr/BLEU.
Text-only, không GPU. "clear" bị LOẠI khỏi weather vocab (đụng "clear line of sight").

CLI:
  val                          # áp lên output_qwen35/caption_val_preds.json (MBR-300) -> score before/after
  test --in DIR --out DIR      # rewrite subtask1_captioning.json -> submission mới (+copy subtask2_vqa.json)
"""
from __future__ import annotations
import argparse, json, re, collections, shutil, sys
from pathlib import Path

CODE = Path("/workspace/AICC/code")
WORK = CODE / "output_qwen35"
DATA = CODE / "output_qwen7b_caption/processed"
sys.path.insert(0, "/workspace/AICC/src/src")

WEATHER_VOCAB = r"cloudy|sunny|rainy|overcast"  # KHÔNG có 'clear' (mơ hồ với clear view/line of sight)


def get_attrs(t: str):
    tl = t.lower()
    a = {}
    m = re.search(r"in (?:his|her|their) (\d0)s", tl);       a["age"] = m.group(1) if m else None
    m = re.search(r"\b(\d{2,3})\s*cm\b", tl);                a["height"] = m.group(1) if m else None
    m = re.search(rf"\b({WEATHER_VOCAB})\b", tl);            a["weather"] = m.group(1) if m else None
    return a


def majority(captions):
    """majority mỗi attr qua mọi caption (ped+veh) của scenario; tie -> None (không sửa)."""
    votes = collections.defaultdict(collections.Counter)
    for c in captions:
        for k, v in get_attrs(c).items():
            if v:
                votes[k][v] += 1
    maj = {}
    for k, cnt in votes.items():
        top = cnt.most_common(2)
        if len(top) == 1 or top[0][1] > top[1][1]:
            maj[k] = top[0][0]
    return maj


def rewrite(t: str, maj: dict):
    a = get_attrs(t)
    n = 0
    if maj.get("age") and a["age"] and a["age"] != maj["age"]:
        t = re.sub(r"(in (?:his|her|their) )\d0s", lambda m: m.group(1) + maj["age"] + "s", t, flags=re.I); n += 1
    if maj.get("height") and a["height"] and a["height"] != maj["height"]:
        t = re.sub(r"\b\d{2,3}(\s*cm\b)", lambda m: maj["height"] + m.group(1), t, flags=re.I); n += 1
    if maj.get("weather") and a["weather"] and a["weather"] != maj["weather"]:
        t = re.sub(rf"\b(?:{WEATHER_VOCAB})\b", maj["weather"], t, flags=re.I); n += 1
    return t, n


def _score(gen):
    """gen: {sid: (sample, pred)} -> evaluate_captions (y hệt cmd_eval của run_qwen35_caption)."""
    from eval.local_eval import evaluate_captions
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
    return evaluate_captions(P, R)


def cmd_val():
    preds = json.load(open(WORK / "caption_val_preds.json"))
    meta = {s["id"]: s for s in json.load(open(DATA / "caption_val.json"))}
    gen = {sid: (meta[sid], p) for sid, p in preds.items() if sid in meta}
    print(f"[harm] val preds: {len(gen)} samples")
    # majority per scenario (gộp ped+veh)
    by_scn = collections.defaultdict(list)
    for sid, (s, p) in gen.items():
        by_scn[s["scenario"]].append(p)
    maj = {scn: majority(cs) for scn, cs in by_scn.items()}
    fixed, nfix = {}, 0
    for sid, (s, p) in gen.items():
        t, n = rewrite(p, maj[s["scenario"]])
        fixed[sid] = (s, t); nfix += n
    print(f"[harm] rewrites: {nfix} attribute-fix trên {len(gen)} captions")
    print("\n=== BEFORE (MBR-300 nguyên bản) ==="); ob = _score(gen)
    print(json.dumps(ob, indent=1))
    print("\n=== AFTER (harmonized) ==="); oa = _score(fixed)
    print(json.dumps(oa, indent=1))
    print(f"\n>>> ΔS1 = {oa['S1'] - ob['S1']:+.3f}  (before {ob['S1']:.2f} -> after {oa['S1']:.2f})")


def cmd_test(in_dir, out_dir):
    inp, outp = Path(in_dir), Path(out_dir)
    sub = json.load(open(inp / "subtask1_captioning.json"))
    tot_fix = 0
    for scn, entries in sub.items():
        caps = [e.get("caption_pedestrian", "") or "" for e in entries] + \
               [e.get("caption_vehicle", "") or "" for e in entries]
        maj = majority([c for c in caps if c.strip()])
        for e in entries:
            for ch in ("caption_pedestrian", "caption_vehicle"):
                if e.get(ch):
                    e[ch], n = rewrite(e[ch], maj); tot_fix += n
    outp.mkdir(parents=True, exist_ok=True)
    json.dump(sub, open(outp / "subtask1_captioning.json", "w"))
    shutil.copy(inp / "subtask2_vqa.json", outp / "subtask2_vqa.json")
    print(f"[harm] test: {tot_fix} attribute-fix -> {outp}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("val")
    t = sub.add_parser("test"); t.add_argument("--in", dest="inp", required=True); t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "val": cmd_val()
    else: cmd_test(a.inp, a.out)


if __name__ == "__main__":
    main()
