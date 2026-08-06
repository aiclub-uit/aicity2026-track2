#!/usr/bin/env python
from __future__ import annotations
import argparse, json, re, collections, shutil, sys
from pathlib import Path

CODE = Path("/workspace/AICC/code")
WORK = CODE / "output_qwen35"
sys.path.insert(0, str(CODE))
from fact_stitch import stitch as stitch_v2, build_oracle as oracle_v2, _opt_text
from fact_stitch3 import stitch_v3, build_oracle_v3, ALL_SLOTS

V3_SLOTS = {"color_upper", "color_lower", "lower_item", "brightness", "surface_cond", "surface_type"}
QT = {"orientation": r"orientation of", "position_ped": r"position of the pedestrian",
      "position_veh": r"position of the vehicle"}

FIND = {
    "position_ped": r"diagonally to the (?:left|right),? in front of the vehicle|directly in front of the vehicle|on the (?:left|right),? behind the vehicle|on the (?:left|right) of the vehicle|behind the vehicle",
    "position_veh": r"diagonally to the (?:left|right),? in front of the pedestrian|behind to the (?:left|right) of the pedestrian|(?:on the )?(?:left|right) side of the pedestrian|in front of the pedestrian|behind the pedestrian",
    "orientation":  r"perpendicular to the vehicle(?:,? and)? to the (?:left|right)|(?:diagonally to the (?:left|right),? )?in the (?:same|opposite) direction (?:as|to|of) the vehicle|perpendicular to the vehicle",
}


def _canon(t):
    t = re.sub(r"[^a-z ]", " ", str(t).lower())
    t = re.sub(r"\b(on|in|the|a|and|to|as|of)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _repl_text(qtype, opt):
    o = str(opt).strip().rstrip(".")
    ol = o.lower()
    if qtype == "orientation":
        if ol.startswith("same direction") or ol.startswith("opposite direction"):
            return "in the " + ol.replace("opposite direction to", "opposite direction to").replace("same direction as", "same direction as")
        return ol
    if qtype == "position_veh" and ol.endswith("side of the pedestrian") and not ol.startswith("on the"):
        return "on the " + ol
    return ol


def build_spatial_oracle(vqa_samples_path, answers):
    samples = json.load(open(vqa_samples_path))
    out = collections.defaultdict(dict)
    for s in samples:
        ql = s["question"].lower()
        qtype = next((n for n, p in QT.items() if re.search(p, ql)), None)
        if not qtype: continue
        L = answers.get(s["id"])
        if not L: continue
        out[(s["scenario"], str(s["phase_label"]))].setdefault(qtype, _opt_text(s, L))
    return out


def stitch_v4(text, sp, target):
    n = 0
    todo = [("position_ped", "pedestrian"), ("orientation", "pedestrian"), ("position_veh", "vehicle")]
    for qtype, tgt in todo:
        if tgt != target or qtype not in sp: continue
        repl = _repl_text(qtype, sp[qtype])
        m = re.search(FIND[qtype], text, re.I)
        if not m: continue
        if _canon(m.group(0)) == _canon(repl): continue
        text = text[:m.start()] + repl + text[m.end():]
        n += 1
    return text, n


def _apply_all(text, o2, o3, sp, target, use_v4):
    t, n1 = stitch_v2(text, o2, target)
    t, n2 = stitch_v3(t, o3, target, V3_SLOTS)
    n3 = 0
    if use_v4:
        t, n3 = stitch_v4(t, sp, target)
    return t, n1 + n2 + n3


def cmd_val():
    from harmonize_attrs import _score
    ans = json.load(open(WORK / "vqa_val_preds_wsd.json"))
    src = CODE / "output_qwen7b/processed/vqa_val.json"
    o2, o3 = oracle_v2(src, ans), build_oracle_v3(src, ans)
    sp = build_spatial_oracle(src, ans)
    preds = json.load(open(WORK / "caption_val_preds.json"))
    meta = {s["id"]: s for s in json.load(open(CODE / "output_qwen7b_caption/processed/caption_val.json"))}
    gens = {}
    for use_v4 in (False, True):
        g = {}
        nf = 0
        for sid, p in preds.items():
            s = meta.get(sid)
            if not s: continue
            key = (s["scenario"], str(s["phase_label"]))
            t, n = _apply_all(p, o2.get(s["scenario"], {}), o3.get(s["scenario"], {}),
                              sp.get(key, {}), s["target"], use_v4)
            g[sid] = (s, t); nf += n
        gens[use_v4] = (g, nf)
    (gb, nb), (ga, na) = gens[False], gens[True]
    print(f"[v4] fixes: v2+v3={nb} -> +v4={na} (v4 thêm {na-nb})")
    ob, oa = _score(gb), _score(ga)
    for k in ("BLEU-4", "METEOR", "ROUGE-L", "CIDEr"):
        print(f"  {k:<8}: {ob['mean_metrics'][k]:.2f} -> {oa['mean_metrics'][k]:.2f}  (Δ{oa['mean_metrics'][k]-ob['mean_metrics'][k]:+.2f})")
    print(f"  S1      : {ob['S1']:.2f} -> {oa['S1']:.2f}  (Δ{oa['S1']-ob['S1']:+.3f})")
    print(f"V4_DELTA={oa['S1']-ob['S1']:+.4f}")


def cmd_test(base, out, answers_path):
    raw = json.load(open(answers_path))
    ans = {x["id"]: x["correct"] for x in raw} if isinstance(raw, list) else dict(raw)
    src = CODE / "test_root_work_full/processed/vqa_val.json"
    o2, o3 = oracle_v2(src, ans), build_oracle_v3(src, ans)
    sp = build_spatial_oracle(src, ans)
    cap = json.load(open(Path(base) / "subtask1_captioning.json"))
    nfix = 0
    for scn, entries in cap.items():
        for e in entries:
            lab = str((e.get("labels") or [""])[0])
            for ch, tgt in (("caption_pedestrian", "pedestrian"), ("caption_vehicle", "vehicle")):
                if e.get(ch):
                    e[ch], n = _apply_all(e[ch], o2.get(scn, {}), o3.get(scn, {}),
                                          sp.get((scn, lab), {}), tgt, True)
                    nfix += n
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    json.dump(cap, open(outp / "subtask1_captioning.json", "w"))
    shutil.copy(answers_path, outp / "subtask2_vqa.json")
    print(f"[v4] test: {nfix} fixes (v2+v3+v4) -> {outp}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("val")
    t = sub.add_parser("test"); t.add_argument("--base", required=True); t.add_argument("--out", required=True)
    t.add_argument("--answers", required=True)
    a = ap.parse_args()
    if a.cmd == "val": cmd_val()
    else: cmd_test(a.base, a.out, a.answers)


if __name__ == "__main__":
    main()
