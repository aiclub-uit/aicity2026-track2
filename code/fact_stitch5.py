#!/usr/bin/env python
"""fact_stitch5.py — Fact-Stitch v5: ACTION + ATTENTION pillar per-phase.
Oracle val (wsd2 preds): action_veh 97.4% / sight 85.7% / awareness 84.3% / action_ped 71.6%.
Corrective-only như v4, chồng lên stack v2+v3+v4. Subject-guard cho action (phrase
"going straight ahead" trong ped-caption 50/50 là ped hay vehicle -> bắt buộc look-back chủ ngữ).

CLI:
  val                       # gate: (v2+v3+v4) vs +v5A (veh_action,awareness,sight) vs +v5AB (+ped_action)
  test --base DIR --out DIR --answers SUBTASK2.json --subs action_veh,awareness,sight[,action_ped]
"""
from __future__ import annotations
import argparse, json, re, collections, shutil, sys
from pathlib import Path

CODE = Path("/workspace/AICC/code")
WORK = CODE / "output_qwen35"
sys.path.insert(0, str(CODE))
from fact_stitch import _opt_text

SUB_Q = {  # subtype -> regex trên câu hỏi
    "action_ped": r"pedestrian's action|fine-grained action taken by the pedestrian",
    "action_veh": r"action taken by vehicle",
    "awareness":  r"awareness regarding",
    "sight":      r"line of sight",
}
SKIP_OPTS = {"cannot be determined", "collision", "unknown"}
V5A = ("action_veh", "awareness", "sight")

AW_PAT = re.compile(
    r"(?:(?:is|was|were)\s+)?(?:almost noticed|(?:un)?aware of the (?:approaching )?vehicle|"
    r"notic(?:es|ed|ing) the (?:approaching )?vehicle)", re.I)
SIGHT_PAT = re.compile(r"line of sight (is|was)\s+[^.,;]*", re.I)
SIGHT_OBJ = {  # option -> phrase (directional = verbatim)
    "vehicle": "focused on the vehicle", "road surface": "focused on the road surface",
    "crossing destination": "focused on the crossing destination",
    "parked vehicle": "focused on the parked vehicle", "passing vehicle": "focused on the passing vehicle",
    "oncoming vehicle": "focused on the oncoming vehicle",
    "smartphone in hand": "focused on the smartphone in hand",
}


def _canon(t):
    t = re.sub(r"[^a-z ]", " ", str(t).lower())
    t = re.sub(r"\b(on|in|the|a|and|to|as|of|is|was|were)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _action_vocab():
    """Raw option texts của action questions (train) -> alternation longest-first."""
    vocab = set()
    for s in json.load(open(CODE / "output_qwen7b/processed/vqa_train.json")):
        ql = s["question"].lower()
        if not re.search(SUB_Q["action_ped"] + "|" + SUB_Q["action_veh"], ql): continue
        o = s["options"]
        vals = list(o.values()) if isinstance(o, dict) else o
        for v in vals:
            v = str(v).strip().rstrip(".").lower()
            if len(v) >= 7 and v not in SKIP_OPTS: vocab.add(v)
    parts = []
    for v in sorted(vocab, key=len, reverse=True):
        p = r"\b" + re.escape(v) + r"\b"
        if v.endswith("crossing"): p += r"(?!\s+destination)"
        parts.append(p)
    return re.compile("|".join(parts), re.I)


_ACT_PAT = None


def _subject_ok(text, start, subject):
    back = text[max(0, start - 100):start].lower()
    ip = max(back.rfind("pedestrian"), back.rfind(" he "), back.rfind(" she "))
    iv = max(back.rfind("vehicle"), back.rfind(" car "))
    if ip < 0 and iv < 0: return None      # không rõ -> để caller quyết theo channel
    return (ip > iv) if subject == "pedestrian" else (iv > ip)


def build_oracle5(vqa_samples_path, answers):
    """(scenario, phase) -> {subtype: option_text}"""
    out = collections.defaultdict(dict)
    for s in json.load(open(vqa_samples_path)):
        ql = s["question"].lower()
        sub = next((n for n, p in SUB_Q.items() if re.search(p, ql)), None)
        if not sub: continue
        L = answers.get(s["id"])
        if not L: continue
        out[(s["scenario"], str(s["phase_label"]))].setdefault(sub, _opt_text(s, L))
    return out


def _st_awareness(text, opt):
    o = str(opt).strip().rstrip(".").lower()
    if o in SKIP_OPTS: return text, 0
    m = AW_PAT.search(text)
    if not m: return text, 0
    cur = m.group(0).lower()
    pres = (" is " in " " + cur) or "notices" in cur or cur.startswith("is ")
    be = "is" if pres else "was"
    if o.startswith("unaware"): repl = f"{be} unaware of the vehicle"
    elif o.startswith("notice"): repl = "notices the vehicle" if pres else "noticed the vehicle"
    elif o.startswith("almost"): repl = f"almost {'notices' if pres else 'noticed'} the vehicle"
    else: return text, 0
    if _canon(cur) == _canon(repl): return text, 0
    return text[:m.start()] + repl + text[m.end():], 1


def _st_sight(text, opt):
    o = str(opt).strip().rstrip(".").lower()
    if o in SKIP_OPTS: return text, 0
    m = SIGHT_PAT.search(text)
    if not m: return text, 0
    phrase = SIGHT_OBJ.get(o, o)
    repl = f"line of sight {m.group(1).lower()} {phrase}"
    if _canon(m.group(0)) == _canon(repl): return text, 0
    return text[:m.start()] + repl + text[m.end():], 1


def _st_action(text, opt, subject, channel):
    global _ACT_PAT
    if _ACT_PAT is None: _ACT_PAT = _action_vocab()
    o = str(opt).strip().rstrip(".").lower()
    if o in SKIP_OPTS or len(o) < 7: return text, 0
    for m in _ACT_PAT.finditer(text):
        ok = _subject_ok(text, m.start(), subject)
        if ok is None: ok = (subject == channel)
        if not ok: continue
        if _canon(m.group(0)) == _canon(o): return text, 0
        return text[:m.start()] + o + text[m.end():], 1
    return text, 0


def stitch_v5(text, orc, target, subs):
    """orc = oracle5[(scn, phase)]; target = channel (pedestrian|vehicle)."""
    n = 0
    if target == "pedestrian":
        if "awareness" in subs and "awareness" in orc:
            text, k = _st_awareness(text, orc["awareness"]); n += k
        if "sight" in subs and "sight" in orc:
            text, k = _st_sight(text, orc["sight"]); n += k
        if "action_ped" in subs and "action_ped" in orc:
            text, k = _st_action(text, orc["action_ped"], "pedestrian", "pedestrian"); n += k
    else:
        if "action_veh" in subs and "action_veh" in orc:
            text, k = _st_action(text, orc["action_veh"], "vehicle", "vehicle"); n += k
    return text, n


def cmd_val():
    from harmonize_attrs import _score
    from fact_stitch import build_oracle as oracle_v2
    from fact_stitch3 import build_oracle_v3
    from fact_stitch4 import build_spatial_oracle, _apply_all
    ans = json.load(open(WORK / "wsd2_val_preds_decoded.json"))
    src = CODE / "output_qwen7b/processed/vqa_val.json"
    o2, o3 = oracle_v2(src, ans), build_oracle_v3(src, ans)
    sp = build_spatial_oracle(src, ans)
    o5 = build_oracle5(src, ans)
    preds = json.load(open(WORK / "caption_val_preds.json"))
    meta = {s["id"]: s for s in json.load(open(CODE / "output_qwen7b_caption/processed/caption_val.json"))}
    variants = {"base": (), "v5A": V5A, "v5AB": V5A + ("action_ped",)}
    outs, shown = {}, 0
    for name, subs in variants.items():
        g, nf = {}, 0
        for sid, p in preds.items():
            s = meta.get(sid)
            if not s: continue
            key = (s["scenario"], str(s["phase_label"]))
            t, _ = _apply_all(p, o2.get(s["scenario"], {}), o3.get(s["scenario"], {}),
                              sp.get(key, {}), s["target"], True)
            if subs:
                t2, k = stitch_v5(t, o5.get(key, {}), s["target"], subs)
                if k and name == "v5AB" and shown < 6 and t2 != t:
                    i = next(i for i in range(min(len(t), len(t2))) if t[i] != t2[i])
                    print(f"  vd[{s['target'][:3]}]: ...{t[max(0,i-40):i+50]}... -> ...{t2[max(0,i-40):i+50]}...")
                    shown += 1
                t = t2; nf += k
            g[sid] = (s, t)
        outs[name] = (g, nf)
        print(f"[v5] {name}: {nf} fixes v5")
    ob = _score(outs["base"][0])
    for name in ("v5A", "v5AB"):
        oa = _score(outs[name][0])
        print(f"=== {name} vs base ===")
        for k in ("BLEU-4", "METEOR", "ROUGE-L", "CIDEr"):
            print(f"  {k:<8}: {ob['mean_metrics'][k]:.2f} -> {oa['mean_metrics'][k]:.2f}  (Δ{oa['mean_metrics'][k]-ob['mean_metrics'][k]:+.2f})")
        print(f"  S1      : {ob['S1']:.2f} -> {oa['S1']:.2f}  (Δ{oa['S1']-ob['S1']:+.3f})")
        print(f"V5_DELTA[{name}]={oa['S1']-ob['S1']:+.4f}")


def cmd_test(base, out, answers_path, subs):
    raw = json.load(open(answers_path))
    ans = {x["id"]: x["correct"] for x in raw} if isinstance(raw, list) else dict(raw)
    src = CODE / "test_root_work_full/processed/vqa_val.json"
    o5 = build_oracle5(src, ans)
    cap = json.load(open(Path(base) / "subtask1_captioning.json"))
    nfix = 0
    for scn, entries in cap.items():
        for e in entries:
            lab = str((e.get("labels") or [""])[0])
            for ch, tgt in (("caption_pedestrian", "pedestrian"), ("caption_vehicle", "vehicle")):
                if e.get(ch):
                    e[ch], n = stitch_v5(e[ch], o5.get((scn, lab), {}), tgt, subs)
                    nfix += n
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    json.dump(cap, open(outp / "subtask1_captioning.json", "w"))
    shutil.copy(answers_path, outp / "subtask2_vqa.json")
    print(f"[v5] test: {nfix} fixes v5 ({','.join(subs)}) trên nền {base} -> {outp}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("val")
    t = sub.add_parser("test"); t.add_argument("--base", required=True); t.add_argument("--out", required=True)
    t.add_argument("--answers", required=True); t.add_argument("--subs", default=",".join(V5A))
    a = ap.parse_args()
    if a.cmd == "val": cmd_val()
    else: cmd_test(a.base, a.out, a.answers, tuple(a.subs.split(",")))


if __name__ == "__main__":
    main()
