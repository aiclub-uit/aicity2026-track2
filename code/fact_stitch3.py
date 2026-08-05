#!/usr/bin/env python
"""fact_stitch3.py — Fact-Stitch v3 "full-pillar": mở rộng oracle sang color/lower/brightness/
inclination/surface/distance với 2 chế độ: CORRECTIVE (sửa giá trị sai) + ADDITIVE (chèn slot thiếu
bằng câu template canonical mined từ GT). Chồng LÊN v2 (age/height/clothing-upper/weather).

Gate: val per-slot (MBR-300) -> chỉ giữ slot có ΔS1 >= 0. CLI:
  val [--slots s1,s2|all]     # per-metric before/after
  test --base DIR --out DIR [--slots ...]
"""
from __future__ import annotations
import argparse, json, re, collections, shutil, sys
from pathlib import Path

CODE = Path("/workspace/AICC/code")
WORK = CODE / "output_qwen35"
# Oracle test = đáp án FULL_v2. Hằng module để reproduce_all.py patch về bản trong gói
# (submissions/base_full_v2_subtask2.json — byte-identical file deploy, md5 f6901835...).
ORACLE_TEST = "/workspace/AICC/SUBMIT_Q35_FULL_v2/subtask2_vqa.json"
sys.path.insert(0, str(CODE))
from fact_stitch import build_oracle as build_oracle_v2, stitch as stitch_v2   # v2 slots

LOWER_NOUNS = r"slacks|trousers|pants|jeans|skirt|shorts"
UPPER_NOUNS = r"t-?shirt|shirt|jacket|coat|hoodie|sweater|parka|blouse"
COLORS = r"(?:navy blue|light blue|dark blue|navy|black|white|blue|beige|brown|gr[ae]y|green|red|purple|yellow|orange|pink|khaki|dark|light)"

# slot: (question-pattern, canon(option_text)->value|None)
V3_Q = {
    "color_upper": (r"colou?r of pedestrian'?s upper body", lambda t: t.lower().strip() or None),
    "color_lower": (r"colou?r of pedestrian'?s lower body", lambda t: t.lower().strip() or None),
    "lower_item":  (r"wearing on lower body", lambda t: t.lower().strip() or None),
    "brightness":  (r"brightness level", lambda t: t.lower().strip() or None),          # bright|dim|dark
    "inclination": (r"road inclination", lambda t: t.lower().strip() or None),          # level|downhill|...
    "surface_cond": (r"road surface conditions", lambda t: t.lower().strip() or None),  # dry|wet|frozen
    "surface_type": (r"surface type of the road", lambda t: t.lower().strip() or None), # asphalt|...
    "distance":    (r"relative distance of pedestrian from vehicle", lambda t: t.lower().strip() or None),
}
ALL_SLOTS = list(V3_Q)


def build_oracle_v3(vqa_samples_path, answers):
    from fact_stitch import _opt_text
    samples = json.load(open(vqa_samples_path))
    votes = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for s in samples:
        ql = s["question"].lower()
        for name, (pat, canon) in V3_Q.items():
            if re.search(pat, ql):
                L = answers.get(s["id"])
                if L:
                    v = canon(_opt_text(s, L))
                    if v: votes[s["scenario"]][name][v] += 1
                break
    oracle = {}
    for scn, d in votes.items():
        o = {}
        for name, cnt in d.items():
            top = cnt.most_common(2)
            if len(top) == 1 or top[0][1] > top[1][1]:
                o[name] = top[0][0]
        oracle[scn] = o
    return oracle


def _set_color(text, noun_re, color):
    """thay/chèn màu ngay trước item noun (corrective/insert)."""
    m = re.search(rf"\b((?:{COLORS})(?:\s+{COLORS})?\s+)?({noun_re})\b", text, re.I)
    if not m: return text, 0
    cur = (m.group(1) or "").strip().lower()
    if cur == color: return text, 0
    start, end = m.span()
    repl = f"{color} {m.group(2)}"
    return text[:start] + repl + text[end:], 1


def _replace_in_ctx(text, ctx_pat, val_pat, new):
    """thay từ khớp val_pat trong CÂU có ctx_pat; trả (text, changed)."""
    out, changed = [], 0
    for sent in re.split(r"(?<=[.;])", text):
        if re.search(ctx_pat, sent, re.I):
            ns = re.sub(val_pat, new, sent, flags=re.I)
            if ns != sent and not changed: changed = 1; sent = ns
        out.append(sent)
    return "".join(out), changed


def stitch_v3(text, o, target, slots):
    n = 0
    if target == "pedestrian":
        if "color_upper" in slots and o.get("color_upper"):
            text, c = _set_color(text, UPPER_NOUNS, o["color_upper"]); n += c
        if "color_lower" in slots and o.get("color_lower"):
            text, c = _set_color(text, LOWER_NOUNS, o["color_lower"]); n += c
        if "lower_item" in slots and o.get("lower_item"):
            new = re.sub(rf"\b(?:{LOWER_NOUNS})\b", o["lower_item"], text, count=1, flags=re.I)
            if new != text: n += 1; text = new
        if "distance" in slots and o.get("distance"):
            if re.search(r"relative distance", text, re.I):
                text, c = _replace_in_ctx(text, r"relative distance", r"\b(far|close|near)\b", o["distance"]); n += c
            else:
                text = text.rstrip() + f" The relative distance between the pedestrian and the vehicle is {o['distance']}."
                n += 1
    # env slots: cả 2 channel
    if "brightness" in slots and o.get("brightness"):
        if re.search(r"\bbrightness|\blighting\b", text, re.I):
            text, c = _replace_in_ctx(text, r"brightness|lighting", r"\b(bright|dim|dark)\b", o["brightness"]); n += c
        else:
            text = text.rstrip() + f" The brightness is {o['brightness']}."; n += 1
    if "inclination" in slots and o.get("inclination"):
        if re.search(r"\binclin", text, re.I):
            text, c = _replace_in_ctx(text, r"inclin", r"\b(level|downhill|steep uphill|gentle slope|uphill)\b", o["inclination"]); n += c
        else:
            text = text.rstrip() + f" The road inclination is {o['inclination']}."; n += 1
    if "surface_cond" in slots and o.get("surface_cond"):
        text, c = _replace_in_ctx(text, r"road|surface|asphalt", r"\b(dry|wet|frozen)\b", o["surface_cond"]); n += c
    if "surface_type" in slots and o.get("surface_type"):
        text, c = _replace_in_ctx(text, r"road|surface", r"\b(asphalt|concrete|gravel|dirt)\b", o["surface_type"]); n += c
    return text, n


def _load_oracles(split):
    if split == "val":
        ans = json.load(open(WORK / "vqa_val_preds_full.json"))
        src = CODE / "output_qwen7b/processed/vqa_val.json"
    else:
        raw = json.load(open(ORACLE_TEST))
        ans = {x["id"]: x["correct"] for x in raw}
        src = CODE / "test_root_work_full/processed/vqa_val.json"
    return build_oracle_v2(src, ans), build_oracle_v3(src, ans)


def cmd_val(slots):
    from harmonize_attrs import _score
    o2, o3 = _load_oracles("val")
    preds = json.load(open(WORK / "caption_val_preds.json"))
    meta = {s["id"]: s for s in json.load(open(CODE / "output_qwen7b_caption/processed/caption_val.json"))}
    gen = {sid: (meta[sid], p) for sid, p in preds.items() if sid in meta}
    fixed, nfix = {}, 0
    for sid, (s, p) in gen.items():
        t, n1 = stitch_v2(p, o2.get(s["scenario"], {}), s["target"])          # v2 luôn bật (đã proven)
        t, n2 = stitch_v3(t, o3.get(s["scenario"], {}), s["target"], slots)
        fixed[sid] = (s, t); nfix += n1 + n2
    ob, oa = _score(gen), _score(fixed)
    print(f"[v3] slots={sorted(slots)} | fixes={nfix}")
    for k in ("BLEU-4", "METEOR", "ROUGE-L", "CIDEr"):
        print(f"  {k:<8}: {ob['mean_metrics'][k]:.2f} -> {oa['mean_metrics'][k]:.2f}  (Δ{oa['mean_metrics'][k]-ob['mean_metrics'][k]:+.2f})")
    print(f"  S1      : {ob['S1']:.2f} -> {oa['S1']:.2f}  (Δ{oa['S1']-ob['S1']:+.3f})")
    print(f"V3_DELTA={oa['S1']-ob['S1']:+.4f}")


def cmd_test(base, out, slots):
    o2, o3 = _load_oracles("test")
    cap = json.load(open(Path(base) / "subtask1_captioning.json"))
    nfix = 0
    for scn, entries in cap.items():
        for e in entries:
            for ch, tgt in (("caption_pedestrian", "pedestrian"), ("caption_vehicle", "vehicle")):
                if e.get(ch):
                    t, n1 = stitch_v2(e[ch], o2.get(scn, {}), tgt)
                    t, n2 = stitch_v3(t, o3.get(scn, {}), tgt, slots)
                    e[ch] = t; nfix += n1 + n2
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    json.dump(cap, open(outp / "subtask1_captioning.json", "w"))
    shutil.copy(Path(base) / "subtask2_vqa.json", outp / "subtask2_vqa.json")
    print(f"[v3] test: {nfix} fixes (v2+v3, slots={sorted(slots)}) -> {outp}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("val"); v.add_argument("--slots", default="all")
    t = sub.add_parser("test"); t.add_argument("--base", required=True); t.add_argument("--out", required=True)
    t.add_argument("--slots", default="all")
    a = ap.parse_args()
    slots = set(ALL_SLOTS) if a.slots == "all" else set(a.slots.split(","))
    if a.cmd == "val": cmd_val(slots)
    else: cmd_test(a.base, a.out, slots)


if __name__ == "__main__":
    main()
