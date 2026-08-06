#!/usr/bin/env python
from __future__ import annotations
import argparse, json, re, random, collections, statistics, sys
from pathlib import Path

CODE = Path("/workspace/AICC/code")
sys.path.insert(0, str(CODE))
from wsd import _qt, LETTERS

AICC = Path("/workspace/AICC")
BBOX_ROOTS = {
    "train": [AICC / "synwts/data/annotations/bbox_annotated"],
    "val":   [AICC / "synwts/data/annotations/bbox_annotated"],
    "test":  [AICC / "test_root/data/annotations/bbox_annotated",
              AICC / "test_root/data/annotations/bbox_generated"],
}
P_BBOX_DROP, P_CTX_DROP, P_CTX_NOISE = 0.30, 0.15, 0.25


def _median_boxes(path):
    try:
        j = json.load(open(path))
    except Exception:
        return {}
    anns = j["annotations"] if isinstance(j, dict) else j
    by = collections.defaultdict(list)
    for a in anns:
        b = a.get("bbox")
        if b and len(b) == 4:
            by[str(a.get("phase_number"))].append(b)
    return {ph: tuple(statistics.median(v[i] for v in bs) for i in range(4)) for ph, bs in by.items()}


class BBoxIndex:
    def __init__(self, split):
        self.roots = BBOX_ROOTS[split]
        self.split = split
        self.cache = {}
        self.size_cache = {}

    def _file(self, kind, s):
        for root in self.roots:
            for sub in ((self.split,) if self.split != "test" else ("", "test", "public")):
                base = root / kind / sub if sub else root / kind
                f = base / s["scenario"] / s["view"] / f"{s['base_video']}_bbox.json"
                if f.exists():
                    return f
            g = list(root.glob(f"{kind}/**/{s['base_video']}_bbox.json"))
            if g:
                return g[0]
        return None

    VIDEO_ROOTS = [AICC / "synwts/data/videos", AICC / "test_root/data/videos"]

    def _frame_size(self, s):
        bv = s["base_video"]
        if bv not in self.size_cache:
            wh = None
            for root in self.VIDEO_ROOTS:
                g = list(root.glob(f"**/{s['scenario']}/{s['view']}/{bv}.mp4")) or list(root.glob(f"**/{bv}.mp4"))
                if g:
                    import subprocess
                    try:
                        o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(g[0])],
                                           capture_output=True, text=True, timeout=15).stdout.strip()
                        w, h = o.split(",")[:2]
                        wh = (int(w), int(h))
                    except Exception:
                        pass
                    break
            self.size_cache[bv] = wh or (1920, 1080)
        return self.size_cache[bv]

    def get(self, s):
        W, H = self._frame_size(s)
        out = []
        for kind in ("pedestrian", "vehicle"):
            key = (kind, s["base_video"])
            if key not in self.cache:
                f = self._file(kind, s)
                self.cache[key] = _median_boxes(f) if f else {}
            b = self.cache[key].get(str(s["phase_label"]))
            out.append(f"({b[0]/W:.2f},{b[1]/H:.2f},{b[2]/W:.2f},{b[3]/H:.2f})" if b else None)
        return out


def _opt_items(s):
    o = s["options"]
    return list(o.items()) if isinstance(o, dict) else list(zip(LETTERS, o))


PSTATE_LABEL = {"orientation": "pedestrian orientation", "position_ped": "pedestrian position",
                "position_veh": "vehicle position", "direction": "pedestrian direction",
                "speed": "vehicle speed", "attention": "pedestrian attention",
                "distance": "relative distance", "action": "action"}


def _pstate_map(samples):
    ps = collections.defaultdict(dict)
    for s in samples:
        t = _qt(s["question"])
        if t:
            ps[(s["scenario"], str(s["phase_label"]))].setdefault(t, s)
    return ps


def _prev_map(samples):
    ch = collections.defaultdict(dict)
    for s in samples:
        t = _qt(s["question"])
        if not t:
            continue
        try:
            ph = int(s["phase_label"])
        except Exception:
            continue
        qn = re.sub(r"\s+", " ", s["question"].lower())[:45]
        ch[(s["scenario"], t, qn)][ph] = s
    prev = {}
    for phases in ch.values():
        ks = sorted(phases)
        for a, b in zip(ks, ks[1:]):
            if b - a == 1:
                prev[phases[b]["id"]] = phases[a]
    return prev


def _ans_text(ps, ans_map, rng, train_noise):
    if train_noise:
        if rng.random() < P_CTX_DROP:
            return None
        L = (ps["conversations"][1]["value"].strip()[:1] or "a").lower()
        if rng.random() < P_CTX_NOISE:
            L = rng.choice([x for x, _ in _opt_items(ps) if x != L] or [L])
    elif ans_map is not None:
        L = str(ans_map.get(ps["id"], "")).lower()[:1]
    else:
        return None
    return dict(_opt_items(ps)).get(L)


def build(samples_path, out_path, split, ctx_answers, mode, train_noise, phase_state=False):
    rng = random.Random(42)
    samples = json.load(open(samples_path))
    prev = _prev_map(samples)
    pstate = _pstate_map(samples) if phase_state else {}
    bidx = BBoxIndex(split)
    ans = json.load(open(ctx_answers)) if ctx_answers else None
    if isinstance(ans, list):
        ans = {x["id"]: x["correct"] for x in ans}
    n_bbox = n_ctx = n_drop_b = n_drop_c = n_noise = n_ps = 0
    for s in samples:
        blocks = []
        if phase_state:
            t0 = _qt(s["question"])
            if t0:
                parts = []
                for t, ps in sorted(pstate.get((s["scenario"], str(s["phase_label"])), {}).items()):
                    if t == t0: continue
                    txt = _ans_text(ps, ans, rng, train_noise)
                    if txt: parts.append(f"{PSTATE_LABEL[t]}: {str(txt).strip().rstrip('.').lower()}")
                if parts:
                    blocks.append("[Phase state] " + "; ".join(parts) + ".")
                    n_ps += 1
        if mode in ("both", "bbox"):
            if train_noise and rng.random() < P_BBOX_DROP:
                blocks.append("[Scene evidence] Bounding boxes: unavailable."); n_drop_b += 1
            else:
                p, v = bidx.get(s)
                if p or v:
                    blocks.append("[Scene evidence] Normalized boxes (x,y,w,h): "
                                  f"pedestrian {p or 'unavailable'}; vehicle {v or 'unavailable'}.")
                    n_bbox += 1
                else:
                    blocks.append("[Scene evidence] Bounding boxes: unavailable.")
        if mode in ("both", "ctx"):
            ps = prev.get(s["id"])
            txt = None
            if ps is not None:
                if train_noise:
                    if rng.random() < P_CTX_DROP:
                        n_drop_c += 1
                    else:
                        L = (ps["conversations"][1]["value"].strip()[:1] or "a").lower()
                        if rng.random() < P_CTX_NOISE:
                            L = rng.choice([x for x, _ in _opt_items(ps) if x != L] or [L]); n_noise += 1
                        d = dict(_opt_items(ps))
                        txt = d.get(L)
                elif ans is not None:
                    L = str(ans.get(ps["id"], "")).lower()[:1]
                    d = dict(_opt_items(ps))
                    txt = d.get(L)
            if txt:
                blocks.append(f"[Temporal context] In the previous phase, the answer to this same question was: {txt}")
                n_ctx += 1
        if blocks:
            s["conversations"][0]["value"] = s["conversations"][0]["value"].rstrip() + "\n" + "\n".join(blocks)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(samples, open(out_path, "w"))
    print(f"[ctx] {split}/{mode}: {len(samples)} samples | bbox {n_bbox} (drop {n_drop_b}) | "
          f"ctx {n_ctx} (drop {n_drop_c}, noise {n_noise}) | pstate {n_ps} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--ctx-answers", default=None)
    ap.add_argument("--mode", default="both", choices=["both", "ctx", "bbox", "none"])
    ap.add_argument("--train-noise", action="store_true")
    ap.add_argument("--phase-state", action="store_true")
    a = ap.parse_args()
    build(a.samples, a.out, a.split, a.ctx_answers, a.mode, a.train_noise, a.phase_state)


if __name__ == "__main__":
    main()
