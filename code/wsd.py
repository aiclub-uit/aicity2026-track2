#!/usr/bin/env python
from __future__ import annotations
import argparse, json, re, collections, math, sys
from pathlib import Path

CODE = Path("/workspace/AICC/code")
WORK = CODE / "output_qwen35"
QTYPES = {"orientation": r"orientation of", "position_ped": r"position of the pedestrian",
          "position_veh": r"position of the vehicle",
          "direction": r"\bdirection\b", "speed": r"\bspeed\b",
          "attention": r"line of sight|aware|attention|notice|looking",
          "distance": r"relative distance", "action": r"action|doing|moving|walking|running|crossing"}
SPATIAL3 = "orientation,position_ped,position_veh"
LETTERS = ["a", "b", "c", "d"]
EPS = 1e-6


def _qt(q):
    ql = q.lower()
    return next((n for n, p in QTYPES.items() if re.search(p, ql)), None)


def _canon(t):
    return re.sub(r"[^a-z ]", "", str(t).lower()).strip()[:60]


def _opts(sample):
    o = sample["options"]
    if isinstance(o, dict):
        return [_canon(o[L]) for L in LETTERS if L in o]
    return [_canon(x) for x in o[:4]]


def _gt_letter(s):
    return (s["conversations"][1]["value"].strip()[:1] or "a").lower()


def _chains(samples, types=None):
    ch = collections.defaultdict(dict)
    for s in samples:
        t = _qt(s["question"])
        if not t or (types and t not in types): continue
        try: ph = int(s["phase_label"])
        except Exception: continue
        qn = re.sub(r"\s+", " ", s["question"].lower())[:45]
        ch[(s["scenario"], t, qn)].setdefault(ph, s)
    return ch


def cmd_mine():
    d = json.load(open(CODE / "output_qwen7b/processed/vqa_train.json"))
    trans = {t: collections.defaultdict(collections.Counter) for t in QTYPES}
    prior = {t: collections.Counter() for t in QTYPES}
    for (scn, t, _qn), phases in _chains(d).items():
        ks = sorted(phases)
        sts = []
        for k in ks:
            s = phases[k]
            os_ = _opts(s); idx = "abcd".find(_gt_letter(s))
            if idx >= len(os_): continue
            st = os_[idx]
            sts.append((k, st)); prior[t][st] += 1
        for (k1, s1), (k2, s2) in zip(sts, sts[1:]):
            if k2 - k1 == 1:
                trans[t][s1][s2] += 1
    out = {t: {"prior": dict(prior[t]),
               "trans": {a: dict(c) for a, c in trans[t].items()}} for t in QTYPES}
    json.dump(out, open(WORK / "wsd_transitions.json", "w"))
    for t in QTYPES:
        sts = list(out[t]["prior"])
        print(f"[wsd] {t}: {len(sts)} states, prior={ {k[:25]: v for k, v in sorted(out[t]['prior'].items(), key=lambda x: -x[1])[:4]} }")


class Decoder:
    def __init__(self):
        self.T = json.load(open(WORK / "wsd_transitions.json"))

    def _p_trans(self, t, a, b, K):
        row = self.T[t]["trans"].get(a, {})
        tot = sum(row.values())
        return (row.get(b, 0) + 0.5) / (tot + 0.5 * K)

    def _p_prior(self, t, a):
        pr = self.T[t]["prior"]; tot = sum(pr.values()); K = len(pr)
        return (pr.get(a, 0) + 0.5) / (tot + 0.5 * K)

    def decode_chain(self, t, chain, probs):
        states = sorted({st for _, s in chain for st in _opts(s)} | set(self.T[t]["prior"]))
        K = len(states)
        def emis(s):
            p = probs.get(s["id"])
            if p is None: return None
            e = {st: EPS for st in states}
            for L, st, pv in zip(LETTERS, _opts(s), p):
                e[st] = max(pv, EPS)
            return e
        V, bp = [], []
        for i, (ph, s) in enumerate(chain):
            e = emis(s)
            if e is None: return {}
            if i == 0:
                V.append({st: math.log(self._p_prior(t, st)) + math.log(e[st]) for st in states}); bp.append({})
            else:
                v, b = {}, {}
                for st in states:
                    best, barg = -1e18, None
                    for pv in states:
                        sc = V[-1][pv] + math.log(self._p_trans(t, pv, st, K))
                        if sc > best: best, barg = sc, pv
                    v[st] = best + math.log(e[st]); b[st] = barg
                V.append(v); bp.append(b)
        cur = max(V[-1], key=V[-1].get); path = [cur]
        for i in range(len(chain) - 1, 0, -1):
            cur = bp[i][cur]; path.append(cur)
        path.reverse()
        out = {}
        for (ph, s), st in zip(chain, path):
            os_ = _opts(s)
            out[s["id"]] = LETTERS[os_.index(st)] if st in os_ else None
        return {k: v for k, v in out.items() if v}


def _decode_all(samples, probs, types=None):
    dec = Decoder()
    new = {}
    for (scn, t, _qn), phases in _chains(samples, types).items():
        ks = sorted(phases)
        chain = [(k, phases[k]) for k in ks]
        if len(chain) < 2: continue
        new.update(dec.decode_chain(t, chain, probs))
    return new


def cmd_gate(samples_path, probs_path, types=None):
    samples = json.load(open(samples_path))
    probs = json.load(open(probs_path))
    new = _decode_all(samples, probs, types)
    agg = collections.defaultdict(lambda: [0, 0, 0])
    tot = [0, 0, 0]
    for s in samples:
        t = _qt(s["question"]) or "other"
        gt = _gt_letter(s)
        am = LETTERS[max(range(4), key=lambda i: probs.get(s["id"], [0, 0, 0, 0])[i])] if s["id"] in probs else None
        vb = new.get(s["id"], am)
        if am is None: continue
        agg[t][2] += 1; agg[t][0] += (am == gt); agg[t][1] += (vb == gt)
        tot[2] += 1; tot[0] += (am == gt); tot[1] += (vb == gt)
    print("=== WSD GATE (argmax -> viterbi) ===")
    for t, (a, v, n) in sorted(agg.items()):
        print(f"  {t:<13}: {100*a/n:5.1f}% -> {100*v/n:5.1f}%  (Δ{100*(v-a)/n:+.1f}, n={n})")
    print(f"  TOTAL        : {100*tot[0]/tot[2]:5.1f}% -> {100*tot[1]/tot[2]:5.1f}%  (Δ{100*(tot[1]-tot[0])/tot[2]:+.2f}, n={tot[2]})")
    print(f"WSD_DELTA={100*(tot[1]-tot[0])/tot[2]:+.3f}")


def cmd_decode(samples_path, probs_path, base_path, out_path, types=None):
    samples = json.load(open(samples_path))
    probs = json.load(open(probs_path))
    base = json.load(open(base_path))
    if isinstance(base, list):
        answers = {x["id"]: x["correct"] for x in base}; as_list = True
    else:
        answers = dict(base); as_list = False
    new = _decode_all(samples, probs, types)
    n = sum(1 for k, v in new.items() if answers.get(k) and answers[k] != v)
    answers.update(new)
    out = [{"id": k, "correct": v} for k, v in answers.items()] if as_list else answers
    json.dump(out, open(out_path, "w"))
    print(f"[wsd] decode: {len(new)} câu qua Viterbi, {n} câu ĐỔI đáp án -> {out_path}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mine")
    g = sub.add_parser("gate"); g.add_argument("--samples", required=True); g.add_argument("--probs", required=True); g.add_argument("--types", default="")
    d = sub.add_parser("decode"); d.add_argument("--samples", required=True); d.add_argument("--probs", required=True)
    d.add_argument("--base-answers", required=True); d.add_argument("--out", required=True); d.add_argument("--types", default="")
    a = ap.parse_args()
    if a.cmd == "mine": cmd_mine()
    elif a.cmd == "gate": cmd_gate(a.samples, a.probs, set(a.types.split(",")) if a.types else None)
    else: cmd_decode(a.samples, a.probs, a.base_answers, a.out, set(a.types.split(",")) if a.types else None)


if __name__ == "__main__":
    main()
