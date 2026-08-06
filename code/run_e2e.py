#!/usr/bin/env python
"""run_e2e.py — Full end-to-end reproduction: raw data -> trained adapters -> final submissions."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parent
PKG = CODE.parent
SPATIAL = "orientation,position_ped,position_veh"


def load_mod(name: str, attrs: dict | None = None):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    return mod


def run(cmd, env=None, cwd=None):
    print(f"[e2e] $ {' '.join(str(c) for c in cmd)}", flush=True)
    e = dict(os.environ)
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    e.update({k: str(v) for k, v in (env or {}).items()})
    subprocess.run([str(c) for c in cmd], env=e, cwd=str(cwd or CODE), check=True)


def py_snippet(script: str, patches: dict, call: str, env=None, raw: dict | None = None):
    lines = [
        "import importlib.util, sys, pathlib",
        "from pathlib import Path",
        f"spec = importlib.util.spec_from_file_location({script!r}, {str(CODE / (script + '.py'))!r})",
        "m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m",
        f"sys.path.insert(0, {str(CODE)!r})",
        "spec.loader.exec_module(m)",
    ]
    for k, v in patches.items():
        lines.append(f"m.{k} = pathlib.Path({str(v)!r})" if isinstance(v, Path)
                     else f"m.{k} = {v!r}")
    for k, expr in (raw or {}).items():
        lines.append(f"m.{k} = {expr}")
    lines.append(f"m.{call}")
    run([sys.executable, "-c", "\n".join(lines)], env=env)


def ctx_patches(a) -> dict:
    syn = str(a.data / "synwts/data/annotations/bbox_annotated")
    t = str(a.work / "test_root/data/annotations")
    vids = (str(a.data / "synwts/data/videos"), str(a.work / "test_root/data/videos"))
    return {"BBOX_ROOTS": ("{'train': [Path(%r)], 'val': [Path(%r)], "
                           "'test': [Path(%r + '/bbox_annotated'), Path(%r + '/bbox_generated')]}"
                           % (syn, syn, t, t)),
            "BBoxIndex.VIDEO_ROOTS": "[Path(%r), Path(%r)]" % vids}


def flatten_synwts_for_caption(a) -> Path:
    flat = a.work / "synwts_flat/data"
    if flat.exists():
        shutil.rmtree(flat.parent)
    raw = a.data / "synwts/data"

    def merge(src: Path, dst: Path):
        dst.mkdir(parents=True, exist_ok=True)
        for e in src.iterdir():
            if e.name == "normal_trimmed":
                for c in e.iterdir():
                    (dst / c.name).symlink_to(c.resolve())
            else:
                (dst / e.name).symlink_to(e.resolve())

    for split in ("train", "val"):
        for sub in ("videos", "annotations/caption", "annotations/vqa",
                    "annotations/bbox_annotated/pedestrian",
                    "annotations/bbox_annotated/vehicle"):
            if (raw / sub / split).exists():
                merge(raw / sub / split, flat / sub / split)
    return flat


def sharded_predict(samples_json, out_probs, out_answers, adapter, quant,
                    qtypes="", batch=4, shard=2000, env_extra=None):
    workdir = Path(str(out_probs) + ".shards")
    workdir.mkdir(parents=True, exist_ok=True)
    data = json.load(open(samples_json))
    if shard % batch:
        sys.exit(f"--predict-batch {batch} must divide the shard size ({shard})")
    aw = Path(adapter) / "adapter_model.safetensors"
    prov = {"adapter": str(adapter), "quant": quant, "qtypes": qtypes,
            "batch": batch, "shard": shard, "n": len(data),
            "adapter_fp": [aw.stat().st_size, int(aw.stat().st_mtime)] if aw.exists() else None}
    pv = workdir / "provenance.json"
    if pv.exists() and json.load(open(pv)) != prov:
        sys.exit(f"shard cache {workdir} was produced with a different config "
                 f"({json.load(open(pv))}) — delete the folder to re-predict")
    json.dump(prov, open(pv, "w"))
    env = {"AICC26_QWEN35_QUANT": quant, "AICC26_QWEN35_ADAPTER": adapter,
           "QWEN35_BATCH": batch, **(env_extra or {})}
    shards = [(i, data[i:i + shard]) for i in range(0, len(data), shard)]
    print(f"[e2e] predict {len(data):,} samples, {len(shards)} shards, quant={quant}", flush=True)
    for k, (off, chunk) in enumerate(shards):
        pf = workdir / f"probs_{off:06d}.json"
        if pf.exists() and pf.stat().st_size > 2:
            continue
        sf = workdir / f"samples_{off:06d}.json"
        json.dump(chunk, open(sf, "w"))
        cmd = [sys.executable, "-u", CODE / "run_qwen35_vqa.py", "predict",
               "--samples", sf, "--out", workdir / f"preds_{off:06d}.json"]
        if qtypes:
            cmd += ["--qtypes", qtypes]
        print(f"[e2e] shard {k + 1}/{len(shards)}", flush=True)
        run(cmd, env={**env, "QWEN35_SAVE_PROBS": pf})
        sf.unlink()
    probs, preds = {}, {}
    for off, _ in shards:
        probs.update(json.load(open(workdir / f"probs_{off:06d}.json")))
        preds.update(json.load(open(workdir / f"preds_{off:06d}.json")))
    json.dump(probs, open(out_probs, "w"))
    json.dump([{"id": k, "correct": v} for k, v in preds.items()], open(out_answers, "w"))
    print(f"[e2e] merged {len(probs):,} -> {out_probs}", flush=True)


def st_prep_test(a):
    ext = a.data / "WTS_TASK/EXTERNAL_WTS_DATASET_TEST"
    inf = a.data / "WTS_DATASET_PUBLIC_TEST"
    for p in (ext / "SubTask2-VQA/WTS_VQA_PUBLIC_TEST.json", inf / "videos/test/public"):
        if not p.exists():
            sys.exit(f"missing input: {p}\nsee README.md §2 for the expected layout")
    bbox = ext / "SubTask1-Caption/WTS_DATASET_PUBLIC_TEST_BBOX"
    if not bbox.exists():
        z = Path(str(bbox) + ".zip")
        if not z.exists():
            sys.exit(f"missing input: {z}\nsee README.md §2 for the expected layout")
        print(f"[e2e] extracting {z.name}")
        zipfile.ZipFile(z).extractall(bbox.parent)
    troot = a.work / "test_root"
    m = load_mod("prepare_test_root")
    m.EXT_ROOT, m.INF_ROOT = ext, inf
    m.WTS_BBOX_ROOT = bbox
    m.BDD_BBOX_ROOT = bbox / "external/BDD_PC_5K"
    m.VQA_GATHERED = ext / "SubTask2-VQA/WTS_VQA_PUBLIC_TEST.json"
    m.WTS_VIDEOS_ROOT = inf / "videos/test/public"
    m.BDD_VIDEOS_ROOT = inf / "external/BDD_PC_5K/videos/test/public"
    m.WTS_CAPTION_ROOT = inf / "annotations/caption/test/public_challenge"
    m.BDD_CAPTION_ROOT = inf / "external/BDD_PC_5K/annotations/caption/test/public_challenge"
    m.DEST_ROOT, m.DEST_DATA = troot, troot / "data"
    m.main()
    env = {"AICC26_DATA_ROOT": troot / "data", "AICC26_PROJECT_ROOT": a.work / "test_prep",
           "AICC26_WORK_ROOT_QWEN7B": a.work / "test_prep/vqa",
           "AICC26_WORK_ROOT_QWEN7B_CAP": a.work / "test_prep/caption"}
    run([sys.executable, CODE / "preprocess_vqa.py", "--workers", a.workers, "--force"], env=env)
    run([sys.executable, CODE / "preprocess_caption.py", "--workers", a.workers, "--force"], env=env)
    ref = PKG / "data_meta/vqa_test.json"
    if ref.exists():
        new = json.load(open(a.work / "test_prep/vqa/processed/vqa_val.json"))
        old = json.load(open(ref))
        same = len(new) == len(old) and all(
            n["id"] == o["id"] and n["question"] == o["question"] and n["options"] == o["options"]
            for n, o in zip(new, old))
        print(f"[e2e] regenerated test metadata: {len(new)} questions -> "
              f"{'MATCHES frozen reference' if same else 'MISMATCH vs data_meta/vqa_test.json'}")
        if not same:
            sys.exit("metadata mismatch — check the test package under data/")


def st_prep_train(a):
    env = {"AICC26_DATA_ROOT": a.data / "synwts/data", "AICC26_PROJECT_ROOT": a.work / "train_prep",
           "AICC26_WORK_ROOT_QWEN7B": a.work / "train_prep/vqa",
           "AICC26_WORK_ROOT_QWEN7B_CAP": a.work / "train_prep/caption"}
    run([sys.executable, CODE / "preprocess_vqa.py", "--workers", a.workers, "--force"], env=env)
    flat = flatten_synwts_for_caption(a)
    run([sys.executable, CODE / "preprocess_caption.py", "--workers", a.workers, "--force"],
        env={**env, "AICC26_DATA_ROOT": flat})


def st_train_base(a):
    cmd = [sys.executable, "-u", CODE / "run_qwen35_vqa.py", "train"]
    if a.train_limit:
        cmd += ["--limit", a.train_limit]
    run(cmd, env={"AICC26_QWEN35_QUANT": "bf16",
                  "AICC26_DATA_DIR": a.work / "train_prep/vqa/processed",
                  "AICC26_QWEN35_ADAPTER": a.work / "adapters/vqa_lora"})


def st_predict_base(a):
    sharded_predict(a.work / "test_prep/vqa/processed/vqa_val.json",
                    a.work / "probs/base_bf16_probs.json", a.work / "vqa/base_answers.json",
                    a.work / "adapters/vqa_lora", "bf16", batch=a.predict_batch or 2)


def st_predict_8bit(a):
    sharded_predict(a.work / "test_prep/vqa/processed/vqa_val.json",
                    a.work / "probs/base_8bit_probs.json", a.work / "vqa/answers_8bit_unused.json",
                    a.work / "adapters/vqa_lora", "8bit", batch=a.predict_batch or 4)


def st_compose_wsd2(a):
    T = str(a.work / "test_prep/vqa/processed/vqa_val.json")
    import tempfile
    shim = Path(tempfile.mkdtemp(prefix="e2e_mine_"))
    (shim / "output_qwen7b/processed").mkdir(parents=True)
    shutil.copy(a.work / "train_prep/vqa/processed/vqa_train.json",
                shim / "output_qwen7b/processed/vqa_train.json")
    wsd = load_mod("wsd", {"WORK": a.work / "probs", "CODE": shim})
    wsd.cmd_mine()
    shutil.rmtree(shim, ignore_errors=True)
    wsd.cmd_decode(T, str(a.work / "probs/base_bf16_probs.json"),
                   str(a.work / "vqa/base_answers.json"),
                   str(a.work / "vqa/answers_wsd.json"), set(SPATIAL.split(",")))
    wsd.cmd_decode(T, str(a.work / "probs/base_8bit_probs.json"),
                   str(a.work / "vqa/answers_wsd.json"),
                   str(a.work / "vqa/answers_wsd2.json"), {"action"})


def st_train_ctx(a):
    d = a.work / "ctx_data"; d.mkdir(parents=True, exist_ok=True)
    py_snippet("ctx_builder", {},
               "build(%r, %r, 'train', None, 'both', True)" % (
                   str(a.work / "train_prep/vqa/processed/vqa_train.json"),
                   str(d / "vqa_train.json")),
               raw=ctx_patches(a))
    cmd = [sys.executable, "-u", CODE / "run_qwen35_vqa.py", "train"]
    if a.train_limit:
        cmd += ["--limit", a.train_limit]
    run(cmd, env={"AICC26_QWEN35_QUANT": "8bit", "AICC26_DATA_DIR": d,
                  "AICC26_QWEN35_ADAPTER": a.work / "adapters/vqa_lora_ctx"})


def st_predict_ctx(a):
    d = a.work / "ctx_data"
    py_snippet("ctx_builder", {},
               "build(%r, %r, 'test', %r, 'both', False)" % (
                   str(a.work / "test_prep/vqa/processed/vqa_val.json"),
                   str(d / "vqa_test_ctx.json"),
                   str(a.work / "vqa/answers_wsd2.json")),
               raw=ctx_patches(a))
    sharded_predict(d / "vqa_test_ctx.json", a.work / "probs/ctx_probs.json",
                    a.work / "vqa/ctx_answers_unused.json",
                    a.work / "adapters/vqa_lora_ctx", "8bit", qtypes=SPATIAL,
                    batch=a.predict_batch or 4)


def cosmos_paths(a):
    cw = a.work / "cosmos"
    pat = {"OUT": str(cw), "RAW": str(cw / "raw"), "META": str(cw / "meta.jsonl"),
           "SPECS": str(cw / "specs"), "RESTYLED": str(cw / "restyled"),
           "PROC": str(a.work / "train_prep/vqa/processed"),
           "DEST": str(a.work / "train_prep/vqa/processed_cosmos")}
    env = {"AICC26_DATA_ROOT": a.data / "synwts/data",
           "AICC26_PROJECT_ROOT": a.work / "train_prep",
           "AICC26_WORK_ROOT_QWEN7B": a.work / "train_prep/vqa"}
    return cw, pat, env


def st_cosmos_prep(a):
    cw, pat, env = cosmos_paths(a)
    cw.mkdir(parents=True, exist_ok=True)
    py_snippet("cosmos_restyle_prep", pat, "cmd_dump()", env=env)
    py_snippet("cosmos_restyle_prep", pat, "cmd_spec()", env=env)
    print(f"[e2e] specs ready: {cw}/specs — restyle them with nvidia/Cosmos-Transfer2.5-2B "
          f"into {cw}/restyled, then run --stages train_bundle")


def st_train_bundle(a):
    cw, pat, env = cosmos_paths(a)
    restyled = [f for f in (cw / "restyled").glob("*.jpg")
                if "control_edge" not in f.name] if (cw / "restyled").exists() else []
    if not restyled:
        sys.exit(f"no restyled frames in {cw}/restyled — run --stages cosmos_prep, restyle the "
                 "specs with Cosmos (README option C), then re-run this stage")
    py_snippet("cosmos_restyle_prep", pat, "cmd_rebuild()", env=env)
    dr = load_mod("dr_prep")
    from multiprocessing.pool import ThreadPool
    from PIL import Image
    src = json.load(open(a.work / "train_prep/vqa/processed_cosmos/vqa_train.json"))
    droot = cw / "frames_dr_bundle"
    droot.mkdir(parents=True, exist_ok=True)
    paths = sorted({p for s in src for p in s["images"]})

    def drp(p):
        return str(droot / (hashlib.md5(p.encode()).hexdigest()[:16] + ".jpg"))

    def work(p):
        out = Path(drp(p))
        if out.exists():
            return 0
        im = Image.open(p).convert("RGB")
        dr.augment(im, dr._rng(p)).save(out, "JPEG", quality=92)
        return 1

    with ThreadPool(16) as pool:
        made = sum(pool.map(work, paths))
    na = nt = 0
    for s in src:
        imgs = []
        for p in s["images"]:
            nt += 1
            if dr._rng("bundle:" + s["id"] + p).random() < 0.6:
                imgs.append(drp(p)); na += 1
            else:
                imgs.append(p)
        s["images"] = imgs
    outd = cw / "processed_bundle"
    outd.mkdir(parents=True, exist_ok=True)
    json.dump(src, open(outd / "vqa_train.json", "w"))
    print(f"[e2e] bundle train set: {made} DR variants, {na}/{nt} images use DR")
    run([sys.executable, "-u", CODE / "run_qwen35_vqa.py", "train"],
        env={"AICC26_QWEN35_QUANT": "bf16", "AICC26_DATA_DIR": outd,
             "AICC26_QWEN35_ADAPTER": a.work / "adapters/vqa_lora_bundle"})


def st_predict_bundle(a):
    ad = a.work / "adapters/vqa_lora_bundle"
    if not (ad / "adapter_model.safetensors").exists():
        src = PKG / "artifacts/adapters/vqa_lora_bundle"
        if not (src / "adapter_model.safetensors").exists() or not shipped_adapter_ok(src):
            sys.exit("bundle adapter not found in work/ or artifacts/ — retrain it via the "
                     "Cosmos route (README Option C), or restore artifacts/adapters/ (git LFS)")
        shutil.copytree(src, ad)
        print("[e2e] bundle adapter: using shipped artifact "
              "(from-scratch retrain requires Cosmos — see README Option C)")
    sharded_predict(a.work / "test_prep/vqa/processed/vqa_val.json",
                    a.work / "probs/bundle_probs.json",
                    a.work / "vqa/bundle_answers_unused.json",
                    ad, "bf16", batch=a.predict_batch or 2)


def st_compose_vqa(a):
    T = str(a.work / "test_prep/vqa/processed/vqa_val.json")
    wsd = load_mod("wsd", {"WORK": a.work / "probs"})
    wsd.cmd_decode(T, str(a.work / "probs/ctx_probs.json"),
                   str(a.work / "vqa/answers_wsd2.json"),
                   str(a.work / "vqa/answers_ctx.json"), set(SPATIAL.split(",")))
    final = a.work / "vqa/answers_ctx.json"
    bundle = a.work / "probs/bundle_probs.json"
    if bundle.exists() and not a.with_bundle:
        print("[e2e] bundle probs present but --no-bundle set -> ignored")
    if a.with_bundle and not bundle.exists():
        sys.exit(f"bundle route needs {bundle} — run the predict_bundle stage first")
    if a.with_bundle and bundle.exists():
        SP = set(SPATIAL.split(",")); VIT = SP | {"action"}
        TGT = {"direction", "speed", "attention", "distance", "action"}
        samples = json.load(open(T))
        base = json.load(open(a.work / "probs/base_8bit_probs.json"))
        ctxp = json.load(open(a.work / "probs/ctx_probs.json"))
        bu = json.load(open(bundle)); EPS = 1e-5
        mixed = {}
        for s in samples:
            qid, p = s["id"], base.get(s["id"])
            if not p:
                continue
            t = wsd._qt(s["question"]) or "other"
            f = ctxp[qid] if (t in SP and qid in ctxp) else (bu[qid] if (t in TGT and qid in bu) else p)
            tt = sum(max(x, EPS) for x in f)
            mixed[qid] = [max(x, EPS) / tt for x in f]
        v = {s["id"]: wsd.LETTERS[max(range(4), key=lambda i: mixed[s["id"]][i])]
             for s in samples if s["id"] in mixed}
        v.update(wsd._decode_all(samples, mixed, VIT))
        ans = {x["id"]: x["correct"] for x in json.load(open(final))}
        for s in samples:
            if (wsd._qt(s["question"]) or "other") in TGT and s["id"] in v:
                ans[s["id"]] = v[s["id"]]
        final = a.work / "vqa/answers_bundle.json"
        json.dump([{"id": k, "correct": x} for k, x in ans.items()], open(final, "w"))
        print("[e2e] bundle probs found -> reality-transfer route applied")
    else:
        print("[e2e] no bundle probs (optional stage skipped) -> final = ctx stage")
    meta = {s_["id"]: s_ for s_ in json.load(open(T))}
    ans = json.load(open(final))
    ids = [x["id"] for x in ans]
    if len(ids) != len(set(ids)) or set(ids) != set(meta):
        sys.exit(f"VALIDATION FAIL: id mismatch (answers {len(set(ids))} vs metadata {len(meta)})")
    p8 = json.load(open(a.work / "probs/base_8bit_probs.json"))
    clamped = 0
    for x in ans:
        opts = meta[x["id"]]["options"]
        n = len(opts)
        if "abcd".index(x["correct"]) >= n:
            pr = p8.get(x["id"], [0.25] * 4)
            x["correct"] = "abcd"[max(range(n), key=lambda i: pr[i])]
            clamped += 1
    if clamped:
        json.dump(ans, open(final, "w"))
    print(f"[e2e] validation: {len(ids):,} answers, ids match metadata, "
          f"{clamped} out-of-range answers clamped")
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(final, a.out / "subtask2_vqa.json")
    print(f"[e2e] FINAL VQA -> {a.out / 'subtask2_vqa.json'}")


def st_train_caption(a):
    env = {"AICC26_QWEN35_QUANT": a.quant_caption,
           "AICC26_QWEN35_CAP_ADAPTER": a.work / "adapters/caption_lora",
           "AICC26_SFT_SRC": a.work / "adapters/caption_lora",
           "AICC26_DPO_OUT": a.work / "adapters/caption_dpo_lora",
           "AICC26_DPO_PAIRS": a.work / "caption/dpo_pairs.json"}
    data = a.work / "train_prep/caption/processed"
    lim = int(a.caption_limit or 0)
    py_snippet("run_qwen35_caption", {"DATA": data}, f"cmd_train({lim or None})", env=env)
    py_snippet("run_qwen35_caption", {"DATA": data}, f"cmd_build_dpo({lim})", env=env)
    py_snippet("run_qwen35_caption", {"DATA": data}, "cmd_train_dpo()", env=env)


def st_predict_caption(a):
    (a.work / "caption").mkdir(parents=True, exist_ok=True)
    test_meta = a.work / "test_prep/caption/processed/caption_val.json"
    py_snippet("run_qwen35_caption", {},
               "cmd_predict_submission(%r, %r)" % (
                   str(test_meta), str(a.work / "caption/caption_test_preds.json")),
               env={"AICC26_QWEN35_QUANT": a.quant_caption,
                    "AICC26_CAP_PRED_ADAPTER": a.work / "adapters/caption_dpo_lora"})
    preds = json.load(open(a.work / "caption/caption_test_preds.json"))
    meta = json.load(open(test_meta))
    sub: dict[str, dict] = {}
    for s in meta:
        key = f"{s['scenario']}|{s['phase_label']}|{s['target']}"
        e = sub.setdefault(s["scenario"], {}).setdefault(
            str(s["phase_label"]),
            {"labels": [str(s["phase_label"])],
             "caption_pedestrian": "", "caption_vehicle": ""})
        e["caption_pedestrian" if s["target"] == "pedestrian" else "caption_vehicle"] = \
            preds.get(key, "")
    out = {scn: [phases[k] for k in sorted(phases)] for scn, phases in sub.items()}
    base_dir = a.work / "caption/base_dir"; base_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(base_dir / "subtask1_captioning.json", "w"))
    shutil.copy(a.work / "vqa/base_answers.json", base_dir / "subtask2_vqa.json")
    print(f"[e2e] draft captions: {len(out)} scenarios -> {base_dir}")


def st_stitch(a):
    import tempfile
    shim = Path(tempfile.mkdtemp(prefix="e2e_shim_"))
    try:
        (shim / "test_root_work_full/processed").mkdir(parents=True)
        (shim / "output_qwen7b/processed").mkdir(parents=True)
        (shim / "output_qwen7b_caption/processed").mkdir(parents=True)
        shutil.copy(a.work / "test_prep/vqa/processed/vqa_val.json",
                    shim / "test_root_work_full/processed/vqa_val.json")
        shutil.copy(a.work / "train_prep/vqa/processed/vqa_train.json",
                    shim / "output_qwen7b/processed/vqa_train.json")
        shutil.copy(a.work / "test_prep/caption/processed/caption_val.json",
                    shim / "output_qwen7b_caption/processed/caption_val.json")
        fs = load_mod("fact_stitch", {"CODE": shim})
        fs3 = load_mod("fact_stitch3", {"CODE": shim,
                                        "ORACLE_TEST": str(a.work / "vqa/base_answers.json")})
        fs4 = load_mod("fact_stitch4", {"CODE": shim})
        fs5 = load_mod("fact_stitch5", {"CODE": shim})
        w = a.work / "caption"
        fs.cmd_test(str(w / "base_dir"), str(w / "v2"))
        fs3.cmd_test(str(w / "v2"), str(w / "v3"),
                     {"brightness", "color_lower", "color_upper", "lower_item",
                      "surface_cond", "surface_type"})
        fs4.cmd_test(str(w / "base_dir"), str(w / "v4"), str(a.work / "vqa/answers_wsd.json"))
        fs5.cmd_test(str(w / "v4"), str(w / "v5"), str(a.work / "vqa/answers_wsd2.json"),
                     fs5.V5A)
        cap = json.load(open(w / "v5/subtask1_captioning.json"))
        need = {(s_["scenario"], str(s_["phase_label"]))
                for s_ in json.load(open(a.work / "test_prep/caption/processed/caption_val.json"))}
        pairs = [(scn, str((e.get("labels") or [""])[0]))
                 for scn, es in cap.items() for e in es]
        have = set(pairs)
        missing, extra, dup = need - have, have - need, len(pairs) - len(have)
        empty = sum(1 for es in cap.values() for e in es
                    for ch in ("caption_pedestrian", "caption_vehicle")
                    if not isinstance(e.get(ch), str) or not e[ch].strip())
        if missing or extra or dup or empty:
            sys.exit(f"VALIDATION FAIL: {len(missing)} missing / {len(extra)} extra "
                     f"(scenario, phase), {dup} duplicates, {empty} empty caption fields; "
                     f"e.g. {sorted(missing or extra)[:3]}")
        print(f"[e2e] validation: {len(cap)} scenarios, {len(pairs)} phase entries, "
              f"all covered, none empty")
        a.out.mkdir(parents=True, exist_ok=True)
        shutil.copy(w / "v5/subtask1_captioning.json", a.out / "subtask1_captioning.json")
        print(f"[e2e] FINAL CAPTION -> {a.out / 'subtask1_captioning.json'}")
    finally:
        shutil.rmtree(shim, ignore_errors=True)


STAGES = [("prep_test", st_prep_test), ("prep_train", st_prep_train),
          ("train_base", st_train_base), ("predict_base", st_predict_base),
          ("predict_8bit", st_predict_8bit), ("compose_wsd2", st_compose_wsd2),
          ("train_ctx", st_train_ctx), ("predict_ctx", st_predict_ctx),
          ("cosmos_prep", st_cosmos_prep), ("train_bundle", st_train_bundle),
          ("predict_bundle", st_predict_bundle),
          ("compose_vqa", st_compose_vqa), ("train_caption", st_train_caption),
          ("predict_caption", st_predict_caption), ("stitch", st_stitch)]


def shipped_adapter_ok(d: Path) -> bool:
    """False for a missing file AND for an unsmudged git-LFS pointer (~134 bytes)."""
    f = d / "adapter_model.safetensors"
    if not f.exists():
        return False
    if f.stat().st_size < 1_000_000:
        sys.exit(f"{f} is a git-LFS pointer, not the real weights — "
                 "run `git lfs install && git lfs pull` in the package first")
    return True


def seed_shipped_adapters(a):
    src = PKG / "artifacts/adapters"
    for stage, name in [("train_base", "vqa_lora"), ("train_ctx", "vqa_lora_ctx"),
                        ("train_caption", "caption_dpo_lora")]:
        s, d = src / name, a.work / "adapters" / name
        if not shipped_adapter_ok(s):
            sys.exit(f"--skip-training: shipped adapter missing: {s}")
        if not d.exists():
            shutil.copytree(s, d)
        (a.work / ".done" / stage).touch()
    b = src / "vqa_lora_bundle"
    if (b / "adapter_model.safetensors").exists() and shipped_adapter_ok(b) \
            and not (a.work / "adapters/vqa_lora_bundle").exists():
        shutil.copytree(b, a.work / "adapters/vqa_lora_bundle")
    print("[e2e] --skip-training: shipped adapters seeded, train stages marked done")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=PKG / "data")
    ap.add_argument("--work", type=Path, default=PKG / "work")
    ap.add_argument("--out", type=Path, default=PKG / "submissions")
    ap.add_argument("--stages", default="all", help="comma list, default all")
    ap.add_argument("--workers", type=int, default=16, help="preprocess workers")
    ap.add_argument("--train-limit", type=int, default=0, help="smoke: cap train samples")
    ap.add_argument("--caption-limit", type=int, default=0, help="smoke: cap caption samples")
    ap.add_argument("--quant-caption", default="bf16", choices=["bf16", "8bit", "4bit"])
    ap.add_argument("--no-bundle", action="store_true",
                    help="skip the bundle (reality-transfer) route; the submitted best VQA uses it")
    ap.add_argument("--skip-training", action="store_true",
                    help="use the shipped adapters in artifacts/ instead of retraining")
    ap.add_argument("--predict-batch", type=int, default=0,
                    help="prediction batch size; 0 = defaults sized for 32 GB VRAM "
                         "(bf16: 2, 8-bit: 4) — raise on larger GPUs, must divide 2000")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    a.with_bundle = not a.no_bundle
    done = a.work / ".done"; done.mkdir(parents=True, exist_ok=True)
    for sub in ("probs", "vqa", "adapters", "caption"):
        (a.work / sub).mkdir(parents=True, exist_ok=True)
    if a.list:
        for n, _ in STAGES:
            print(f"  {'✔' if (done / n).exists() else '·'} {n}")
        return
    if a.skip_training:
        seed_shipped_adapters(a)
    want = None if a.stages == "all" else set(a.stages.split(","))
    for name, fn in STAGES:
        if want and name not in want:
            continue
        if name in ("cosmos_prep", "train_bundle") and not want:
            continue
        if name == "predict_bundle" and not a.with_bundle and not want:
            print("[e2e] predict_bundle: skipped (--no-bundle)")
            continue
        if not want and (done / name).exists():
            print(f"[e2e] {name}: done, skip")
            continue
        print(f"\n[e2e] ======== STAGE {name} ========", flush=True)
        fn(a)
        (done / name).touch()
    print("\n[e2e] all requested stages finished")


if __name__ == "__main__":
    main()
