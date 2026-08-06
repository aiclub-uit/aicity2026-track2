# Team 24 (UIT–Kitchen) — AI City Challenge 2026 Track 2 Reproduction

End-to-end reproduction of our Track 2 submission.
The pipeline retrains every model from the SynWTS training set and regenerates
both submission files from the raw public-test package — no pretrained adapters,
no precomputed predictions.

## 1. Requirements

- Docker with the NVIDIA Container Toolkit (`--gpus all`)
- 1 NVIDIA GPU, Ampere or newer (compute capability ≥ 8.0 — bf16 training; the
  PyTorch 2.11 cu128 binaries no longer ship Volta kernels), ≥ 32 GB VRAM
  (bf16 training peaks ~29 GB; the original run used 96 GB)
- ~60 GB free disk (datasets + frames + adapters + HF model cache)

## 2. Data

Place the three datasets under `data/` (nothing in `data/` ships with the package):

```
data/
├── synwts/                          # SynWTS training set
│   └── data/
│       ├── annotations/{bbox_annotated,caption,vqa}/
│       └── videos/{train,val}/
├── WTS_DATASET_PUBLIC_TEST/         # official public-test package
│   ├── annotations/caption/test/public_challenge/
│   ├── videos/test/public/
│   └── external/BDD_PC_5K/{annotations,videos}/
└── WTS_TASK/
    └── EXTERNAL_WTS_DATASET_TEST/
        ├── SubTask1-Caption/WTS_DATASET_PUBLIC_TEST_BBOX.zip   # auto-extracted
        └── SubTask2-VQA/WTS_VQA_PUBLIC_TEST.json
```

- SynWTS (~2.5 GB): `huggingface_hub.snapshot_download("mlcglab/synwts",
  repo_type="dataset", revision="9d6e7abce9a906e3e9bb6a2941dcce8b128654cf")`
  — the download root is `data/synwts/` (revision pinned to the version we trained on).
- Public test + task package (~21 GB): distributed by the AI City Challenge
  organizers (Track 2).

If a dataset lives elsewhere, do not symlink it (links pointing outside the
package do not resolve in the container) — bind-mount it instead, e.g.
`-v /path/to/synwts:/pkg/data/synwts:ro` added to the `docker run` below.

## 3. Run

```bash
docker build -t team24-e2e .
```

### Option A — full end-to-end (retrains everything, no artifacts needed)

Retrains all LoRA adapters from SynWTS, then predicts. ~3 GPU-days total.

```bash
docker run --rm --gpus all --shm-size 8g \
  -v "$PWD:/pkg" -v team24-hf:/root/.cache/huggingface \
  team24-e2e
```

### Option B — use the shipped artifacts (skip training, ~1 GPU-day)

Uses our deployed LoRA adapters from `artifacts/adapters/` (git LFS) and goes
straight to prediction.

```bash
docker run --rm --gpus all --shm-size 8g \
  -v "$PWD:/pkg" -v team24-hf:/root/.cache/huggingface \
  team24-e2e --skip-training
```

Both options write the final outputs `submissions/subtask2_vqa.json` and
`submissions/subtask1_captioning.json`. Intermediates stay under `work/`;
the run is resumable (append `--list` for stage status).

## 4. Expected results

Target scores on the public test set: **VQA accuracy ≈ 84.7**, **caption S1 ≈ 29.9–30.0**.
GPU LoRA training and sampling are not bit-exact across hardware/library
versions, so scores land within a small band of these values
(±0.1–0.2 accuracy / S1 in our re-runs).

## 5. Code map

| File | Role |
|---|---|
| `run_e2e.py` | orchestrator (the only new file; everything below is the deployed code, kept verbatim) |
| `preprocess_vqa.py` / `preprocess_caption.py` | frames + bbox drawing + crops + metadata (cv2 only) |
| `prepare_test_root.py` | mounts the official test package as a val-as-test layout |
| `run_qwen35_vqa.py` | VQA train / predict (letter-logit, LoRA, bf16/8-bit/4-bit) |
| `run_qwen35_caption.py` | caption SFT / DPO / greedy test prediction |
| `ctx_builder.py` | temporal-context + bbox-evidence training/eval data |
| `wsd.py` | transition mining + canonical-state Viterbi decoding |
| `fact_stitch.py`, `fact_stitch3/4/5.py` | caption fact correction waves v2–v5 |
| `harmonize_attrs.py` | cross-phase attribute harmonization (val lever) |
| `cosmos_restyle_prep.py`, `dr_prep.py`, `*.sh` | optional bundle route + historical deploy pipelines |
