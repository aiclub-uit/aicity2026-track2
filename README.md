# Team 24 (UIT–Kitchen) — AI City Challenge 2026 Track 2 Reproduction

End-to-end reproduction of our Track 2 submission.
The pipeline retrains every model from the SynWTS training set and regenerates
both submission files from the raw public-test package — no pretrained adapters,
no precomputed predictions.

## 1. Requirements

- Docker with the NVIDIA Container Toolkit (`--gpus all`)
- 1 GPU with ≥ 32 GB VRAM (bf16 training peaks ~29 GB; the original run used 96 GB)
- ~60 GB free disk (datasets + frames + adapters + HF model cache)

## 2. Data

Place (or symlink) the datasets under `data/` as described in
[data/README.md](data/README.md):

```
data/synwts/                   SynWTS training set  (HF: mlcglab/synwts, ~2.5 GB)
data/WTS_DATASET_PUBLIC_TEST/  official public-test videos + annotations (~21 GB)
data/WTS_TASK/                 official task package (BBOX zip + VQA json)
```

## 3. Run

```bash
docker build -t team24-e2e .

docker run --rm --gpus all --shm-size 8g \
  -v "$PWD:/pkg" -v team24-hf:/root/.cache/huggingface \
  team24-e2e
```

Final outputs: `submissions/subtask2_vqa.json` and
`submissions/subtask1_captioning.json`. Intermediates stay under `work/`;
the run is resumable (append `--list` for stage status).

## 4. Expected results

GPU LoRA training and sampling are not bit-exact across hardware/library
versions, so scores land within a small band of the leaderboard values
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
