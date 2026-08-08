# Team 24 (UIT–Kitchen) — AI City Challenge 2026 Track 2 Reproduction

End-to-end reproduction of our Track 2 submission.
The pipeline retrains the VQA, context, and caption adapters from the SynWTS
training set and regenerates both submission files from the raw public-test
package — no precomputed predictions. The one exception is the bundle adapter:
its training needs an external Cosmos restyle step (Option C), so by default
the pipeline predicts with the shipped copy from `artifacts/adapters/`.

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
  — the download root is `data/synwts/` (revision pinned to the version we trained
  on). The dataset is gated: log in to Hugging Face (`hf auth login`) and accept
  the access terms on the dataset page first.
- Public test + task package (~21 GB): distributed by the AI City Challenge
  organizers (Track 2).

If a dataset lives elsewhere, do not symlink it (links pointing outside the
package do not resolve in the container) — bind-mount it instead, e.g.
`-v /path/to/synwts:/pkg/data/synwts:ro` added to the `docker run` below.

## 3. Run

```bash
docker build -t team24-e2e .
```

### Option A — reproduce with the shipped adapters (recommended)

One command from raw data to both submission files. Uses our trained LoRA
adapters from `artifacts/adapters/` (git LFS); regenerates everything else —
preprocessing, all predictions, composition, captions. ~1 GPU-day.

```bash
docker run --rm --gpus all --shm-size 8g \
  -v "$PWD:/pkg" -v team24-hf:/root/.cache/huggingface \
  team24-e2e --skip-training
```

### Option B — retrain the main adapters (VQA, context, caption)

Retrains the three main adapters from SynWTS before predicting
(~3 GPU-days total). Not fully from scratch: the bundle adapter still comes
from `artifacts/` — retraining that one as well requires the external Cosmos
step in Option C.

```bash
docker run --rm --gpus all --shm-size 8g \
  -v "$PWD:/pkg" -v team24-hf:/root/.cache/huggingface \
  team24-e2e
```

Both options write the final outputs `submissions/subtask2_vqa.json` and
`submissions/subtask1_captioning.json`. Intermediates stay under `work/`;
the run is resumable (append `--list` for stage status).

Batch sizes default to the 32 GB VRAM minimum. On larger GPUs (≥ 48 GB) append
`--predict-batch 8` (any divisor of 2000) to speed up the prediction stages.
On GPUs under 40 GB the `train_ctx` and caption-DPO steps automatically train
on a 4-bit frozen base (their original settings need more than 32 GB). All
predictions keep the original precisions either way.

Both options reproduce our submitted best VQA by default: the `predict_bundle`
stage — a standard bf16 prediction with the `vqa_lora_bundle` adapter — routes
five question types through those probabilities, exactly as our best submission
was composed. Prediction needs no Cosmos; only retraining that adapter from
scratch does (Option C), so under Option B the stage falls back to the
shipped adapter and says so. Append `--no-bundle` to run the plain route
without it.

### Option C — advanced: also retrain the bundle adapter (external Cosmos step)

`vqa_lora_bundle` was trained on Cosmos-restyled **synthetic** overhead
keyframes plus photometric domain randomization. Reproducing it needs one step
outside this image — restyling with
[`nvidia/Cosmos-Transfer2.5-2B`](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)
(distilled edge variant; gated repo, separate install per its own README):

```bash
# 1. dump the 1,447 overhead train keyframes + per-frame restyle specs
docker run --rm --gpus all -v "$PWD:/pkg" -v team24-hf:/root/.cache/huggingface \
  team24-e2e --stages cosmos_prep

# 2. (external) restyle work/cosmos/specs/*.json with Cosmos-Transfer2.5-2B,
#    writing outputs to work/cosmos/restyled/   (~4 h on a 96 GB GPU)

# 3. rebuild the frame tree, overlay domain randomization, retrain (~6 h)
docker run --rm --gpus all -v "$PWD:/pkg" -v team24-hf:/root/.cache/huggingface \
  team24-e2e --stages train_bundle

# 4. redo the bundle prediction with the fresh adapter, then finish
rm -rf work/probs/bundle_probs.json.shards work/.done/predict_bundle
docker run --rm --gpus all --shm-size 8g \
  -v "$PWD:/pkg" -v team24-hf:/root/.cache/huggingface team24-e2e
```

## 4. Expected results

Target scores on the public test set: **VQA accuracy ≈ 84.7** (default run =
our submitted composition; `--no-bundle` lands ≈ 0.05 lower),
**caption S1 ≈ 29.9–30.0**.
GPU LoRA training and sampling are not bit-exact across hardware/library
versions, so scores land within a small band of these values
(±0.1–0.2 accuracy / S1 in our re-runs).

## 5. Models

| Model | Source | Revision |
|---|---|---|
| Qwen/Qwen3.5-9B (base for all adapters) | Hugging Face, public | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| nvidia/Cosmos-Transfer2.5-2B (restyles synthetic training frames only) | Hugging Face, public (gated) | `ce8440327c632d8313c3bde69db13b627ba5cae1` (distilled edge) |
| LoRA adapters (ours) | `artifacts/adapters/` (git LFS) | trained per this repo |

## 6. Code map

| File | Role |
|---|---|
| `run_e2e.py` | orchestrator (the only new file; everything below is the deployed competition code) |
| `preprocess_vqa.py` / `preprocess_caption.py` | frames + bbox drawing + crops + metadata (cv2 only) |
| `prepare_test_root.py` | mounts the official test package as a val-as-test layout |
| `run_qwen35_vqa.py` | VQA train / predict (letter-logit, LoRA, bf16/8-bit/4-bit) |
| `run_qwen35_caption.py` | caption SFT / DPO / greedy test prediction |
| `ctx_builder.py` | temporal-context + bbox-evidence training/eval data |
| `wsd.py` | transition mining + canonical-state Viterbi decoding |
| `fact_stitch.py`, `fact_stitch3/4/5.py` | caption fact correction waves v2–v5 |
| `cosmos_restyle_prep.py`, `dr_prep.py`, `*.sh` | optional bundle route + historical deploy pipelines |

Data-usage and rules compliance: see [COMPLIANCE.md](COMPLIANCE.md).

## Note on BDD_PC_5K

The challenge rules state the BDD_PC_5K subset is not used for evaluation.
However, the official public-test files do contain it (15,123 of the 19,624
VQA questions and 375 of the 459 caption scenarios are BDD), so the pipeline
answers **everything** in `WTS_VQA_PUBLIC_TEST.json` and captions every
scenario in the test package — the submission ID set must match the official
files exactly. Entries outside the scored ("internal"/"main" WTS) subset are
simply ignored by the evaluator. Training complies with the usage rule:
all models are fine-tuned **only** on the synthetic SynWTS set; no BDD or
real-world WTS data is used for training.
