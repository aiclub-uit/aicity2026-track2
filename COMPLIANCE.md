# Compliance statement — AI City Challenge 2026 Track 2, Team 24 (UIT–Kitchen)

## Training data

All models are fine-tuned **only** on the synthetic SynWTS set
(`mlcglab/synwts`, revision `9d6e7abce9a906e3e9bb6a2941dcce8b128654cf`).
No real-world WTS training/validation videos, no BDD_PC_5K data, and no models
pre-trained on the WTS dataset are used anywhere in training. Model selection
and ablations used the SynWTS val split.

## Test data usage (inference only)

- Test videos, captions metadata, and the organizer-provided test bbox archives
  (`WTS_DATASET_PUBLIC_TEST_BBOX.zip`) are consumed **at inference only**, as
  distributed for the task.
- No test images, labels, or annotations enter any training loop. No manual
  annotation of test data and no manual rewriting of test outputs was
  performed; every post-processing step is automated code in this repository.
- One robustness hyperparameter is disclosed for transparency: the
  context-model training applies bbox dropout `P_BBOX_DROP = 0.30`
  (`code/ctx_builder.py`). The value was motivated by the observed
  *availability rate* of the organizer-provided test bbox input files
  (~82.6% of phases have a bbox file) so that "bbox unavailable" is
  in-distribution at inference. This statistic concerns the presence of
  organizer-shipped input files only — no test labels or image content —
  and retraining reproduces the submitted system from SynWTS alone, since
  the value is a fixed constant in code.

## Models

| Model | Source | Revision |
|---|---|---|
| Qwen/Qwen3.5-9B (base for all adapters) | Hugging Face, public | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| LoRA adapters (ours) | `artifacts/adapters/` (git LFS) | trained per this repo |

Python dependencies are pinned in the `Dockerfile`.

## Exact submission path

`code/run_e2e.py` (stages `prep_test … stitch`) is the complete path from raw
data to `submissions/subtask2_vqa.json` + `submissions/subtask1_captioning.json`.

Included in the repo but **not part of the submitted system's output path**:

- `harmonize_attrs.py` (cross-phase attribute harmonization) — an experiment
  that was never part of the submitted system; the file has been removed from
  the package. Its only references are inside the fact-stitch `cmd_val`
  validation helpers, which the pipeline never invokes (they also require the
  unshipped local `eval/` module).
- The MBR best-of-N prediction path (`QWEN35_PRED_NCAND > 1`) — the pipeline
  uses greedy decoding (default).
- The Cosmos-restyle "bundle" route (`--with-bundle`, `cosmos_restyle_prep.py`,
  `dr_prep.py`) — off by default; the default pipeline reproduces the
  submitted scores without it.
- `*.sh` scripts — historical deploy pipelines kept for reference.

## BDD_PC_5K

See the note in `README.md`: the submission answers every entry in the official
test files (77% of which are BDD_PC_5K); entries outside the scored subset are
ignored by the evaluator. No BDD data is used in training.
