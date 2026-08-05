# data/ — datasets required by the pipeline

Place (or symlink) the following three directories here before running
`code/run_e2e.py`. Nothing in this folder ships with the package.

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

Sources
- SynWTS: `huggingface_hub.snapshot_download("mlcglab/synwts", repo_type="dataset")`
  (~2.5 GB, 846 videos) — the download root is the `synwts/` folder above.
- Public test + task package: distributed by the AI City Challenge organizers
  (Track 2). `WTS_DATASET_PUBLIC_TEST` is ~21 GB (664 videos).
