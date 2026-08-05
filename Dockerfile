# GPU environment for the end-to-end reproduction (code/run_e2e.py).
#
#   docker build -t team24-e2e .
#   docker run --rm --gpus all --shm-size 8g -v "$PWD:/pkg" team24-e2e
#
# The package directory is bind-mounted at /pkg with data/ populated
# (see data/README.md); outputs land in /pkg/work and /pkg/submissions.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir \
    "transformers>=5.14" "peft>=0.20" "bitsandbytes>=0.50" "accelerate>=1.14" \
    "datasets>=5.0" "trl>=1.6" qwen-vl-utils opencv-python-headless tqdm

# the training scripts only setdefault HF_HOME — pin it so the team24-hf
# volume (mounted at /root/.cache/huggingface) actually caches the checkpoint
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /pkg
ENTRYPOINT ["python", "code/run_e2e.py"]
