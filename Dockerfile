# GPU environment for the end-to-end reproduction (code/run_e2e.py).
#
#   docker build -t team24-e2e .
#   docker run --rm --gpus all --shm-size 8g \
#     -v "$PWD:/pkg" -v team24-hf:/root/.cache/huggingface team24-e2e
#
# The package directory is bind-mounted at /pkg with data/ populated
# (see data/README.md); outputs land in /pkg/work and /pkg/submissions.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# java is required by pycocoevalcap's METEOR scorer (caption DPO stage)
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# pinned to the versions the pipeline was verified with
RUN pip install --no-cache-dir \
    transformers==5.14.1 peft==0.20.0 bitsandbytes==0.50.0 accelerate==1.14.0 \
    datasets==5.0.1 trl==1.9.2 qwen-vl-utils==0.0.14 pycocoevalcap==1.2 \
    opencv-python-headless==5.0.0.93 tqdm==4.68.0

# the training scripts only setdefault HF_HOME — pin it so the team24-hf
# volume (mounted at /root/.cache/huggingface) actually caches the checkpoint
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /pkg
ENTRYPOINT ["python", "code/run_e2e.py"]
