# Training image: PyTorch for gfx1151 (AMD's wheels bundle the ROCm runtime) + upstream's trainer.
# Build from the repo root:  docker build -f finetune/Dockerfile -t laya-train:gfx1151 .
FROM ubuntu:26.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv ca-certificates libatomic1 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 HF_HOME=/cache/hf
# torch first, from AMD's gfx1151 index; everything else from PyPI
RUN pip install --no-cache-dir --index-url https://repo.amd.com/rocm/whl/gfx1151/ "torch==2.11.0+rocm7.13.0"
# laya: upstream's RLCD trainer (laya.train / laya-train), pinned -- the recipe is validated against it
RUN pip install --no-cache-dir "laya==0.4.8" pyarrow
WORKDIR /work
