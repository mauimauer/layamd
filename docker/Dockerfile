# Laya decision engine for AMD Strix Halo (gfx1151).
# Stage 1 compiles the HIP engine; stage 2 ships only the gfx1151 ROCm runtime.
# syntax=docker/dockerfile:1
ARG ROCM_REPO=https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/

FROM ubuntu:26.04 AS builder
ARG ROCM_REPO
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY docker/amdrocm.gpg /etc/apt/keyrings/amdrocm.gpg
RUN printf 'Types: deb\nURIs: %s\nSuites: stable\nComponents: main\nArchitectures: amd64\nSigned-By: /etc/apt/keyrings/amdrocm.gpg\nEnabled: yes\n' "$ROCM_REPO" > /etc/apt/sources.list.d/amdrocm.sources
RUN apt-get update && apt-get install -y --no-install-recommends \
      amdrocm-core-dev10.0-gfx1151 amdrocm-blas-dev10.0 amdrocm-blas10.0-gfx1151 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/rocm/bin:$PATH
WORKDIR /build
COPY src/ ./src/
RUN hipcc -O3 --offload-arch=gfx1151 -std=c++17 -fPIC -shared \
      src/laya.cpp src/kernels.hip src/engine.hip src/wgemm.hip -o liblaya.so -lhipblas -lhipblaslt && \
    hipcc -O3 --offload-arch=gfx1151 -std=c++17 \
      src/test_parity.cpp src/laya.cpp src/kernels.hip src/engine.hip src/wgemm.hip -o laya-parity -lhipblas -lhipblaslt && \
    hipcc -O3 --offload-arch=gfx1151 -std=c++17 \
      src/bench_engine.cpp src/laya.cpp src/kernels.hip src/engine.hip src/wgemm.hip -o laya-bench -lhipblas -lhipblaslt

FROM ubuntu:26.04 AS runtime
ARG ROCM_REPO
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY docker/amdrocm.gpg /etc/apt/keyrings/amdrocm.gpg
RUN printf 'Types: deb\nURIs: %s\nSuites: stable\nComponents: main\nArchitectures: amd64\nSigned-By: /etc/apt/keyrings/amdrocm.gpg\nEnabled: yes\n' "$ROCM_REPO" > /etc/apt/sources.list.d/amdrocm.sources
RUN apt-get update && apt-get install -y --no-install-recommends \
      amdrocm-core10.0-gfx1151 amdrocm-blas10.0-gfx1151 amdrocm-blas-host10.0 \
      python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir numpy tokenizers
ENV PATH=/venv/bin:/opt/rocm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    LAYA_MODEL_DIR=/weights/laya-en \
    LAYA_LIB=/usr/local/lib/liblaya.so \
    LAYA_PORT=8080
COPY --from=builder /build/liblaya.so /usr/local/lib/liblaya.so
COPY --from=builder /build/laya-parity /build/laya-bench /usr/local/bin/
COPY server/ /app/
WORKDIR /app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3).status==200 else 1)"
CMD ["python3","/app/app.py"]
