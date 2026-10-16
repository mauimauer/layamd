# Feasibility probe: lean gfx1151-only ROCm image that builds + runs HIP/hipBLAS code.
# Build from the repo root:  docker build -f bench/probes/Dockerfile -t laya-probe:gfx1151 .
ARG UBUNTU=26.04
FROM ubuntu:${UBUNTU} AS builder
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY docker/amdrocm.gpg /etc/apt/keyrings/amdrocm.gpg
RUN cat > /etc/apt/sources.list.d/amdrocm.sources <<'SRC'
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes
SRC
RUN apt-get update && apt-get install -y --no-install-recommends \
      amdrocm-core-dev10.0-gfx1151 amdrocm-blas-dev10.0 amdrocm-blas10.0-gfx1151 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/rocm/bin:$PATH
WORKDIR /build
COPY bench/probes/skel.cpp bench/probes/gemm.cpp ./
RUN hipcc -O3 --offload-arch=gfx1151 -std=c++17 skel.cpp -o skel -lhipblas && \
    hipcc -O3 --offload-arch=gfx1151 -std=c++17 gemm.cpp -o gemm -lhipblas

FROM ubuntu:26.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY docker/amdrocm.gpg /etc/apt/keyrings/amdrocm.gpg
RUN cat > /etc/apt/sources.list.d/amdrocm.sources <<'SRC'
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes
SRC
RUN apt-get update && apt-get install -y --no-install-recommends \
      amdrocm-core10.0-gfx1151 amdrocm-blas10.0-gfx1151 amdrocm-blas-host10.0 \
    && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/opt/rocm/lib
ENV PATH=/opt/rocm/bin:$PATH
COPY --from=builder /build/skel /build/gemm /usr/local/bin/
WORKDIR /w
CMD ["skel","512","1"]
