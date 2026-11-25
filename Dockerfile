# Standalone build & runtime image for inkling-cmp-engine (CUDA 13.x + Rust 1.98).
# Override CUDA_IMAGE if your host driver / CUDA toolkit uses another 13.x patch version.
ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
ARG RUST_VERSION=1.98.0

FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      pkg-config \
      python3 \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://sh.rustup.rs | sh -s -- \
      -y --profile minimal --default-toolchain "${RUST_VERSION}" \
      --component clippy,rustfmt \
    && chmod -R a+rX /usr/local/rustup /usr/local/cargo

RUN rustc --version && cargo --version && nvcc --version

WORKDIR /workspace
