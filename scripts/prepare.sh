#!/usr/bin/env bash
# Idempotent first-run preparation:
#   1. Build the CUDA + Rust Docker image (if missing)
#   2. Download thinkingmachines/Inkling-2-Mini (if MODEL_DIR has no weights)
#   3. Fold DSpark weights into build/dspark-fold/{fold,lowrank16}.bin (unless DFLASH2_DIR is set)
#   4. Compile the sm_80 kernel pack (kpack) and ink-server / ink-tools release binaries
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/config.sh"

mkdir -p "$BUILD_DIR" "$FOLD_DIR" "$KPACK_DIR" "$BIN_DIR" "$CFG_DIR"

# 1. Docker image
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> Building Docker image $IMAGE (CUDA_IMAGE=$CUDA_IMAGE)..."
  docker build \
    --build-arg "CUDA_IMAGE=$CUDA_IMAGE" \
    -t "$IMAGE" \
    "$ROOT_DIR"
fi

# 2. Model checkpoint
if [ ! -f "$MODEL_DIR/model-00001-of-00001.safetensors" ]; then
  echo "==> Downloading $MODEL_REPO into $MODEL_DIR..."
  mkdir -p "$MODEL_DIR"
  if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  else
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_REPO', local_dir='$MODEL_DIR')" \
      || {
        echo "ERROR: $MODEL_DIR/model-00001-of-00001.safetensors not found and huggingface_hub is not installed." >&2
        echo "Install with: pip install huggingface-hub, or set MODEL_DIR in scripts/local.sh." >&2
        exit 1
      }
  fi
fi

# 3. DSpark folded weights (skip if user selected DFlash2)
if [ -z "$DFLASH2_DIR" ] && { [ ! -f "$FOLD_DIR/fold.bin" ] || [ ! -f "$FOLD_DIR/lowrank16.bin" ]; }; then
  echo "==> Folding DSpark weights -> $FOLD_DIR/{fold,lowrank16}.bin..."
  if python3 -c "import torch, safetensors" >/dev/null 2>&1; then
    python3 "$ROOT_DIR/scripts/fold-dspark.py" --model-dir "$MODEL_DIR" --out-dir "$FOLD_DIR"
  else
    echo "ERROR: Python packages 'torch' and 'safetensors' are needed once to fold DSpark weights." >&2
    echo "Install in a venv (pip install torch safetensors) and re-run ./start.sh," >&2
    echo "or run: python3 scripts/fold-dspark.py --model-dir '$MODEL_DIR' --out-dir '$FOLD_DIR'" >&2
    exit 1
  fi
fi

# 4. Kernel pack (sm_80 cubins) + ink-server binary
if [ ! -f "$KPACK_DIR/manifest.json" ] || [ ! -f "$KPACK_DIR/pack_engine.cubin" ] || [ ! -f "$KPACK_DIR/pack_drafter.cubin" ]; then
  echo "==> Compiling CUDA sm_80 kernel pack into $KPACK_DIR..."
  docker run --rm --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    "$IMAGE" \
    bash -lc "engine/kernels/pack/build.sh /workspace/build/kpack"
fi

if [ ! -x "$BIN_DIR/ink-server" ]; then
  echo "==> Building ink-server and ink-tools (release, locked)..."
  mkdir -p "$BUILD_DIR/cargo-home" "$BUILD_DIR/target"
  docker run --rm --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e CARGO_HOME=/workspace/build/cargo-home \
    -e CARGO_TARGET_DIR=/workspace/build/target \
    -e INK_KPACK_DIR=/workspace/build/kpack \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace/engine/rust \
    "$IMAGE" \
    bash -lc '
      # Regenerate ink-kernels bindings if local nvcc version produced a different kpack id
      if ! python3 -I /workspace/engine/kernels/pack/check_fresh.py >/dev/null 2>&1 || \
         ! grep -q "$(python3 -c "import json; print(json.load(open(\"/workspace/build/kpack/manifest.json\"))[\"kpack_id\"])")" crates/ink-kernels/src/gen.rs; then
        echo "==> Syncing ink-kernels bindings with local kpack..."
        cargo run --release --locked -p xtask -- gen-kernels /workspace/build/kpack /workspace/engine/rust/crates/ink-kernels
      fi
      cargo build --release --locked -p ink-server -p ink-tools
      cp /workspace/build/target/release/ink-server /workspace/build/bin/ink-server
      for b in ink-bench ink-inv ink-spec ink-gold ink-load ink-tune; do
        [ -f "/workspace/build/target/release/$b" ] && cp "/workspace/build/target/release/$b" "/workspace/build/bin/$b" || true
      done
    '
  echo "==> Built $BIN_DIR/ink-server"
fi
