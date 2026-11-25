#!/usr/bin/env bash
# Shared configuration loader for start.sh, stop.sh, status.sh, and scripts/prepare.sh.
# Load order (lowest to highest priority): defaults < scripts/local.sh < caller environment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Snapshot caller env overrides before sourcing local.sh.
_ENV_GPUS="${GPUS-}"
_ENV_BASE_PORT="${BASE_PORT-}"
_ENV_MODEL_DIR="${MODEL_DIR-}"
_ENV_HF_CACHE="${HF_CACHE-}"
_ENV_SLOTS="${SLOTS-}"
_ENV_MAX_RUNNING="${MAX_RUNNING-}"
_ENV_PAGES="${PAGES-}"
_ENV_KV_FP8="${KV_FP8-}"
_ENV_DCTX="${DCTX-}"
_ENV_CPUS="${CPUS-}"
_ENV_DFLASH2_DIR="${DFLASH2_DIR-}"
_ENV_EXTRA_ARGS="${EXTRA_ARGS-}"
_ENV_IMAGE="${IMAGE-}"
_ENV_CUDA_IMAGE="${CUDA_IMAGE-}"

if [ -f "$ROOT_DIR/scripts/local.sh" ]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/scripts/local.sh"
fi

# Re-apply explicit caller env overrides.
[ -n "$_ENV_GPUS" ]        && GPUS="$_ENV_GPUS"
[ -n "$_ENV_BASE_PORT" ]   && BASE_PORT="$_ENV_BASE_PORT"
[ -n "$_ENV_MODEL_DIR" ]   && MODEL_DIR="$_ENV_MODEL_DIR"
[ -n "$_ENV_HF_CACHE" ]    && HF_CACHE="$_ENV_HF_CACHE"
[ -n "$_ENV_SLOTS" ]       && SLOTS="$_ENV_SLOTS"
[ -n "$_ENV_MAX_RUNNING" ] && MAX_RUNNING="$_ENV_MAX_RUNNING"
[ -n "$_ENV_PAGES" ]       && PAGES="$_ENV_PAGES"
[ -n "$_ENV_KV_FP8" ]      && KV_FP8="$_ENV_KV_FP8"
[ -n "$_ENV_DCTX" ]        && DCTX="$_ENV_DCTX"
[ -n "$_ENV_CPUS" ]        && CPUS="$_ENV_CPUS"
[ -n "$_ENV_DFLASH2_DIR" ] && DFLASH2_DIR="$_ENV_DFLASH2_DIR"
[ -n "$_ENV_EXTRA_ARGS" ]  && EXTRA_ARGS="$_ENV_EXTRA_ARGS"
[ -n "$_ENV_IMAGE" ]       && IMAGE="$_ENV_IMAGE"
[ -n "$_ENV_CUDA_IMAGE" ]  && CUDA_IMAGE="$_ENV_CUDA_IMAGE"

# Defaults
GPUS="${GPUS:-0}"
BASE_PORT="${BASE_PORT:-8400}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MODEL_REPO="${MODEL_REPO:-thinkingmachines/Inkling-2-Mini}"
MODEL_DIR="${MODEL_DIR:-$HF_CACHE/Inkling-2-Mini}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
FOLD_DIR="${FOLD_DIR:-$BUILD_DIR/dspark-fold}"
KPACK_DIR="${KPACK_DIR:-$BUILD_DIR/kpack}"
BIN_DIR="${BIN_DIR:-$BUILD_DIR/bin}"
CFG_DIR="${CFG_DIR:-$BUILD_DIR/cfg}"

SLOTS="${SLOTS:-8}"
MAX_RUNNING="${MAX_RUNNING:-8}"
KV_FP8="${KV_FP8:-0}"
if [ -z "${PAGES:-}" ]; then
  if [ "$KV_FP8" = "1" ]; then
    PAGES=1008
  else
    PAGES=504
  fi
fi
DCTX="${DCTX:-102400}"
CPUS="${CPUS:-6}"
DFLASH2_DIR="${DFLASH2_DIR:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

IMAGE="${IMAGE:-inkling-cmp-engine:v1.0.0}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:13.0.0-devel-ubuntu24.04}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-ink-engine-g}"
