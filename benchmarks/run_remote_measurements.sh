#!/usr/bin/env bash
# Run the speedup validation on the target GPU and collect the results.
#
# Usage (from a machine that can reach the GPU host):
#   benchmarks/run_remote_measurements.sh <ssh-host> [remote-dir]
#
# The script copies the working tree, runs the tests and the benchmarks on the
# remote host, and copies the logs back to ./benchmark-results/.
set -uo pipefail

HOST="${1:?usage: run_remote_measurements.sh <ssh-host> [remote-dir]}"
REMOTE_DIR="${2:-~/qwen3-tts-bench}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$LOCAL_DIR/benchmark-results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

echo "== syncing $LOCAL_DIR -> $HOST:$REMOTE_DIR"
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'benchmark-results' --exclude '*.wav' \
  "$LOCAL_DIR/" "$HOST:$REMOTE_DIR/" || exit 1

echo "== environment"
ssh "$HOST" "cd $REMOTE_DIR && {
  hostname; date;
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader;
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader;
  nproc; free -g | head -2;
  command -v uv; command -v python3; python3 -V;
  ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS* 2>/dev/null;
} " 2>&1 | tee "$OUT_DIR/00-environment.txt"

echo "== Python environment"
ssh "$HOST" "cd $REMOTE_DIR && {
  if command -v uv >/dev/null; then
    uv venv .venv --python 3.12 >/dev/null 2>&1 || true
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
    uv pip install --python .venv/bin/python 'transformers==4.57.3' 'accelerate==1.12.0' librosa einops soundfile sox onnxruntime torchaudio --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple 2>&1 | tail -3
  fi
  .venv/bin/python -c 'import torch, transformers; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"avail\", torch.cuda.is_available()); print(\"transformers\", transformers.__version__)'
  .venv/bin/python -c 'import triton; print(\"triton\", triton.__version__)' 2>&1 | tail -1
}" 2>&1 | tee "$OUT_DIR/01-python-env.txt"

echo "== tests"
ssh "$HOST" "cd $REMOTE_DIR && .venv/bin/python -m unittest discover -s tests -v" 2>&1 \
  | tee "$OUT_DIR/02-tests.txt"

echo "== benchmark: short, reused prompt"
ssh "$HOST" "cd $REMOTE_DIR && .venv/bin/python benchmarks/bench_generation.py --label short --runs 10 --profile" 2>&1 \
  | tee "$OUT_DIR/10-short.txt"

echo "== benchmark: long, reused prompt"
ssh "$HOST" "cd $REMOTE_DIR && .venv/bin/python benchmarks/bench_generation.py --label long --runs 10 --profile" 2>&1 \
  | tee "$OUT_DIR/11-long.txt"

echo "== benchmark: long, prompt rebuilt every run (ICL)"
ssh "$HOST" "cd $REMOTE_DIR && .venv/bin/python benchmarks/bench_generation.py --label long --runs 5 --rebuild-prompt" 2>&1 \
  | tee "$OUT_DIR/12-long-rebuild-icl.txt"

echo "== benchmark: long, prompt rebuilt every run (x-vector only)"
ssh "$HOST" "cd $REMOTE_DIR && .venv/bin/python benchmarks/bench_generation.py --label long --runs 5 --rebuild-prompt --x-vector-only" 2>&1 \
  | tee "$OUT_DIR/13-long-rebuild-xvec.txt"

echo "== done: $OUT_DIR"
