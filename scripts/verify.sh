#!/usr/bin/env bash
# Run the onp-spec proof on ANY NVIDIA GPU with compute capability >= 6.0
# (GTX 10xx ... RTX 50xx, P100, V100, T4, A100, L4, H100, B200), on Linux or WSL2:
# a home PC, Kaggle, Colab, or a rented cloud box.
#
#   export LWS_MODEL_DIR=/path/to/gemma-2-2b-it     # BF16 safetensors
#   export LWS_DEVICE=0                              # optional: which GPU (nvidia-smi index)
#   bash scripts/verify.sh <feature>
#
# Proof is per GPU: run it on every GPU you want a verdict for.
set -euo pipefail
FEATURE="${1:?usage: verify.sh <feature>}"
: "${LWS_MODEL_DIR:?export LWS_MODEL_DIR=/path/to/gemma-2-2b-it (Gemma 2 BF16 safetensors)}"
GPU_INDEX="${LWS_DEVICE:-0}"
# Only the chosen GPU is visible from here on, so Python always sees it as device 0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_INDEX" LWS_DEVICE=0

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver (WSL2: the Windows driver)"; exit 1; }
q() { nvidia-smi -i "$GPU_INDEX" --query-gpu="$1" --format=csv,noheader | head -1; }
NAME=$(q name); CC=$(q compute_cap); DRIVER=$(q driver_version); VRAM=$(q memory.total)
MAJOR=${CC%.*}; MINOR=${CC#*.}
SLUG="$(echo "$NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g')-sm${MAJOR}${MINOR}"
echo "GPU ${GPU_INDEX}: ${NAME} (sm_${MAJOR}${MINOR}, ${VRAM}, driver ${DRIVER}) → slug ${SLUG}"
if (( MAJOR < 6 )); then
  echo "compute capability ${CC} < 6.0 is not supported (see fused-decode-gemv spec, Out of scope)"; exit 1
fi

# 1. A PyTorch build that really runs on THIS architecture.
#    sm_60/70 need a CUDA 12 build that still ships them (cu126);
#    Blackwell (sm_100/120) needs cu128 or newer; the rest works with any current CUDA 12 build.
smoke() { python -c "import torch; assert (torch.ones(8, device='cuda') * 2).sum().item() == 16" 2>/dev/null; }
if ! smoke; then
  TV=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
  if   (( MAJOR < 7 || (MAJOR == 7 && MINOR == 0) )); then INDICES="cu126"
  elif (( MAJOR >= 10 ));                                then INDICES="cu129 cu128"
  else                                                        INDICES="cu126 cu128"
  fi
  ok=0
  for IDX in $INDICES; do
    URL="https://download.pytorch.org/whl/${IDX}"
    echo "torch ${TV:-<none>} cannot run on sm_${MAJOR}${MINOR} — trying the ${IDX} build"
    if [[ -n "$TV" ]] && pip install -q --force-reinstall --no-deps "torch==${TV}" --index-url "$URL" && smoke; then ok=1; break; fi
    echo "  no working torch==${TV} on ${IDX}; trying the newest torch on ${IDX}"
    if pip install -q --force-reinstall torch --index-url "$URL" && smoke; then ok=1; break; fi
  done
  (( ok )) || { echo "no PyTorch build found for sm_${MAJOR}${MINOR}: check https://pytorch.org/get-started/locally/"; exit 1; }
fi
python -c "import torch; print('torch', torch.__version__, 'OK on sm_${MAJOR}${MINOR}', torch.cuda.get_arch_list())"

# 2. Node >= 18 for the onp-spec engine (verify + audit).
if ! node -e "process.exit(+process.versions.node.split('.')[0] < 18)" 2>/dev/null; then
  case "$(uname -m)" in x86_64) NARCH=x64 ;; aarch64|arm64) NARCH=arm64 ;; *) echo "unsupported CPU arch for Node download"; exit 1 ;; esac
  NODE_DIR="${TMPDIR:-/tmp}/node-v22.11.0-linux-${NARCH}"
  [[ -d "$NODE_DIR" ]] || curl -fsSL "https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-${NARCH}.tar.xz" | tar -xJ -C "${TMPDIR:-/tmp}"
  export PATH="${NODE_DIR}/bin:$PATH"
fi

# 3. Project + CuPy (NVRTC compiles the .cu kernels for this GPU at runtime).
pip install -q -e ".[dev]"

# 4. Device doctor (once lws.device exists — task T-011): explains unsupported setups
#    (NVRTC too old for this arch, not enough VRAM for a feature, ...) before any test.
if python -c "import lws.device" 2>/dev/null; then
  python -m lws.device || { echo "device doctor refused this setup (see message above)"; exit 1; }
fi

# 5. The proof: tests run HERE, on this GPU, so GPU/model tests are not skipped.
ENGINE=.claude/skills/onp-spec-driven/scripts/onp-spec.mjs
node "$ENGINE" verify "$FEATURE"
node "$ENGINE" audit --ci || true   # print everything; the exit code is read below
node "$ENGINE" audit --ci >/dev/null && echo "AUDIT: exit 0 (aligned)" || echo "AUDIT: exit 1 (see findings above)"

# 6. Keep a per-GPU copy: the engine's file is overwritten by the next GPU's run.
mkdir -p reports/verification
if [[ -f ".spec/verification/${FEATURE}.json" ]]; then
  cp ".spec/verification/${FEATURE}.json" "reports/verification/${FEATURE}-${SLUG}.json"
fi
echo "Commit .spec/verification/${FEATURE}.json and reports/ back to the repo (proof on ${NAME}, $(date -u +%FT%TZ))."
