#!/usr/bin/env bash
# Kaggle shortcut for scripts/verify.sh (P100, T4 or 2×T4 — pick the accelerator in the notebook).
#
# In a Kaggle notebook (Internet ON, GPU accelerator selected):
#   !git clone https://github.com/<you>/<repo>.git && cd <repo> && bash scripts/kaggle_verify.sh weight-codec
#
# Required: LWS_MODEL_DIR pointing to the Gemma 2 BF16 checkpoint
# (e.g. a Kaggle Models input under /kaggle/input/...).
# On 2×T4, LWS_DEVICE=1 measures the second card (same GPU slug, so same verdict file).
set -euo pipefail
exec bash "$(dirname "$0")/verify.sh" "$@"
