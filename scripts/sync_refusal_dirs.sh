#!/bin/bash
# Sync gitignored d_hat artifacts from local to Greene.
set -euo pipefail

REMOTE="${REMOTE:-ak13124@greene.hpc.nyu.edu}"
DEST="${DEST:-/scratch/ak13124/attention_dilutionv2}"

ssh "${REMOTE}" "mkdir -p ${DEST}/results/qwen3-14b/refusal_direction ${DEST}/results/qwen3-14b/refusal_direction_matched"

rsync -av results/qwen3-14b/refusal_direction/d_hat_all_layers.pt "${REMOTE}:${DEST}/results/qwen3-14b/refusal_direction/"
rsync -av results/qwen3-14b/refusal_direction_matched/d_hat_all_layers.pt "${REMOTE}:${DEST}/results/qwen3-14b/refusal_direction_matched/"

echo "=== sync done ==="
