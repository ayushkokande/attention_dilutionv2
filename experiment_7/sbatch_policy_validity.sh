#!/bin/bash
#SBATCH --job-name=qwen3_14b_policy_validity
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ak13124/attention_dilutionv2/logs/qwen3_14b_policy_validity_%j.out
#SBATCH --error=/scratch/ak13124/attention_dilutionv2/logs/qwen3_14b_policy_validity_%j.err

# Experiment 7: policy-refusal vs harm-refusal validity for d_hat.
#
# Phase B (projection eval) produces the headline AUC(harm vs policy) number
# at canonical layer 36 (post Suraj-judge reconcile cascade; previous run was
# at L18 under the long-window judge — see experiment_2/refusal_direction.py
# docstring).
#
# Phase A (preflight refusal-rate gate) is OFF by default. Rationale: prior
# preflight (2026-05-04) returned policy_refusal_rate=0.44 vs gate 0.90 on
# SORRY-Bench cats 41-45 — Qwen3-14B does not reliably refuse policy-style
# prompts (per-cat: 41=0.70, 42=0.30, 43=0.10, 44=0.80, 45=0.30). This is a
# property of the model, not a pool bug. Preregistered fallback (memory
# project_experiment_7.md) applies the AUC rule unchanged with caveat. Low
# policy-side refusal is acceptable for this test: if d_hat projection stays
# low on policy-tone prompts irrespective of model compliance, that strengthens
# the harm-axis claim (d_hat is not a generic-decline direction).
#
# Set RUN_PREFLIGHT=1 to re-run phase A (e.g. after pool changes).
#
# Before GPU step (locally):
#   # Author experiment_7/data/manual_policy_refusal.json (50 prompts across
#   # 5 categories: medical, legal, copyright, identity, brand)
#   python experiment_7/build_policy_pools.py
#
# Outputs:
#   results/qwen3-14b/policy_validity/refusal_preflight.jsonl
#   results/qwen3-14b/policy_validity/refusal_preflight_summary.json
#   results/qwen3-14b/policy_validity/refusal_preflight_failures.json
#   results/qwen3-14b/policy_validity/per_prompt.jsonl
#   results/qwen3-14b/policy_validity/aucs.json
#   results/qwen3-14b/policy_validity/pool_means.json
#   results/qwen3-14b/policy_validity/boxplot.png
#
# Slurm usage:
#   sbatch experiment_7/sbatch_policy_validity.sh
#
# Runpod usage:
#   bash experiment_7/sbatch_policy_validity.sh
#   # common overrides:
#   #   export REPO=/workspace/attention_dilutionv2
#   #   export WORK_ROOT=/workspace
#   #   export HF_TOKEN=...
#   #   export BATCH_SIZE=4
#   #   export RUN_PREFLIGHT=1   # re-run phase A (default 0; off by design)

set -euo pipefail

echo "=== debug: pre-resolve env ==="
echo "PWD=$(pwd)"
echo "REPO_env=${REPO:-<unset>}"
echo "LOG_DIR_env=${LOG_DIR:-<unset>}"
echo "WORK_ROOT_env=${WORK_ROOT:-<unset>}"
echo "CACHE_ROOT_env=${CACHE_ROOT:-<unset>}"
echo "SCRATCH=${SCRATCH:-<unset>}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "==="

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -d "${SLURM_SUBMIT_DIR}" ]]; then
  REPO_DEFAULT="${SLURM_SUBMIT_DIR}"
else
  REPO_DEFAULT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
fi

REPO="${REPO:-${REPO_DEFAULT}}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LAYERS="${LAYERS:-36 28}"
ORIG_DIR="${ORIG_DIR:-results/qwen3-14b/refusal_direction}"
MATCHED_DIR="${MATCHED_DIR:-results/qwen3-14b/refusal_direction_matched}"
OUT_DIR="${OUT_DIR:-results/qwen3-14b/policy_validity}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"
PREFLIGHT_MAX_NEW="${PREFLIGHT_MAX_NEW:-128}"

WORK_ROOT="${WORK_ROOT:-${SCRATCH:-/workspace}}"
if [[ ! -d "${WORK_ROOT}" ]]; then
  WORK_ROOT="${REPO}"
fi

CACHE_ROOT="${CACHE_ROOT:-${WORK_ROOT}}"
LOG_DIR="${LOG_DIR:-${REPO}/logs}"
VENV_DIR="${VENV_DIR:-${REPO}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
USE_SINGULARITY="${USE_SINGULARITY:-auto}"

if [[ -n "${SCRATCH:-}" ]]; then
  SIF_DEFAULT="${SCRATCH}/ubuntu-20.04.3.sif"
  OVERLAY_DEFAULT="${SCRATCH}/overlay-25GB-500K.ext3:ro"
else
  SIF_DEFAULT=""
  OVERLAY_DEFAULT=""
fi
SIF="${SIF:-${SIF_DEFAULT}}"
OVERLAY="${OVERLAY:-${OVERLAY_DEFAULT}}"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/.cache/torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/.uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${CACHE_ROOT}/.uv_python}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${LOG_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

find_python() {
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    printf '%s\n' python
    return 0
  fi
  echo "ERROR: no python interpreter found." >&2
  exit 1
}

setup_runtime() {
  local py_bin="$1"

  if [[ -f /ext3/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /ext3/miniconda3/etc/profile.d/conda.sh
    export PATH=/ext3/miniconda3/bin:${PATH}
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    if command -v uv >/dev/null 2>&1; then
      uv venv --python "${py_bin}" "${VENV_DIR}"
    else
      "${py_bin}" -m venv "${VENV_DIR}"
    fi
  fi

  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"

  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python "${VENV_DIR}/bin/python" -r requirements.txt
    else
      python -m pip install --upgrade pip setuptools wheel
      python -m pip install -r requirements.txt
    fi
  fi
}

run_job() {
  local py_bin
  local missing
  local f
  local -a layer_args

  cd "${REPO}"
  py_bin="$(find_python)"

  echo "=== preflight | REPO=$(pwd) ==="

  missing=0
  for f in requirements.txt \
           experiment_1/utils.py \
           experiment_2/refusal_direction.py \
           experiment_7/preflight_refusal_rates.py \
           experiment_7/eval_policy_vs_harm.py \
           experiment_7/data/harm_refusal.jsonl \
           experiment_7/data/policy_refusal.jsonl \
           experiment_7/data/harmless.jsonl \
           "${ORIG_DIR}/d_hat_all_layers.pt" \
           "${MATCHED_DIR}/d_hat_all_layers.pt"; do
    if [[ ! -e "${f}" ]]; then
      echo "ERROR: missing $(pwd)/${f}" >&2
      missing=1
    fi
  done

  if [[ "${missing}" == "1" ]]; then
    echo >&2
    echo "Run locally first:" >&2
    echo "  # author experiment_7/data/manual_policy_refusal.json (50 rows)" >&2
    echo "  python experiment_7/build_policy_pools.py" >&2
    echo "Also sync refusal_direction artifacts from experiments 2 and 4." >&2
    exit 1
  fi

  setup_runtime "${py_bin}"
  read -r -a layer_args <<< "${LAYERS}"

  echo "=== qwen3_14b_policy_validity | job ${SLURM_JOB_ID:-local} ==="
  echo "CUDA : ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "WORK_ROOT : ${WORK_ROOT}"
  echo "HF_HOME   : ${HF_HOME}"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
  python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())'

  if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    echo "=== Phase A: preflight refusal-rate gate ==="
    time python experiment_7/preflight_refusal_rates.py \
      --model "${MODEL}" \
      --dtype "${DTYPE}" \
      --batch-size "${BATCH_SIZE}" \
      --max-new-tokens "${PREFLIGHT_MAX_NEW}" \
      --output-dir "${OUT_DIR}"
  else
    echo "=== Phase A: preflight skipped (RUN_PREFLIGHT=0) ==="
  fi

  echo "=== Phase B: projection eval ==="
  time python experiment_7/eval_policy_vs_harm.py \
    --model "${MODEL}" \
    --dtype "${DTYPE}" \
    --refusal-dirs "${ORIG_DIR}" "${MATCHED_DIR}" \
    --refusal-dir-names refusal_direction refusal_direction_matched \
    --layers "${layer_args[@]}" \
    --batch-size "${BATCH_SIZE}" \
    --output-dir "${OUT_DIR}"

  echo "=== done ==="
  ls -la "${OUT_DIR}" || true
}

if [[ "${USE_SINGULARITY}" == "auto" ]]; then
  if command -v singularity >/dev/null 2>&1 && [[ -n "${SIF}" ]] && [[ -f "${SIF}" ]]; then
    USE_SINGULARITY=1
  else
    USE_SINGULARITY=0
  fi
fi

export REPO MODEL DTYPE BATCH_SIZE LAYERS ORIG_DIR MATCHED_DIR OUT_DIR
export RUN_PREFLIGHT PREFLIGHT_MAX_NEW
export WORK_ROOT CACHE_ROOT LOG_DIR VENV_DIR PYTHON_BIN INSTALL_DEPS

if [[ "${USE_SINGULARITY}" == "1" ]]; then
  if [[ ! -f "${SIF}" ]]; then
    echo "ERROR: USE_SINGULARITY=1 but SIF not found at ${SIF}" >&2
    exit 1
  fi

  overlay_path="${OVERLAY%%:*}"
  singularity_args=(exec --nv --bind "${WORK_ROOT}")
  if [[ "${REPO}" != "${WORK_ROOT}"* ]]; then
    singularity_args+=(--bind "${REPO}")
  fi
  if [[ -n "${OVERLAY}" ]] && [[ -e "${overlay_path}" ]]; then
    singularity_args+=(--overlay "${OVERLAY}")
  fi

  export -f find_python
  export -f setup_runtime
  export -f run_job
  singularity "${singularity_args[@]}" "${SIF}" /bin/bash -lc "run_job"
else
  run_job
fi
