#!/bin/bash
# Experiment 8: two-arm context-length sweep (binary refusal rate vs N).
#   Arm A (baseline): refusal vector intact.
#   Arm B (ablated):  refusal vector projected out at every block.
# Each (arm, L) cell is written as soon as it finishes; summary.json is updated
# incrementally so a walltime kill mid-sweep still leaves usable data.
#
# Prereq: experiment_2/sbatch_refusal_direction.sh must have already produced
#   results/qwen3-14b/refusal_direction/d_hat_all_layers.pt
#
# Target host: RunPod pod with 2x A100 SXM 80GB and a network volume mounted at
# /workspace. (NYU Greene slurm+singularity is no longer the active runner;
# git history retains the older variant if needed.)
#
# Run:
#   bash experiment_8/sbatch_context_sweep.sh

set -euo pipefail

# RunPod default mount for the persistent network volume.
WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${REPO:-${WORKSPACE}/attention_dilutionv2}"

MODEL="Qwen/Qwen3-14B"
# Ablation layer: canonical L36 (Suraj 24-token / 26-phrase causal judge,
# exp_2 reconcile rerun — see experiment_2/refusal_direction.py docstring;
# previous 128-token judge picked L18). Default in context_sweep.py reads
# meta.json:default_layer which is now 36, but pin explicitly to make the
# cascade auditable in the run log.
ABLATION_LAYER=36
# Sweep grid (INVARIANTS.md merged grid for exp_8/9/10). 32768 is Qwen3-14B's
# native RoPE window; past that the model degrades without YaRN and the
# refusal classifier reads noise as "complied", which would spuriously
# inflate the dilution effect. Stay <=32K unless you also flip on
# rope_scaling=yarn in context_sweep.py and treat those cells as a separate
# "extrapolation regime" plot.
LENGTHS="0 128 512 1024 2048 4096 8192 16384 32768"
HARMFUL_N=100
MAX_NEW=256
SEED=0

mkdir -p "${REPO}/logs"

cd "${REPO}"

# Persist HF and torch caches on the network volume so model weights survive
# pod restarts. ~28GB for Qwen3-14B bf16; budget is fine on 150-200GB volume.
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${WORKSPACE}/.cache/torch"
export UV_CACHE_DIR="${WORKSPACE}/.uv_cache"
export UV_PYTHON_INSTALL_DIR="${WORKSPACE}/.uv_python"
export UV_HTTP_TIMEOUT=1800
export UV_CONCURRENT_DOWNLOADS=2
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

JOB_ID="${RUNPOD_POD_ID:-local}"
LOG_OUT="${REPO}/logs/qwen3_14b_sweep_${JOB_ID}.out"
LOG_ERR="${REPO}/logs/qwen3_14b_sweep_${JOB_ID}.err"
exec > >(tee -a "${LOG_OUT}") 2> >(tee -a "${LOG_ERR}" >&2)

echo "=== preflight | REPO=$(pwd) ==="
ls -la

missing=0
for f in requirements.txt experiment_1/utils.py experiment_8/context_sweep.py \
         results/qwen3-14b/refusal_direction/d_hat_all_layers.pt; do
  if [ ! -e "$f" ]; then
    echo "ERROR: missing $(pwd)/$f" >&2
    missing=1
  fi
done
if [ "$missing" = "1" ]; then
  echo >&2
  echo "Run experiment_2/sbatch_refusal_direction.sh first to produce d_hat_all_layers.pt," >&2
  echo "or sync the repo with git/rsync if scripts are missing." >&2
  exit 1
fi

# uv usually pre-installed on RunPod PyTorch images; install if absent.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [ ! -d .venv ]; then
  uv venv --python 3.11 --seed .venv
fi
source .venv/bin/activate

install_ok=0
for attempt in 1 2 3; do
  if uv pip install --python .venv/bin/python -r requirements.txt; then
    install_ok=1
    break
  fi
  echo "uv install attempt ${attempt} failed, retrying..." >&2
  sleep 5
done
if [ "${install_ok}" = "0" ]; then
  echo "uv install failed 3x, falling back to pip..." >&2
  python -m pip install --upgrade pip
  pip install --retries 10 --timeout 600 -r requirements.txt
fi

echo "=== qwen3_14b_context_sweep | job ${JOB_ID} ==="
echo "Repo : $(pwd)"
echo "CUDA : ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())'

time python experiment_8/context_sweep.py \
    --model "${MODEL}" \
    --dtype bfloat16 \
    --refusal-dir results/qwen3-14b/refusal_direction \
    --ablation-layer ${ABLATION_LAYER} \
    --lengths ${LENGTHS} \
    --harmful-n ${HARMFUL_N} \
    --max-new-tokens ${MAX_NEW} \
    --temperature 0.0 \
    --seed ${SEED} \
    --arms both

echo
echo "=== done ==="
ls -la results/qwen3-14b/context_sweep || true
