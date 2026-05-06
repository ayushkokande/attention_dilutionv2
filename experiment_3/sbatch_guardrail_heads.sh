#!/bin/bash
# Experiment 3: Guardrail Heads via Direct Logit Attribution onto d_hat at the
# canonical read-out layer (L18 for Qwen3-14B per Exp.2 causal sweep).
#
# Target host: RunPod pod with 2x A100 SXM 80GB and a network volume mounted at
# /workspace. (NYU Greene slurm+singularity is no longer the active runner;
# git history retains the older variant if needed.)
#
# Prereq: experiment_2 must have produced
#   results/qwen3-14b/refusal_direction/{d_hat_all_layers.pt, meta.json}
#
# Run:
#   bash experiment_3/sbatch_guardrail_heads.sh

set -euo pipefail

# RunPod default mount for the persistent network volume.
WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${REPO:-${WORKSPACE}/attention_dilutionv2}"

MODEL="Qwen/Qwen3-14B"
# Reuse the held-out pool that defined L_read in Exp.2 causal sweep
# (AdvBench[480:504], n=24). Disjoint from d_hat training and from exp_8 sweep eval.
N_PROMPTS=24
START_IDX=480
TOP_K=12
BATCH_SIZE=4

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
# Cap parallel downloads. Runpod CA-MTL-3 network drops connections when too
# many wheels stream at once; serializing is more reliable than fast.
export UV_CONCURRENT_DOWNLOADS=2
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

JOB_ID="${RUNPOD_POD_ID:-local}"
LOG_OUT="${REPO}/logs/qwen3_14b_guard_${JOB_ID}.out"
LOG_ERR="${REPO}/logs/qwen3_14b_guard_${JOB_ID}.err"
exec > >(tee -a "${LOG_OUT}") 2> >(tee -a "${LOG_ERR}" >&2)

echo "=== preflight | REPO=$(pwd) ==="
ls -la

missing=0
for f in requirements.txt experiment_1/utils.py experiment_3/guardrail_heads.py \
         results/qwen3-14b/refusal_direction/d_hat_all_layers.pt \
         results/qwen3-14b/refusal_direction/meta.json; do
  if [ ! -e "$f" ]; then
    echo "ERROR: missing $(pwd)/$f" >&2
    missing=1
  fi
done
if [ "$missing" = "1" ]; then
  echo >&2
  echo "Run experiment_2 (refusal_direction) first to produce d_hat_all_layers.pt," >&2
  echo "or sync the repo with rsync/git if scripts are missing." >&2
  exit 1
fi

# uv usually pre-installed on RunPod PyTorch images; install if absent.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if [ ! -d .venv ]; then
  # --seed installs pip into the venv so the pip fallback below works.
  uv venv --python 3.11 --seed .venv
fi
source .venv/bin/activate

# Try uv (fast). If uv aborts on a single-wheel network timeout (common on
# CA-MTL-3), retry up to 3x — uv resumes from cache. If it still fails,
# fall back to pip whose retry semantics are more forgiving on flaky links.
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

echo "=== qwen3_14b_guardrail_heads | job ${JOB_ID} ==="
echo "Repo : $(pwd)"
echo "CUDA : ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())'

time python experiment_3/guardrail_heads.py \
    --model "${MODEL}" \
    --dtype bfloat16 \
    --refusal-dir results/qwen3-14b/refusal_direction \
    --n-prompts ${N_PROMPTS} \
    --start-idx ${START_IDX} \
    --top-k ${TOP_K} \
    --batch-size ${BATCH_SIZE}

echo
echo "=== done ==="
ls -la results/qwen3-14b/guardrail_heads || true
