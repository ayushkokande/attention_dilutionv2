#!/bin/bash
# Experiment 4: covariate-matched refusal direction (E-prof-2 validity check).
# Re-runs the diff-of-means recipe on harmful/harmless pools matched on
# verb-class and word-length. Resulting d_hat* is compared per layer to the
# canonical d_hat (results/qwen3-14b/refusal_direction/d_hat_all_layers.pt) to
# quantify how much of the original direction was driven by the surface
# confound the prof's mid-semester comment flagged.
#
# Cascade: canonical layer is L18 (Exp.2 causal sweep). compare_directions.py
# defaults --canonical-layer 18, so verdict re-evaluates against L18 d_hat.
#
# Target host: RunPod pod with 2x A100 SXM 80GB and a network volume mounted at
# /workspace. (NYU Greene slurm+singularity is no longer the active runner;
# git history retains the older variant if needed.)
#
# Outputs:
#   experiment_4/data/matched_harmful.jsonl
#   experiment_4/data/matched_harmless.jsonl
#   experiment_4/data/curation_stats.json
#   results/qwen3-14b/refusal_direction_matched/d_hat_all_layers.pt
#   results/qwen3-14b/refusal_direction_matched/meta.json
#   results/qwen3-14b/refusal_direction_matched/compare_to_orig.json
#   results/qwen3-14b/refusal_direction_matched/cos_per_layer.png
#
# Run:
#   bash experiment_4/sbatch_refusal_direction_matched.sh

set -euo pipefail

# RunPod default mount for the persistent network volume.
WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${REPO:-${WORKSPACE}/attention_dilutionv2}"

MODEL="Qwen/Qwen3-14B"
BATCH_SIZE=8
MATCHED_OUT="results/qwen3-14b/refusal_direction_matched"
CANONICAL_LAYER=18

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
LOG_OUT="${REPO}/logs/qwen3_14b_refdir_matched_${JOB_ID}.out"
LOG_ERR="${REPO}/logs/qwen3_14b_refdir_matched_${JOB_ID}.err"
exec > >(tee -a "${LOG_OUT}") 2> >(tee -a "${LOG_ERR}" >&2)

echo "=== preflight | REPO=$(pwd) ==="
ls -la

missing=0
for f in requirements.txt experiment_1/utils.py experiment_2/refusal_direction.py \
         experiment_4/curate_matched_pools.py experiment_4/compare_directions.py \
         results/qwen3-14b/refusal_direction/d_hat_all_layers.pt; do
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

echo "=== qwen3_14b_refdir_matched | job ${JOB_ID} ==="
echo "Repo : $(pwd)"
echo "CUDA : ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())'

echo
echo "=== step 1: curate matched pools (CPU only) ==="
python experiment_4/curate_matched_pools.py

echo
echo "=== step 2: refusal-direction forward pass on matched pools ==="
# Causal-ablation-n: reuse exp_2's held-out pool (AdvBench[480:504],
# n=24, default). Earlier runs passed --causal-ablation-n 0 which left
# meta.json:causal_sweep_records=[] and forced compare_directions.py to
# inherit canonical_layer=18 from the orig d_hat. Run the sweep on
# matched d_hat* as well so canonical_layer is independently validated
# against the same held-out causal pool used for orig.
time python experiment_2/refusal_direction.py \
    --model "${MODEL}" \
    --dtype bfloat16 \
    --batch-size ${BATCH_SIZE} \
    --harmful-file experiment_4/data/matched_harmful.jsonl \
    --harmless-file experiment_4/data/matched_harmless.jsonl \
    --output-dir ${MATCHED_OUT} \
    --causal-ablation-n 24

echo
echo "=== step 3: cosine similarity d_hat vs d_hat* per layer (canonical L${CANONICAL_LAYER}) ==="
python experiment_4/compare_directions.py \
    --orig-dir results/qwen3-14b/refusal_direction \
    --matched-dir ${MATCHED_OUT} \
    --canonical-layer ${CANONICAL_LAYER}

echo
echo "=== done ==="
ls -la ${MATCHED_OUT} || true
