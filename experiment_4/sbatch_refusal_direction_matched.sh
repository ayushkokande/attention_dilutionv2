#!/bin/bash
#SBATCH --job-name=qwen3_14b_refdir_matched
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ak13124/attention_dilutionv2/logs/qwen3_14b_refdir_matched_%j.out
#SBATCH --error=/scratch/ak13124/attention_dilutionv2/logs/qwen3_14b_refdir_matched_%j.err

# Experiment 4: covariate-matched refusal direction (E-prof-2).
# Re-runs the diff-of-means recipe on harmful/harmless pools matched on
# verb-class and word-length. Resulting d̂* is compared per layer to the
# original d̂ (results/qwen3-14b/refusal_direction/d_hat_all_layers.pt) to
# quantify how much of the original direction was driven by the surface
# confound the prof's mid-semester comment flagged.
#
# Outputs:
#   experiment_4/data/matched_harmful.jsonl
#   experiment_4/data/matched_harmless.jsonl
#   experiment_4/data/curation_stats.json
#   results/qwen3-14b/refusal_direction_matched/d_hat_all_layers.pt
#   results/qwen3-14b/refusal_direction_matched/meta.json
#
# Submit:
#   sbatch experiment_4/sbatch_refusal_direction_matched.sh

set -euo pipefail

SCRATCH="/scratch/ak13124"
SIF="${SCRATCH}/ubuntu-20.04.3.sif"
OVERLAY="${SCRATCH}/overlay-25GB-500K.ext3:ro"
REPO="${SCRATCH}/attention_dilutionv2"

MODEL="Qwen/Qwen3-14B"
BATCH_SIZE=8
MATCHED_OUT="results/qwen3-14b/refusal_direction_matched"

mkdir -p "${REPO}/logs"

singularity exec --bind "${SCRATCH}" --nv \
  --overlay "${OVERLAY}" \
  "${SIF}" \
  /bin/bash -c "
set -euo pipefail

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
export PATH=${SCRATCH}/tools/bin:\$PATH
export UV_CACHE_DIR=${SCRATCH}/.uv_cache

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

export UV_PYTHON_INSTALL_DIR=${SCRATCH}/.uv_python
export HF_HOME=${SCRATCH}/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=\${HF_HOME}/hub
export TRANSFORMERS_CACHE=\${HF_HOME}/hub
export TORCH_HOME=${SCRATCH}/.cache/torch
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export PYTHONUNBUFFERED=1
mkdir -p \"\${HF_HOME}\" \"\${TORCH_HOME}\"

cd \"${REPO}\"

echo \"=== preflight | REPO=\$(pwd) ===\"
ls -la

missing=0
for f in requirements.txt experiment_1/utils.py experiment_2/refusal_direction.py \
         experiment_4/curate_matched_pools.py \
         results/qwen3-14b/refusal_direction/d_hat_all_layers.pt; do
  if [ ! -e \"\$f\" ]; then
    echo \"ERROR: missing \$(pwd)/\$f\" >&2
    missing=1
  fi
done
if [ \"\$missing\" = \"1\" ]; then
  echo >&2
  echo \"Sync the full repo to \${REPO} (and run experiment_2/sbatch_refusal_direction.sh first)\" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  uv venv --python 3.11 .venv
fi
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt

echo \"=== qwen3_14b_refdir_matched | job \${SLURM_JOB_ID:-local} ===\"
echo \"Repo : \$(pwd)\"
echo \"CUDA : \${CUDA_VISIBLE_DEVICES:-unset}\"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"ngpu\", torch.cuda.device_count())'

echo \"\n=== step 1: curate matched pools (CPU only) ===\"
python experiment_4/curate_matched_pools.py

echo \"\n=== step 2: refusal-direction forward pass on matched pools ===\"
time python experiment_2/refusal_direction.py \
    --model \"${MODEL}\" \
    --dtype bfloat16 \
    --batch-size ${BATCH_SIZE} \
    --harmful-file experiment_4/data/matched_harmful.jsonl \
    --harmless-file experiment_4/data/matched_harmless.jsonl \
    --output-dir ${MATCHED_OUT}

echo \"\n=== step 3: cosine similarity d̂ vs d̂* per layer ===\"
python experiment_4/compare_directions.py \
    --orig-dir results/qwen3-14b/refusal_direction \
    --matched-dir ${MATCHED_OUT}

echo
echo \"=== done ===\"
ls -la ${MATCHED_OUT} || true
"
