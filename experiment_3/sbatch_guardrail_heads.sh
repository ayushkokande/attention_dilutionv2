#!/bin/bash
#SBATCH --job-name=qwen3_14b_guard
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ak13124/attention_dilutionv2/logs/qwen3_14b_guard_%j.out
#SBATCH --error=/scratch/ak13124/attention_dilutionv2/logs/qwen3_14b_guard_%j.err

# Experiment 3: Guardrail Heads via Direct Logit Attribution onto d_hat at
# the canonical read-out layer (L18 for Qwen3-14B per Exp.2 causal sweep).
#
# Prereq: experiment_2/sbatch_refusal_direction.sh must have produced
#   results/qwen3-14b/refusal_direction/{d_hat_all_layers.pt, meta.json}
#
# Submit:
#   sbatch experiment_3/sbatch_guardrail_heads.sh

set -euo pipefail

SCRATCH="/scratch/ak13124"
SIF="${SCRATCH}/ubuntu-20.04.3.sif"
OVERLAY="${SCRATCH}/overlay-25GB-500K.ext3:ro"
REPO="${SCRATCH}/attention_dilutionv2"

MODEL="Qwen/Qwen3-14B"
# Reuse the held-out pool that defined L_read in Exp.2 causal sweep
# (AdvBench[480:504], n=24). Disjoint from d_hat training and from exp_8 sweep eval.
N_PROMPTS=24
START_IDX=480
TOP_K=12
BATCH_SIZE=4

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
for f in requirements.txt experiment_1/utils.py experiment_3/guardrail_heads.py \
         results/qwen3-14b/refusal_direction/d_hat_all_layers.pt \
         results/qwen3-14b/refusal_direction/meta.json; do
  if [ ! -e \"\$f\" ]; then
    echo \"ERROR: missing \$(pwd)/\$f\" >&2
    missing=1
  fi
done
if [ \"\$missing\" = \"1\" ]; then
  echo >&2
  echo \"Run sbatch_refusal_direction.sh first to produce d_hat_all_layers.pt,\" >&2
  echo \"or sync the repo with rsync if scripts are missing.\" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  uv venv --python 3.11 .venv
fi
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt

echo \"=== qwen3_14b_guardrail_heads | job \${SLURM_JOB_ID:-local} ===\"
echo \"Repo : \$(pwd)\"
echo \"CUDA : \${CUDA_VISIBLE_DEVICES:-unset}\"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"ngpu\", torch.cuda.device_count())'

time python experiment_3/guardrail_heads.py \
    --model \"${MODEL}\" \
    --dtype bfloat16 \
    --refusal-dir results/qwen3-14b/refusal_direction \
    --n-prompts ${N_PROMPTS} \
    --start-idx ${START_IDX} \
    --top-k ${TOP_K} \
    --batch-size ${BATCH_SIZE}

echo
echo \"=== done ===\"
ls -la results/qwen3-14b/guardrail_heads || true
"
