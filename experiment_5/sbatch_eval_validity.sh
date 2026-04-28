#!/bin/bash
#SBATCH --job-name=qwen3_14b_validity
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/ak13124/attention_dilutionv2/logs/qwen3_14b_validity_%j.out
#SBATCH --error=/scratch/ak13124/attention_dilutionv2/logs/qwen3_14b_validity_%j.err

# Experiment 5: style-mismatched validity (AUC_intent vs AUC_vocab).
#
# Before GPU step (login node):
#   python experiment_5/build_edgy_lexicon.py
#   python experiment_5/build_styled_pools.py --dump-seeds-only   # optional
#   # Author experiment_5/data/manual_rewrites.json then:
#   python experiment_5/build_styled_pools.py
#
# Preflight expects refusal dirs + styled pools under experiment_5/data/.
#
# Outputs:
#   results/qwen3-14b/validity/per_prompt.jsonl
#   results/qwen3-14b/validity/auc.json
#   results/qwen3-14b/validity/scatter.png

set -euo pipefail

SCRATCH="/scratch/ak13124"
SIF="${SCRATCH}/ubuntu-20.04.3.sif"
OVERLAY="${SCRATCH}/overlay-25GB-500K.ext3:ro"
REPO="${SCRATCH}/attention_dilutionv2"

MODEL="Qwen/Qwen3-14B"
BATCH_SIZE=8
LAYERS="20 28"
ORIG_DIR="results/qwen3-14b/refusal_direction"
MATCHED_DIR="results/qwen3-14b/refusal_direction_matched"
OUT_DIR="results/qwen3-14b/validity"

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

missing=0
for f in requirements.txt experiment_1/utils.py experiment_2/refusal_direction.py \\
         experiment_5/eval_validity.py \\
         experiment_5/data/edgy_lexicon.json \\
         experiment_5/data/camouflaged_harmful.jsonl \\
         experiment_5/data/edgy_harmless.jsonl \\
         ${ORIG_DIR}/d_hat_all_layers.pt \\
         ${MATCHED_DIR}/d_hat_all_layers.pt; do
  if [ ! -e \"\$f\" ]; then
    echo \"ERROR: missing \$(pwd)/\$f\" >&2
    missing=1
  fi
done
if [ \"\$missing\" = \"1\" ]; then
  echo >&2
  echo \"Run locally first:\" >&2
  echo \"  python experiment_5/build_edgy_lexicon.py\" >&2
  echo \"  python experiment_5/build_styled_pools.py  (needs data/manual_rewrites.json)\" >&2
  echo \"Also sync refusal_direction artifacts from experiment 2 and 4.\" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  uv venv --python 3.11 .venv
fi
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt

echo \"=== qwen3_14b_validity | job \${SLURM_JOB_ID:-local} ===\"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'

time python experiment_5/eval_validity.py \\
    --model \"${MODEL}\" \\
    --dtype bfloat16 \\
    --refusal-dirs \"${ORIG_DIR}\" \"${MATCHED_DIR}\" \\
    --refusal-dir-names refusal_direction refusal_direction_matched \\
    --layers ${LAYERS} \\
    --batch-size ${BATCH_SIZE} \\
    --output-dir \"${OUT_DIR}\"

echo \"=== done ===\"
ls -la \"${OUT_DIR}\" || true
"
