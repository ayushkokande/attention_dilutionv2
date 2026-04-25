#!/bin/bash
#SBATCH --job-name=qwen3_14b_baseline
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
# Log paths must be absolute: ./logs depends on cwd when you run sbatch.
# Keep prefix in sync with REPO below.
#SBATCH --output=/scratch/ak13124/attention_dilutionv2/logs/qwen3_14b_baseline_%j.out
#SBATCH --error=/scratch/ak13124/attention_dilutionv2/logs/qwen3_14b_baseline_%j.err

# Experiment 1 / step 0: establish a Qwen3-14B baseline on harmless (Alpaca)
# and harmful (AdvBench) prompts. Single HF stack across the project: weights
# are sharded across the 2 A100s via accelerate device_map="auto".
# Outputs (matches the ayush/ pipeline's results/<slug>/ layout):
#   results/qwen3-14b/baseline_alpaca.jsonl
#   results/qwen3-14b/baseline_advbench.jsonl
#   results/qwen3-14b/baseline_summary.json   # refusal / compliance rates
#
# Submit from anywhere (script always cd's to ${REPO}):
#   sbatch experiment_1/sbatch_baseline_qwen14b.sh

set -euo pipefail

# --- edit these ---
SCRATCH="/scratch/ak13124"
SIF="${SCRATCH}/ubuntu-20.04.3.sif"
OVERLAY="${SCRATCH}/overlay-25GB-500K.ext3:ro"
REPO="${SCRATCH}/attention_dilutionv2"

MODEL="Qwen/Qwen3-14B"
ALPACA_N=512
ADVBENCH_N=520
MAX_NEW_TOKENS=512
BATCH_SIZE=8
SEED=0
# ------------------

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
for f in requirements.txt experiment_1/utils.py experiment_1/baseline_benchmark.py; do
  if [ ! -f \"\$f\" ]; then
    echo \"ERROR: missing \$(pwd)/\$f\" >&2
    missing=1
  fi
done
if [ \"\$missing\" = \"1\" ]; then
  echo >&2
  echo \"Sync the full repo to \${REPO} first, e.g. from your laptop:\" >&2
  echo \"  rsync -avz --exclude .venv --exclude __pycache__ --exclude results/ \\\\\" >&2
  echo \"    ./ greene:${REPO}/\" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  uv venv --python 3.11 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt

echo \"=== qwen3_14b_baseline | job \${SLURM_JOB_ID:-local} ===\"
echo \"Repo : \$(pwd)\"
echo \"CUDA : \${CUDA_VISIBLE_DEVICES:-unset}\"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"ngpu\", torch.cuda.device_count())'

SLUG=qwen3-14b
OUT_DIR=results/\${SLUG}
mkdir -p \"\${OUT_DIR}\"

run_step() {
  local name=\"\$1\"; shift
  echo
  echo \"----- [\${name}] \$(date -u +%FT%TZ) -----\"
  time \"\$@\"
}

run_step baseline_qwen3_14b python experiment_1/baseline_benchmark.py \
    --model \"${MODEL}\" \
    --dtype bfloat16 \
    --batch-size ${BATCH_SIZE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --temperature 0.0 \
    --alpaca-n ${ALPACA_N} \
    --advbench-n ${ADVBENCH_N} \
    --seed ${SEED}

echo
echo \"=== done | results under: \${PWD}/\${OUT_DIR} ===\"
ls -la \"\${OUT_DIR}\" || true
"
