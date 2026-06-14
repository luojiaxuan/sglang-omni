#!/usr/bin/env bash
#SBATCH --job-name=p1nsys
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=00:50:00
#SBATCH --output=benchmarks/diagnostics/thinker_decode_forward/artifacts/logs/p1nsys_%j.log
#SBATCH --error=benchmarks/diagnostics/thinker_decode_forward/artifacts/logs/p1nsys_%j.log

# Edit #SBATCH --partition= and #SBATCH --nodelist= for your cluster (e.g. B200).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export GPUS="${GPUS:-0,1}"
export RUN_MODE="${RUN_MODE:-local}"

bash benchmarks/diagnostics/thinker_decode_forward/scripts/p1_nsys_decode.sh
