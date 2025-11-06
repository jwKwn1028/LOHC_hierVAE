#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # CPU cores per node
#SBATCH --partition=snu-gpu1         # Partition
#SBATCH --job-name=qm9_1
#SBATCH --time=07-12:00               # D-HH:MM
#SBATCH -o test.%N.%j.out            # STDOUT
#SBATCH -e test.%N.%j.err            # STDERR
#SBATCH --gres=gpu:a5000:1           # GPU: a5000 or a6000, then :1

set -euo pipefail

echo "Host: $(hostname)"
echo "Date: $(date)"

# Use ONLY your Miniconda; avoid module Anaconda to prevent path conflicts
module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hiervae

# Clean Python environment
unset PYTHONPATH
export PYTHONNOUSERSITE=1


# Helpful diagnostics
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found"
echo "conda: $(command -v conda)"; conda --version

StartTime=$(date +%s)

cd "$SLURM_SUBMIT_DIR"
echo "Workdir: $(pwd)"

echo "Python (hiervae):"
srun --export=ALL python train_generator.py \
  --train train_processed_qm9/ \
  --vocab get_vocabresults/QM9loosekekulizevocab.txt \
  --save_dir ckpt/qm9dropout \
  --epoch 50 \
  --batch_size 64 \
  --lr 5e-4 \
  --clip_norm 5.0 \
  --print_iter 100 \
  --save_iter 1000 \
  --anneal_iter 2000 \
  --max_beta 0.2 \
  --step_beta 0.02 \
  --warmup 3000 \
  --kl_anneal_iter 300 \
  --dropout 0.2
# Run your program (add --no-capture-output to see live prints in job log)

EndTime=$(date +%s)

echo "Run time"
echo $StartTime $EndTime | awk '{print $2-$1 " sec"}'
echo $StartTime $EndTime | awk '{print ($2-$1)/60 " min"}'
echo $StartTime $EndTime | awk '{print ($2-$1)/3600 " hour"}'
