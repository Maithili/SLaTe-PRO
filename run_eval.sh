#!/bin/bash
#SBATCH --job-name=slatepro_eval
#SBATCH --partition=rail-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slatepro_eval_%j.out
#SBATCH --error=logs/slatepro_eval_%j.err

HOUSEHOLD=${1}

cd /coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/SLaTe-PRO

/coc/flash5/mpatel377/anaconda3/envs/pyml/bin/python \
  ./run.py \
  --path /coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/HOMER_PLUS/Household${HOUSEHOLD}/processed_seqLM_coarse \
  --read_ckpt \
  --ckpt_dir ./logs/newData2/Household${HOUSEHOLD}/default_100 \
  --logs_dir ./logs/newData2eval \
  --name default_100

echo "Done: Household $HOUSEHOLD"
