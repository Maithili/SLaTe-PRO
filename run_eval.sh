#!/bin/bash
#SBATCH --job-name=slatepro_eval
#SBATCH --partition=rail-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slatepro_eval_%j.out
#SBATCH --error=logs/slatepro_eval_%j.err
#SBATCH --exclude=qt-1,deebot

HOUSEHOLD=${1}

cd /coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/SLaTe-PRO

## Generated Dataset
ckpt_path=./logs/gen_warm65_resetTrue_suppressTrue_noRep0_every99999_vigoff_0421_2120/Household${HOUSEHOLD}/default_100
logs_dir=./logs/gen_warm65_resetTrue_suppressTrue_noRep0_every99999_vigoff_0421_2120
# ## Original Dataset
# ckpt_path=./logs/newData2/Household${HOUSEHOLD}/default_100
# logs_dir=./logs/newData2EvalOnGeneratedDataset_0331_1646

# ## Original datasat
# dataset_path=/coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/HOMER_PLUS/Household${HOUSEHOLD}/processed_seqLM_coarse
## Generated dataset
dataset_path=/coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/SLaTe-PRO/data/gen_warm65_resetTrue_suppressTrue_noRep0_every99999_vigoff_0421_2120/Household${HOUSEHOLD}

echo "ckpt_path: $ckpt_path"
echo "logs_dir: $logs_dir"
echo "dataset_path: $dataset_path"
echo "Household: $HOUSEHOLD"

/coc/flash5/mpatel377/anaconda3/envs/pyml/bin/python \
  ./run.py \
  --path $dataset_path \
  --read_ckpt \
  --ckpt_dir $ckpt_path \
  --logs_dir $logs_dir \
  --name default_100

echo "Done: Household $HOUSEHOLD"
