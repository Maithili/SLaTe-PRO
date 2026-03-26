#!/bin/bash
#SBATCH --job-name=SP_New2
#SBATCH --partition=rail-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slateproNewData_%j.out
#SBATCH --error=logs/slateproNewData_%j.err

HOUSEHOLD=${1}

DATA_DIR=/coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/HOMER_PLUS/Household${HOUSEHOLD}
LOGS_DIR=/coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/SLaTe-PRO/logs/newData2

echo "DATA_DIR: $DATA_DIR"
echo "LOGS_DIR: $LOGS_DIR"

mkdir -p $LOGS_DIR/Household${HOUSEHOLD}
cp $DATA_DIR/processed_seqLM_coarse/common_data.json $LOGS_DIR/Household${HOUSEHOLD}

/coc/flash5/mpatel377/anaconda3/envs/pyml/bin/python \
./run.py \
--activity_availability=100 \
--path=$DATA_DIR  \
--logs_dir=$LOGS_DIR

echo "Done: Household $HOUSEHOLD"