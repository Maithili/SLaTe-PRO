#!/bin/bash
#SBATCH --job-name=slatepro10days
#SBATCH --partition=rail-lab
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --output=logs/slateproTrain_%j.out
#SBATCH --error=logs/slateproTrain_%j.err
#SBATCH --exclude=chitti,clippy,conroy,deebot,qt-1

HOUSEHOLD=${1}

# DATA_DIR=/coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/HOMER_PLUS/Household${HOUSEHOLD}
DATA_DIR=/coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/SLaTe-PRO/data/gen_warm65_resetTrue_suppressTrue_noRep0_every99999_vigoff_0421_2120/Household${HOUSEHOLD}
LOGS_DIR=/coc/flash5/mpatel377/repos/CoAdaptationSimulation/external/SLaTe-PRO/logs/gen_warm65_resetTrue_suppressTrue_noRep0_every99999_vigoff_0421_2120

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