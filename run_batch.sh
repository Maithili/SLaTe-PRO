#! /bin/bash
#SBATCH -o slurm/output_%j.txt
#SBATCH -e slurm/err_%j.txt
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node 1
#SBATCH -J emb
#SBATCH -p overcap

variation=gpt_valid_all
# for variation in gpt_only_split_0 gpt_only_split_1 gpt_only_split_2 gpt_only_split_3 gpt_only_split_4  real_only_split_0 real_only_split_1 real_only_split_2 real_only_split_3 real_only_split_4 real_only_split_5 real_only_split_6 real_and_gpt_split_0 real_and_gpt_split_1 real_and_gpt_split_2 real_and_gpt_split_3 real_and_gpt_split_4 real_and_gpt_split_5 real_and_gpt_split_6 gpt_valid_split_6 gpt_valid_all real_all real_valid_all; do sbatch run_batch.sh $variation; done
# do
embedder=custom_object_concat

logsDir=logs_0817stepwise_${embedder}_variations_${variation}

rm -r ./${logsDir}

/srv/rail-lab/flash5/mpatel377/anaconda3/envs/pyml/bin/python ./run.py \
    --coarse \
    --activity_availability=100 \
    --path=/coc/flash5/mpatel377/repos/SLaTe-PRO/data/Variations/${variation} \
    --logs_dir=./${logsDir} \
    --node_embedder=${embedder}

# done
