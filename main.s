#!/bin/bash
#
#SBATCH --output=slurm_MT_%j.out
#SBATCH --job-name=train_MT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1

module load python3/intel/3.6.3

source /home/dam740/pytorch_venv/bin/activate

#val=$SLURM_ARRAY_TASK_ID
#arams=$(sed -n ${val}p to_train.txt)
#ead -r data_model_kind data_path model_out_dir log_path num_positives net_size dropout_hidden dropout_input weight lr kind <<< $params
python main.py --epochs 1\
               --batch-size 32\
               --num-directions 1\
               --encoder-num-layers 1\
               --decoder-num-layers 1\
               --encoder-emb-size 30\
               --decoder-emb-size 30\
               --encoder-hid-size 50\
               --beam-size 5\
               --target-vocab 10000\
               --max-len-source 100\
               --max-len-target 100\
               --teacher-forcing-prob 0.5\
               --num-directions 2
