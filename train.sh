#!/bin/bash
#SBATCH --job-name=few_shot
#SBATCH --output=few_shot.out
#SBATCH --error=few_shot.err
#SBATCH --gres=gpu:1


cd /home/hlee959
source miniconda3/bin/activate
conda activate CSUL
cd /home/hlee959/projects/2023_CSUL/CSUL_MOCO

python --version

# Run the training script
CUDA_LAUNCH_BLOCKING=1 python train.py --dataset cifar100 \
                --model_type ViT-B_16 \
                --pretrained_dir '/local/scratch2/hlee959/CSUL_MOCO/pretrained/ViT-B_16.npz' \
                --output_dir '/local/scratch2/hlee959/CSUL_MOCO/vit_c100_few_shot' \
                --img_size 224 \
                --train_batch_size 512 \
                --eval_batch_size 64 \
                --eval_every 100 \
                --learning_rate 3e-2 \
                --weight_decay 5e-5 \
                --name few_shot_cifar100_ViT-B_16
