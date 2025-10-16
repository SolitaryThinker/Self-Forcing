#!/bin/bash
#SBATCH --job-name=sf-dmd-prof
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=dmd_sf_output/sf_dmd_prof_%j.out
#SBATCH --error=dmd_sf_output/sf_dmd_prof_%j.err
#SBATCH --exclusive
set -e -x

export HOME=/mnt/weka/home/hao.zhang/wl/
# source ~/conda/miniconda/bin/activate
# conda activate wl-sf2

# Basic Info
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# export WANDB_API_KEY="8d9f4b39abd68eb4e29f6fc010b7ee71a2207cde"
export WANDB_API_KEY="2f25ad37933894dbf0966c838c0b8494987f9f2f"

echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

NUM_GPUS=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0

# srun nsys profile --trace=cuda,nvtx -o result_sf \
srun \
torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py \
  --config_path self_forcing_dmd_prof.yaml \
  --logdir /mnt/sharefs/users/hao.zhang/wl/sf_logs/self_forcing_dmd_prof 
