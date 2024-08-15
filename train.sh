#!/bin/

#!/usr/bin/env bash

# set -x

# env
# echo "Activating conda env..."
# /usr/local/lib/miniconda3/condabin/conda init bash
# source activate
# conda deactivate
# conda activate /mnt/afs/user/liuyangzhou/.conda/envs/clip

# cd /mnt/afs/user/liuyangzhou/workspace/CorrFormer/corrformer

# export PYTHONPATH=/mnt/afs/user/chenzhe/workspace/petrel-oss-sdk
# export PYTHONPATH=$PYTHONPATH:/mnt/afs/user/liuyangzhou/workspace/CorrFormer/corrformer
# export CUDA_HOME="/usr/local/cuda-11.8/"
# export PATH="/usr/local/cuda-11.8/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64/:$LD_LIBRARY_PATH"

# GPUS=${GPUS:-8}
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29990}

# torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} train_multi_gpu.py | tee ./logs/corrformer_v5_4_yfcc_v1.txt

torchrun --nnodes=1 --nproc_per_node=8 --master_port=29990 train_multi_gpu.py | tee ./logs/corradaptor.txt

# s8a bash /mnt/afs/user/liuyangzhou/workspace/CorrFormer/corrformer/train.sh