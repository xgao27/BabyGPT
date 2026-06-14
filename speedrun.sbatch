#!/bin/bash
#SBATCH -J BabyGPT        
#SBATCH --account=gts-jxu81
#SBATCH -N1 --gres=gpu:H200:1                    
#SBATCH --mem=140G                               
#SBATCH -t 04:00:00                               
#SBATCH -q inferno                                 
#SBATCH -o Report-%j.out                         
#SBATCH --mail-type=BEGIN,END,FAIL              
#SBATCH --mail-user=xxxxxxxxxing@gmail.com      

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh


# Reset home directary
export HOME=/storage/home/hcoda1/7/jxu81/d-techai-0/jxu81
# Default intermediate artifacts directory is in ~/.cache/nanochat
# export OMP_NUM_THREADS=1
# export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# # install uv (if not already installed)
# command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# # create a .venv local virtual environment (if it doesn't exist)
# [ -d ".venv" ] || uv venv
# # install the repo dependencies
# uv sync --extra gpu
# # activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
module load cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
cd $SLURM_SUBMIT_DIR
# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`

# torchrun --standalone --nproc_per_node=8 train_gpt2.py
python 1_finetune/finetune_babygpt_tinystories.py
# torchrun --standalone --nproc_per_node=2 1_finetune/finetune_babygpt_tinystories.py