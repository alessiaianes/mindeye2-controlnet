#!/bin/bash
#SBATCH --job-name=eyeinference
#SBATCH -p edu-long
#SBATCH -t 12:00:00          
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=edu01
#SBATCH -o slurms/%j.out
#SBATCH -e slurms/%j.err

set -e

unset LD_LIBRARY_PATH
unset PYTHONPATH

source ~/.bashrc
conda activate mindeye

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_USE_CPU=False


# STEP 1 — recon_inference
echo "===== [$(date)] START recon_inference ====="

jupyter nbconvert --to script recon_inference.ipynb
sed -i '/get_ipython/d' recon_inference.py

python recon_inference.py \
    --data_path=datasets \
    --cache_dir=datasets \
    --model_name=final_subj01_pretrained_40sess_24bs \
    --subj=1 \
    --hidden_dim=4096 \
    --n_blocks=4 \
    --new_test

echo "===== [$(date)] END recon_inference ====="


# STEP 2a — enhanced_recon_inference
echo "===== [$(date)] START enhanced_recon_inference ====="

jupyter nbconvert --to script enhanced_recon_inference.ipynb
sed -i '/get_ipython/d' enhanced_recon_inference.py

python enhanced_recon_inference.py \
    --model_name=final_subj01_pretrained_40sess_24bs \
    --subj=1 \
    --cache_dir=datasets

echo "===== [$(date)] END enhanced_recon_inference ====="


# STEP 2b — canny_recon_inference 
conda activate controlnet
echo "===== [$(date)] START canny_recon_inference ====="



sed -i '/get_ipython/d' canny_recon_inference.py
python -c "import torch; torch.cuda.empty_cache()"


python canny_recon_inference.py \
    --model_name=final_subj01_pretrained_40sess_24bs \
    --subj=1 \
    --cache_dir=datasets

echo "===== [$(date)] END canny_recon_inference ====="


# STEP 3 — final_evaluations 
conda activate mindeye
echo "===== [$(date)] START final_evaluations ====="

jupyter nbconvert --to script final_evaluations.ipynb
sed -i '/get_ipython/d' final_evaluations.py

MODEL=final_subj01_pretrained_40sess_24bs

# 3a — Baseline MindEye2
python final_evaluations.py \
    --model_name=${MODEL} --subj=1 \
    --data_path=datasets --cache_dir=datasets \
    --all_recons_path=evals/${MODEL}/${MODEL}_all_recons.pt

# 3b — SDXL Enhanced (standard pipeline)
python final_evaluations.py \
    --model_name=${MODEL} --subj=1 \
    --data_path=datasets --cache_dir=datasets \
    --all_recons_path=evals/${MODEL}/${MODEL}_all_enhancedrecons.pt

# 3c — ControlNet (our idea)
python final_evaluations.py \
    --model_name=${MODEL} --subj=1 \
    --data_path=datasets --cache_dir=datasets \
    --all_recons_path=evals/${MODEL}/${MODEL}_all_controlnet_canny.pt \
    --predcaptions_path=evals/${MODEL}/${MODEL}_all_blip2captions.pt

echo "===== [$(date)] END final_evaluations ====="


# STEP 4 — compare_methods_evaluations 
echo "===== [$(date)] START compare_methods_evaluations ====="

jupyter nbconvert --to script compare_methods_evaluations.ipynb
sed -i '/get_ipython/d' compare_methods_evaluations.py

python compare_methods_evaluations.py

echo "===== [$(date)] END compare_methods_evaluations ====="