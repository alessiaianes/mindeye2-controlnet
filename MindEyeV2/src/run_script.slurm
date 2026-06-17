#!/bin/bash
#SBATCH --job-name=eyeinference
#SBATCH -p edu-medium
#SBATCH -t 02:00:00          
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
conda activate myenv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_USE_CPU=False

# ─────────────────────────────────────────────────────────────
# STEP 1 — recon_inference
# ─────────────────────────────────────────────────────────────
echo "===== [$(date)] START recon_inference ====="

# jupyter nbconvert --to script recon_inference.ipynb
# sed -i '/get_ipython/d' recon_inference.py

# python recon_inference.py \
#     --data_path=datasets \
#     --cache_dir=datasets \
#     --model_name=final_subj01_pretrained_40sess_24bs \
#     --subj=1 \
#     --hidden_dim=4096 \
#     --n_blocks=4 \
#     --new_test

# echo "===== [$(date)] END recon_inference ====="

# # ─────────────────────────────────────────────────────────────
# # STEP 2 — enhanced_recon_inference
# # ─────────────────────────────────────────────────────────────
# echo "===== [$(date)] START enhanced_recon_inference ====="

# jupyter nbconvert --to script enhanced_recon_inference.ipynb
# sed -i '/get_ipython/d' enhanced_recon_inference.py

# python enhanced_recon_inference.py \
#     --model_name=final_subj01_pretrained_40sess_24bs \
#     --subj=1 \
#     --cache_dir=datasets

# echo "===== [$(date)] END enhanced_recon_inference ====="

# ─────────────────────────────────────────────────────────────
# STEP 3 — canny_recon_inference  (controlnet separato)
# ─────────────────────────────────────────────────────────────
conda activate controlnet
echo "===== [$(date)] START canny_recon_inference ====="


sed -i '/get_ipython/d' canny_recon_inference.py
python -c "import torch; torch.cuda.empty_cache()"


python canny_recon_inference.py \
    --model_name=final_subj01_pretrained_40sess_24bs \
    --subj=1 \
    --cache_dir=datasets

echo "===== [$(date)] END canny_recon_inference ====="