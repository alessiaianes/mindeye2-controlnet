#!/bin/bash
#SBATCH --job-name=eyeinference
#SBATCH -p edu-long
#SBATCH -t 07:00:00          
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
conda activate myenv3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_USE_CPU=False

# # ─────────────────────────────────────────────────────────────
# # STEP 1 — recon_inference
# # ─────────────────────────────────────────────────────────────
# echo "===== [$(date)] START recon_inference ====="

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

# # ─────────────────────────────────────────────────────────────
# # STEP 3 — canny_recon_inference  (controlnet separato)
# # ─────────────────────────────────────────────────────────────
# conda activate controlnet6
# echo "===== [$(date)] START canny_recon_inference ====="



# sed -i '/get_ipython/d' canny_recon_inference.py
# python -c "import torch; torch.cuda.empty_cache()"


# python canny_recon_inference.py \
#     --model_name=final_subj01_pretrained_40sess_24bs \
#     --subj=1 \
#     --cache_dir=datasets

# echo "===== [$(date)] END canny_recon_inference ====="

# ─────────────────────────────────────────────────────────────
# STEP 4 — final_evaluations (torna a myenv3: serve lo stack MindEye2)
# ─────────────────────────────────────────────────────────────
conda activate myenv
echo "===== [$(date)] START final_evaluations ====="

jupyter nbconvert --to script final_evaluations.ipynb
sed -i '/get_ipython/d' final_evaluations.py

MODEL=final_subj01_pretrained_40sess_24bs

# 4a — Baseline MindEye2
python final_evaluations.py \
    --model_name=${MODEL} --subj=1 \
    --data_path=datasets --cache_dir=datasets \
    --all_recons_path=evals/${MODEL}/${MODEL}_all_recons.pt

# 4b — SDXL Enhanced (pipeline standard, con blending blurry)
python final_evaluations.py \
    --model_name=${MODEL} --subj=1 \
    --data_path=datasets --cache_dir=datasets \
    --all_recons_path=evals/${MODEL}/${MODEL}_all_enhancedrecons.pt \
    --blend_blurry

# 4c — ControlNet (il tuo contributo)
python final_evaluations.py \
    --model_name=${MODEL} --subj=1 \
    --data_path=datasets --cache_dir=datasets \
    --all_recons_path=evals/${MODEL}/${MODEL}_all_controlnet_canny.pt \
    --predcaptions_path=evals/${MODEL}/${MODEL}_all_blip2captions.pt

echo "===== [$(date)] END final_evaluations ====="

# ─────────────────────────────────────────────────────────────
# STEP 5 — compare_methods_evaluations (leggero, stesso env va bene)
# ─────────────────────────────────────────────────────────────
echo "===== [$(date)] START compare_methods_evaluations ====="

jupyter nbconvert --to script compare_methods_evaluations.ipynb
sed -i '/get_ipython/d' compare_methods_evaluations.py

python compare_methods_evaluations.py

echo "===== [$(date)] END compare_methods_evaluations ====="