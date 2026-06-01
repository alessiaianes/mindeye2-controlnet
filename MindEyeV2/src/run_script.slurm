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



unset LD_LIBRARY_PATH
unset PYTHONPATH

source ~/.bashrc
conda activate myenv

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_USE_CPU=False

# 1. Convertiamo il notebook in script
jupyter nbconvert --to script recon_inference_claude.ipynb

# 2. Ripuliamo in automatico le righe di installazione sporche di Jupyter
sed -i '/get_ipython/d' recon_inference_claude.py

# 3. Eseguiamo il codice python
python recon_inference_claude.py \
    --data_path=datasets \
    --cache_dir=datasets \
    --model_name=final_subj01_pretrained_40sess_24bs \
    --subj=1 \
    --hidden_dim=4096 \
    --n_blocks=4 \
    --new_test