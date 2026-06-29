#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, json, argparse
# os.system('wget -O evals/all_images.pt https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/evals/all_images.pt')
import numpy as np, math
from einops import rearrange
import time, random, string, h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from torchvision import transforms
from accelerate import Accelerator   # ← rimosso DeepSpeedPlugin
import transformers
from transformers import pipeline, AutoTokenizer

sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import (
    FrozenOpenCLIPImageEmbedder, FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
)
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True

import utils
from models import *

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device

# ── DUAL GPU ──────────────────────────────────────────────
device_0 = torch.device("cuda:0")   # text embedders, clip_img_embedder
device_1 = torch.device("cuda:1")   # base_engine (SDXL), latents
# ──────────────────────────────────────────────────────────

# PATCH 1 — mock xformers con attention nativa PyTorch
if "xformers" not in sys.modules:
    mock_xformers = types.ModuleType("xformers")
    mock_ops = types.ModuleType("xformers.ops")
    mock_xformers.ops = mock_ops
    sys.modules["xformers"] = mock_xformers
    sys.modules["xformers.ops"] = mock_ops

    def native_sdpa(query, key, value, *args, **kwargs):
        needs_transpose = query.dim() == 4
        if needs_transpose:
            query, key, value = query.transpose(1,2), key.transpose(1,2), value.transpose(1,2)
        out = F.scaled_dot_product_attention(query, key, value)
        if needs_transpose:
            out = out.transpose(1, 2)
        return out

    mock_ops.memory_efficient_attention = native_sdpa

import xformers
print("device:", device)


# In[ ]:


if utils.is_interactive():
    model_name = "final_subj01_pretrained_40sess_24bs"
    print("model_name:", model_name)

    path = "datasets"   # ← adatta al tuo percorso

    jupyter_args = f"--model_name={model_name} --subj=1 --cache_dir={path}"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()

    from IPython.display import clear_output


# In[ ]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument("--model_name", type=str, default="testing")
parser.add_argument("--subj", type=int, default=1, choices=[1,2,3,4,5,6,7,8])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--cache_dir", type=str, default=os.getcwd(),
    help="Cartella con i checkpoint scaricati (es. zavychromaxl_v30.safetensors)",
)

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

utils.seed_everything(seed)
os.makedirs("evals", exist_ok=True)
os.makedirs(f"evals/{model_name}", exist_ok=True)

# ── Carica output di recon_inference.ipynb ─────────────────
# all_images.pt → scaricabile da:
#   https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/evals
# Gli altri → prodotti dal tuo recon_inference.ipynb

all_images      = torch.load("evals/all_images.pt")
all_recons      = torch.load(f"evals/{model_name}/{model_name}_all_recons.pt")
all_clipvoxels  = torch.load(f"evals/{model_name}/{model_name}_all_clipvoxels.pt")
all_blurryrecons= torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons.pt")
all_predcaptions= torch.load(f"evals/{model_name}/{model_name}_all_predcaptions.pt")



# Resize a 768×768 per il passaggio SDXL
all_recons       = transforms.Resize((768, 768))(all_recons).float()
all_blurryrecons = transforms.Resize((768, 768))(all_blurryrecons).float()

print(model_name)
print(all_images.shape, all_recons.shape, all_clipvoxels.shape,
      all_blurryrecons.shape, len(all_predcaptions))


# In[ ]:


import gc
gc.collect()
torch.cuda.empty_cache()

# ── Carica config unclip6 (solo per sampler_config) ───────
config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
sampler_config = config["model"]["params"]["sampler_config"]
sampler_config['params']['num_steps'] = 38

# ── Carica config SDXL base ────────────────────────────────
config = OmegaConf.load("generative_models/configs/inference/sd_xl_base.yaml")
config = OmegaConf.to_container(config, resolve=True)
refiner_params = config["model"]["params"]

network_config            = refiner_params["network_config"]
denoiser_config           = refiner_params["denoiser_config"]
first_stage_config        = refiner_params["first_stage_config"]
conditioner_config        = refiner_params["conditioner_config"]
scale_factor              = refiner_params["scale_factor"]
disable_first_stage_autocast = refiner_params["disable_first_stage_autocast"]

# ── Checkpoint SDXL ────────────────────────────────────────
# Usa cache_dir (come nel tuo recon_inference per gli altri ckpt).
# Scarica sd_xl_base_1.0.safetensors oppure zavychromaxl_v30.safetensors
# os.system(f'wget -O {cache_dir}/zavychromaxl_v30.safetensors https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/zavychromaxl_v30.safetensors')
# base_ckpt_path = f'{cache_dir}/sd_xl_base_1.0.safetensors'
base_ckpt_path = f'{cache_dir}/zavychromaxl_v30.safetensors'  # alternativa

base_engine = DiffusionEngine(
    network_config=network_config,
    denoiser_config=denoiser_config,
    first_stage_config=first_stage_config,
    conditioner_config=conditioner_config,
    sampler_config=sampler_config,
    scale_factor=scale_factor,
    disable_first_stage_autocast=disable_first_stage_autocast,
    ckpt_path=base_ckpt_path,
)
base_engine.eval().requires_grad_(False)

# ── PATCH 2 — safe decode per base_engine ─────────────────
if hasattr(base_engine, 'first_stage_model'):
    base_engine.first_stage_model.to(device=device_1, dtype=torch.float32)

_orig_decode = base_engine.decode_first_stage

# def safe_decode_base(z, *args, **kwargs):
#     with torch.cuda.amp.autocast(enabled=False):
#         return _orig_decode(z.to(device_1, dtype=torch.float32), *args, **kwargs)
def safe_decode_base(z, *args, **kwargs):
    # Rimuoviamo la disattivazione dell'autocast o forziamo float16
    return _orig_decode(z.to(device_1, dtype=torch.float16), *args, **kwargs)

base_engine.decode_first_stage = safe_decode_base

# ── PATCH 3 — sposta tutti i tensor SGM su device_1 ───────
for module in base_engine.modules():
    for attr_name in dir(module):
        if attr_name.startswith('__'):
            continue
        try:
            attr_val = getattr(module, attr_name)
            if isinstance(attr_val, torch.Tensor):
                setattr(module, attr_name, attr_val.to(device_1))
        except Exception:
            pass

# base_engine → GPU 1
base_engine.to(device_1, dtype=torch.float16)

# ── Text embedders → GPU 0 ─────────────────────────────────
base_text_embedder1 = FrozenCLIPEmbedder(
    layer=conditioner_config['params']['emb_models'][0]['params']['layer'],
    layer_idx=conditioner_config['params']['emb_models'][0]['params']['layer_idx'],
)
base_text_embedder1.to(device_0)

base_text_embedder2 = FrozenOpenCLIPEmbedder2(
    arch=conditioner_config['params']['emb_models'][1]['params']['arch'],
    version=conditioner_config['params']['emb_models'][1]['params']['version'],
    freeze=conditioner_config['params']['emb_models'][1]['params']['freeze'],
    layer=conditioner_config['params']['emb_models'][1]['params']['layer'],
    always_return_pooled=conditioner_config['params']['emb_models'][1]['params']['always_return_pooled'],
    legacy=conditioner_config['params']['emb_models'][1]['params']['legacy'],
)
base_text_embedder2.to(device_0)

# ── Conditional/unconditional embeddings ──────────────────
# Calcola su GPU 0, poi sposta su GPU 1 per l'inferenza
batch = {
    "txt": "",
    "original_size_as_tuple": torch.ones(1, 2).to(device_0) * 768,
    "crop_coords_top_left":   torch.zeros(1, 2).to(device_0),
    "target_size_as_tuple":   torch.ones(1, 2).to(device_0) * 1024,
}
# Usa base_engine.conditioner ma con i tensori su device_0
# (conditioner internamente usa base_text_embedder che è su device_0)
base_engine_cpu_conditioner = base_engine.conditioner.to(device_0)
out = base_engine_cpu_conditioner(batch)

crossattn      = out["crossattn"].to(device_1)
vector_suffix  = out["vector"][:, -1536:].to(device_1)
print("crossattn", crossattn.shape)
print("vector_suffix", vector_suffix.shape)
print("---")

batch_uc = {
    "txt": ("painting, extra fingers, mutated hands, poorly drawn hands, "
            "poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, "
            "extra limbs, cloned face, skinny, glitchy, double torso, extra arms, "
            "extra hands, mangled fingers, missing lips, ugly face, distorted face, "
            "extra legs, anime"),
    "original_size_as_tuple": torch.ones(1, 2).to(device_0) * 768,
    "crop_coords_top_left":   torch.zeros(1, 2).to(device_0),
    "target_size_as_tuple":   torch.ones(1, 2).to(device_0) * 1024,
}
out = base_engine_cpu_conditioner(batch_uc)
crossattn_uc = out["crossattn"].to(device_1)
vector_uc    = out["vector"].to(device_1)
print("crossattn_uc", crossattn_uc.shape)
print("vector_uc", vector_uc.shape)

# Rimetti il conditioner su device_1 dopo uso
base_engine.conditioner.to(device_1)


# In[ ]:


if utils.is_interactive():
    plotting = True

num_samples         = 1
img2img_timepoint   = 13   # più alto = più influenza del prompt, meno dell'immagine
base_engine.sampler.guider.scale = 5  # CFG guidance scale

def denoiser(x, sigma, c):
    return base_engine.denoiser(base_engine.model, x, sigma, c)

# clip_img_embedder usato solo se plotting=True o num_samples>1
if utils.is_interactive() or num_samples > 1:
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to(device_0)   # ← GPU 0, leggero


# In[ ]:


import os
from torchvision import transforms
import matplotlib.pyplot as plt

all_enhancedrecons = None

# 1. Imposta plotting a False per l'esecuzione autonoma dello script
plotting = False 

# 2. Crea la cartella dove salvare i file PNG
png_dir = f"evals/{model_name}/enhanced_pngs"
os.makedirs(png_dir, exist_ok=True)

original_discretization = base_engine.sampler.discretization
def patched_discretization(*args, **kwargs):
    return original_discretization(*args, **kwargs).to(device_1)
base_engine.sampler.discretization = patched_discretization

for img_idx in tqdm(range(len(all_recons))):
# for img_idx in tqdm(range(0, 10)):

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        with base_engine.ema_scope():
            base_engine.sampler.num_steps = 25

            image = all_recons[[img_idx]]

            # Metriche in modalità plotting (opzionale)
            if plotting:
                print("blur pixcorr:", utils.pixcorr(
                    all_blurryrecons[[img_idx]].float(), all_images[[img_idx]].float()))
                print("blur cossim:", nn.functional.cosine_similarity(
                    clip_img_embedder(utils.resize(all_blurryrecons[[img_idx]].float(), 256).to(device_0)).flatten(1),
                    clip_img_embedder(utils.resize(all_images[[img_idx]].float(), 224).to(device_0)).flatten(1)))
                print("recon pixcorr:", utils.pixcorr(image, all_images[[img_idx]].float()))
                print("recon cossim:", nn.functional.cosine_similarity(
                    clip_img_embedder(utils.resize(image, 224).to(device_0)).flatten(1),
                    clip_img_embedder(utils.resize(all_images[[img_idx]].float(), 224).to(device_0)).flatten(1)))

            # ── Immagine e prompt ──────────────────────────────
            # image = image.to(device_1)   # ← GPU 1 per encoding
            # ── Immagine e prompt ──────────────────────────────
            image = image.to(device_1, dtype=torch.float16)   # ← GPU 1 in formato Half (FP16)
            prompt = all_predcaptions[img_idx]
            if isinstance(prompt, (list, np.ndarray)):
                prompt = str(prompt[0])

            if plotting:
                print("prompt:", prompt)
                plt.imshow(transforms.ToPILImage()(all_blurryrecons[img_idx].float())); plt.show()
                plt.imshow(transforms.ToPILImage()(all_recons[img_idx].float()));       plt.show()
                plt.imshow(transforms.ToPILImage()(image[0].cpu()));                    plt.show()

            # ── Encode immagine in spazio latente (GPU 1) ─────
            assert image.shape[-1] == 768
            z = base_engine.encode_first_stage(image * 2 - 1).repeat(num_samples, 1, 1, 1)

            # ── Text embeddings (GPU 0 → GPU 1) ──────────────
            openai_clip_text               = base_text_embedder1(prompt)          # GPU 0
            clip_text_tokenized, clip_text_emb = base_text_embedder2(prompt)      # GPU 0
            clip_text_emb        = torch.hstack((clip_text_emb, vector_suffix.to(device_0)))
            clip_text_tokenized  = torch.cat((openai_clip_text, clip_text_tokenized), dim=-1)

            # Sposta tutto su GPU 1 per il denoising
            # c = {
            #     "crossattn": clip_text_tokenized.to(device_1).repeat(num_samples, 1, 1),
            #     "vector":    clip_text_emb.to(device_1).repeat(num_samples, 1),
            # }
            # uc = {
            #     "crossattn": crossattn_uc.repeat(num_samples, 1, 1),
            #     "vector":    vector_uc.repeat(num_samples, 1),
            # }
            # Sposta tutto su GPU 1 per il denoising
            c = {
                "crossattn": clip_text_tokenized.to(device_1).repeat(num_samples, 1, 1),
                "vector":    clip_text_emb.to(device_1).repeat(num_samples, 1),
            }
            uc = {
                "crossattn": crossattn_uc.to(device_1).repeat(num_samples, 1, 1),
                "vector":    vector_uc.to(device_1).repeat(num_samples, 1),
            }


            # ── Img2img con SDXL (tutto su GPU 1) ─────────────
            noise  = torch.randn_like(z)
            sigmas = base_engine.sampler.discretization(
                base_engine.sampler.num_steps).to(device_1)
            init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) \
                     / torch.sqrt(1.0 + sigmas[0] ** 2.0)
            sigmas = sigmas[-img2img_timepoint:].repeat(num_samples, 1)

            base_engine.sampler.num_steps = sigmas.shape[-1] - 1
            noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(
                init_z, cond=c, uc=uc, num_steps=base_engine.sampler.num_steps)

            for timestep in range(base_engine.sampler.num_steps):
                noised_z = base_engine.sampler.sampler_step(
                    sigmas[:, timestep], sigmas[:, timestep + 1],
                    denoiser, noised_z, cond=c, uc=uc, gamma=0)

            samples_z_base = noised_z
            samples_x      = base_engine.decode_first_stage(samples_z_base)   # usa safe_decode
            samples        = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            # ── Scelta del campione migliore ───────────────────
            if num_samples == 1:
                samples = samples[0]
            else:
                sample_cossim = nn.functional.cosine_similarity(
                    clip_img_embedder(utils.resize(samples, 224).to(device_0)).flatten(1),
                    clip_img_embedder(utils.resize(all_images[[img_idx]].float(), 224).to(device_0)).flatten(1))
                which_sample = torch.argmax(sample_cossim)

                if plotting:
                    for n in range(num_samples):
                        plt.imshow(transforms.ToPILImage()(samples[n]))
                        plt.show()
                        if (n == which_sample).item(): print("CHOSEN ABOVE")
                    raise StopIteration  # interrompe con plotting=True

                samples = samples[which_sample]

            # 3. ── SALVATAGGIO IN PNG ──────────────────────────────
            # Convertiamo il tensore della singola immagine generata in formato PIL
            final_pil_img = transforms.ToPILImage()(samples.cpu())
            final_pil_img.save(f"{png_dir}/enhanced_img_{img_idx}.png")

            # (Opzionale) Salva anche una griglia combinata Originale -> Blurry -> Recon -> Enhanced
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(str(prompt).capitalize(), fontsize=14, wrap=True)

            axes[0].imshow(transforms.ToPILImage()(all_images[img_idx].cpu().float()))
            axes[0].set_title("Original")
            axes[1].imshow(transforms.ToPILImage()(all_blurryrecons[img_idx].cpu().float()))
            axes[1].set_title("Blurry")
            axes[2].imshow(transforms.ToPILImage()(all_recons[img_idx].cpu().float()))
            axes[2].set_title("Recon")
            axes[3].imshow(final_pil_img)
            axes[3].set_title("Enhanced")

            for ax in axes: ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"{png_dir}/combined_img_{img_idx}.png", bbox_inches='tight', dpi=150)
            plt.close() # Chiude la figura per liberare RAM
            # ───────────────────────────────────────────────────────

            # ── Accumula risultati su CPU ──────────────────────
            samples = samples.cpu()[None]
            if all_enhancedrecons is None:
                all_enhancedrecons = samples
            else:
                all_enhancedrecons = torch.vstack((all_enhancedrecons, samples))

# Ridimensiona e salva
all_enhancedrecons = transforms.Resize((256, 256))(all_enhancedrecons).float()
print("all_enhancedrecons", all_enhancedrecons.shape)
torch.save(all_enhancedrecons, f"evals/{model_name}/{model_name}_all_enhancedrecons.pt")
print(f"saved evals/{model_name}/{model_name}_all_enhancedrecons.pt")

if not utils.is_interactive():
    sys.exit(0)


# In[ ]:


# all_enhancedrecons = None
# plotting = True

# # for img_idx in tqdm(range(len(all_recons))):
# for img_idx in tqdm(range(0,10)):

#     with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
#         with base_engine.ema_scope():
#             base_engine.sampler.num_steps = 25

#             image = all_recons[[img_idx]]

#             # Metriche in modalità plotting (opzionale)
#             if plotting:
#                 print("blur pixcorr:", utils.pixcorr(
#                     all_blurryrecons[[img_idx]].float(), all_images[[img_idx]].float()))
#                 print("blur cossim:", nn.functional.cosine_similarity(
#                     clip_img_embedder(utils.resize(all_blurryrecons[[img_idx]].float(), 256).to(device_0)).flatten(1),
#                     clip_img_embedder(utils.resize(all_images[[img_idx]].float(), 224).to(device_0)).flatten(1)))
#                 print("recon pixcorr:", utils.pixcorr(image, all_images[[img_idx]].float()))
#                 print("recon cossim:", nn.functional.cosine_similarity(
#                     clip_img_embedder(utils.resize(image, 224).to(device_0)).flatten(1),
#                     clip_img_embedder(utils.resize(all_images[[img_idx]].float(), 224).to(device_0)).flatten(1)))

#             # ── Immagine e prompt ──────────────────────────────
#             image = image.to(device_1)   # ← GPU 1 per encoding
#             prompt = all_predcaptions[img_idx]
#             if isinstance(prompt, (list, np.ndarray)):
#                 prompt = str(prompt[0])

#             if plotting:
#                 print("prompt:", prompt)
#                 plt.imshow(transforms.ToPILImage()(all_blurryrecons[img_idx].float())); plt.show()
#                 plt.imshow(transforms.ToPILImage()(all_recons[img_idx].float()));       plt.show()
#                 plt.imshow(transforms.ToPILImage()(image[0].cpu()));                    plt.show()

#             # ── Encode immagine in spazio latente (GPU 1) ─────
#             assert image.shape[-1] == 768
#             z = base_engine.encode_first_stage(image * 2 - 1).repeat(num_samples, 1, 1, 1)

#             # ── Text embeddings (GPU 0 → GPU 1) ──────────────
#             openai_clip_text               = base_text_embedder1(prompt)          # GPU 0
#             clip_text_tokenized, clip_text_emb = base_text_embedder2(prompt)      # GPU 0
#             clip_text_emb        = torch.hstack((clip_text_emb, vector_suffix.to(device_0)))
#             clip_text_tokenized  = torch.cat((openai_clip_text, clip_text_tokenized), dim=-1)

#             # Sposta tutto su GPU 1 per il denoising
#             c = {
#                 "crossattn": clip_text_tokenized.to(device_1).repeat(num_samples, 1, 1),
#                 "vector":    clip_text_emb.to(device_1).repeat(num_samples, 1),
#             }
#             uc = {
#                 "crossattn": crossattn_uc.repeat(num_samples, 1, 1),
#                 "vector":    vector_uc.repeat(num_samples, 1),
#             }

#             # ── Img2img con SDXL (tutto su GPU 1) ─────────────
#             noise  = torch.randn_like(z)
#             sigmas = base_engine.sampler.discretization(
#                 base_engine.sampler.num_steps).to(device_1)
#             init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) \
#                      / torch.sqrt(1.0 + sigmas[0] ** 2.0)
#             sigmas = sigmas[-img2img_timepoint:].repeat(num_samples, 1)

#             base_engine.sampler.num_steps = sigmas.shape[-1] - 1
#             noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(
#                 init_z, cond=c, uc=uc, num_steps=base_engine.sampler.num_steps)

#             for timestep in range(base_engine.sampler.num_steps):
#                 noised_z = base_engine.sampler.sampler_step(
#                     sigmas[:, timestep], sigmas[:, timestep + 1],
#                     denoiser, noised_z, cond=c, uc=uc, gamma=0)

#             samples_z_base = noised_z
#             samples_x      = base_engine.decode_first_stage(samples_z_base)   # usa safe_decode
#             samples        = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

#             # ── Scelta del campione migliore ───────────────────
#             if num_samples == 1:
#                 samples = samples[0]
#             else:
#                 sample_cossim = nn.functional.cosine_similarity(
#                     clip_img_embedder(utils.resize(samples, 224).to(device_0)).flatten(1),
#                     clip_img_embedder(utils.resize(all_images[[img_idx]].float(), 224).to(device_0)).flatten(1))
#                 which_sample = torch.argmax(sample_cossim)

#                 if plotting:
#                     for n in range(num_samples):
#                         plt.imshow(transforms.ToPILImage()(samples[n]))
#                         plt.show()
#                         if (n == which_sample).item(): print("CHOSEN ABOVE")
#                     raise StopIteration  # interrompe con plotting=True

#                 samples = samples[which_sample]

#             # ── Accumula risultati su CPU ──────────────────────
#             samples = samples.cpu()[None]
#             if all_enhancedrecons is None:
#                 all_enhancedrecons = samples
#             else:
#                 all_enhancedrecons = torch.vstack((all_enhancedrecons, samples))

# # Ridimensiona e salva
# all_enhancedrecons = transforms.Resize((256, 256))(all_enhancedrecons).float()
# print("all_enhancedrecons", all_enhancedrecons.shape)
# torch.save(all_enhancedrecons, f"evals/{model_name}/{model_name}_all_enhancedrecons.pt")
# print(f"saved evals/{model_name}/{model_name}_all_enhancedrecons.pt")

# if not utils.is_interactive():
#     sys.exit(0)

