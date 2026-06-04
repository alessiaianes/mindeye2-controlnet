#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# conda env create -f env.yml --> from CLI


# In[ ]:


import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf



# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
from models import *

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:",device)


# In[ ]:


# import os
# import sys
# import json
# import argparse
# import numpy as np
# import math
# from einops import rearrange
# import time
# import random
# import string
# import h5py
# from tqdm import tqdm
# import webdataset as wds

# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from accelerate import Accelerator

# # SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
# sys.path.append('generative_models/')
# import sgm
# from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
# from generative_models.sgm.models.diffusion import DiffusionEngine
# from generative_models.sgm.util import append_dims
# from omegaconf import OmegaConf



# # tf32 data type is faster than standard float32
# torch.backends.cuda.matmul.allow_tf32 = True

# # custom functions #
# import utils
# from models import *

# accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
# device = accelerator.device
# print("device:",device)


# In[ ]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "final_subj01_pretrained_40sess_24bs"
    print("model_name:", model_name)

    path = "datasets"

    # other variables can be specified in the following string:
    jupyter_args = f"--data_path={path} \
                    --cache_dir={path} \
                    --model_name={model_name} --subj=1 \
                    --hidden_dim=4096 --n_blocks=4 --new_test"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()

    from IPython.display import clear_output # function to clear print outputs in cell
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions


# In[3]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default=os.getcwd(),
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default=os.getcwd(),
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=2048,
)
parser.add_argument(
    "--new_test",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--seed",type=int,default=42,
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# seed all random functions
utils.seed_everything(seed)

# make output directory
os.makedirs("evals",exist_ok=True)
os.makedirs(f"evals/{model_name}",exist_ok=True)


# In[4]:


voxels = {}
# Load hdf5 data for betas
f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
betas = f['betas'][:]
betas = torch.Tensor(betas).to("cpu")
num_voxels = betas[0].shape[-1]
voxels[f'subj0{subj}'] = betas
print(f"num_voxels for subj0{subj}: {num_voxels}")

if not new_test: # using old test set from before full dataset released (used in original MindEye paper)
    if subj==3:
        num_test=2113
    elif subj==4:
        num_test=1985
    elif subj==6:
        num_test=2113
    elif subj==8:
        num_test=1985
    else:
        num_test=2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
else: # using larger test set from after full dataset released
    if subj==3:
        num_test=2371
    elif subj==4:
        num_test=2188
    elif subj==6:
        num_test=2371
    elif subj==8:
        num_test=2188
    else:
        num_test=3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"

print(test_url)
def my_split_by_node(urls): return urls
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{subj}!\n")


# In[5]:


# Prep images but don't load them all to memory
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

# Prep test voxels and indices of test images
test_images_idx = []
test_voxels_idx = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
    test_voxels_idx = np.append(test_images_idx, behav[:,0,5].cpu().numpy())
    test_images_idx = np.append(test_images_idx, behav[:,0,0].cpu().numpy())
test_images_idx = test_images_idx.astype(int)
test_voxels_idx = test_voxels_idx.astype(int)

assert (test_i+1) * num_test == len(test_voxels) == len(test_images_idx)
print(test_i, len(test_voxels), len(test_images_idx), len(np.unique(test_images_idx)))


# In[6]:


import torch
import os

device_0 = torch.device("cuda:0") 
device_1 = torch.device("cuda:1") 


clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="laion2b_s39b_b160k",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device_0)
clip_seq_dim = 256
clip_emb_dim = 1664

if blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)

    autoenc.to(device_0)
    utils.count_params(autoenc)


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x

model = MindEyeModule()

class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_sizes, out_features): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:,0]).unsqueeze(1)
        return out

model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)

from diffusers.models.vae import Decoder
from models import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1, 
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim) 
utils.count_params(model.ridge)
utils.count_params(model.backbone)
utils.count_params(model)

# setup diffusion prior network
out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens = clip_seq_dim,
        learned_query_mode="pos_emb"
    )

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)

model.to(device_1)

utils.count_params(model.diffusion_prior)
utils.count_params(model)

# Load pretrained model ckpt

tag='last'
outdir = os.path.abspath(f'train_logs/{model_name}')
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
try:
    checkpoint = torch.load(outdir+f'/{tag}.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    del checkpoint
except: # probably ckpt is saved using deepspeed format
    import deepspeed
    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=f"{tag}.pth")
    model.load_state_dict(state_dict, strict=False)
    del state_dict
print("ckpt loaded!")


# In[ ]:


# setup text caption networks
from transformers import AutoProcessor, AutoModelForCausalLM
from modeling_git import GitForCausalLMClipEmb

device_0 = torch.device("cuda:0")

processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")

clip_text_model.to(device_0) 
clip_text_model.eval().requires_grad_(False)
clip_text_seq_dim = 257
clip_text_emb_dim = 1024

class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super(CLIPConverter, self).__init__()
        self.linear1 = torch.nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = torch.nn.Linear(clip_emb_dim, clip_text_emb_dim)
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0,2,1))
        return x

clip_convert = CLIPConverter()
state_dict = torch.load(f"{cache_dir}/bigG_to_L_epoch8.pth", map_location='cpu')['model_state_dict']
clip_convert.load_state_dict(state_dict, strict=True)

clip_convert.to(device_0) 
del state_dict


# In[ ]:


import gc
gc.collect()
torch.cuda.empty_cache()

# prep unCLIP
config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 75

diffusion_engine = DiffusionEngine(network_config=network_config,
                       denoiser_config=denoiser_config,
                       first_stage_config=first_stage_config,
                       conditioner_config=conditioner_config,
                       sampler_config=sampler_config,
                       scale_factor=scale_factor,
                       disable_first_stage_autocast=disable_first_stage_autocast)
# set to inference
diffusion_engine.eval().requires_grad_(False)

ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])

del ckpt
gc.collect()

diffusion_engine.to(device_1, dtype=torch.float16)

batch={
      "jpg": torch.randn(1,3,1,1).to(device_1, dtype=torch.float16), 
      "original_size_as_tuple": torch.ones(1, 2).to(device_1, dtype=torch.float16) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device_1, dtype=torch.float16),}

out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device_1, dtype=torch.float16)
print("vector_suffix", vector_suffix.shape)


# In[ ]:


# @torch.no_grad()
# def _caption_refinement(samples, caption, diffusion_engine, vector_suffix,
#                          strength=0.5, ddim_steps=20, guidance_scale=5.0):
#     """
#     Replica lo step di enhanced_recon_inference.ipynb:
#     parte dall'output unCLIP (neuralmente puro) e lo raffina
#     usando la caption come condizionamento testuale nel diffusion_engine.

#     strength: 0.0 = output neurale invariato, 1.0 = massima influenza del testo
#     """
#     from generative_models.sgm.util import append_dims

#     # Porta l'immagine [0,1] nello spazio latente del VAE
#     x0 = samples.to(device_1, dtype=torch.float16) * 2.0 - 1.0  # [0,1] → [-1,1]
#     z0 = diffusion_engine.encode_first_stage(x0.unsqueeze(0) if x0.dim() == 3 else x0)
#     z0 = diffusion_engine.get_first_stage_encoding(z0)  # scala con scale_factor

#     # Condizionamento testuale tramite il conditioner dell'engine
#     # Il conditioner del unclip6 supporta sia image embedding che testo
#     batch_txt = {
#         "txt": [caption],
#         "original_size_as_tuple": torch.ones(1, 2, dtype=torch.float16, device=device_1) * 768,
#         "crop_coords_top_left": torch.zeros(1, 2, dtype=torch.float16, device=device_1),
#     }
#     try:
#         out_txt = diffusion_engine.conditioner(batch_txt)
#         c_txt = out_txt.get("crossattn", out_txt.get("vector", None))
#     except Exception:
#         # Fallback: usa get_learned_conditioning se il conditioner non supporta il testo
#         c_txt = diffusion_engine.get_learned_conditioning([caption])

#     uc_txt = diffusion_engine.get_learned_conditioning([""])

#     # Img2img: aggiungi rumore al latente, poi fai denoise con il testo
#     t_enc = int(strength * ddim_steps)
#     sampler = diffusion_engine.sampler

#     # Prepara latente rumoroso al timestep t_enc
#     noise = torch.randn_like(z0)
#     t = torch.tensor([t_enc], device=device_1)
#     z_noisy = diffusion_engine.q_sample(z0, t, noise=noise)

#     # Decode del latente raffinato
#     samples_refined = diffusion_engine.decode_first_stage(z_noisy)
#     samples_refined = (samples_refined / 2 + 0.5).clamp(0, 1)

#     return samples_refined.squeeze(0)  # CHW [0,1]


# In[ ]:


import sys
import types
import torch.nn.functional as F

# --- PATCH 1: MOCK XFORMERS ---
# Create a fake module to avoid errors and use native attention from PyTorch
if "xformers" not in sys.modules:
    mock_xformers = types.ModuleType("xformers")
    mock_ops = types.ModuleType("xformers.ops")
    mock_xformers.ops = mock_ops
    sys.modules["xformers"] = mock_xformers
    sys.modules["xformers.ops"] = mock_ops

    def native_sdpa(query, key, value, *args, **kwargs):
        needs_transpose = query.dim() == 4
        if needs_transpose:
            query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        out = F.scaled_dot_product_attention(query, key, value)

        if needs_transpose:
            out = out.transpose(1, 2)
        return out

    mock_ops.memory_efficient_attention = native_sdpa

import xformers 


# get all reconstructions
model.to(device_1)
model.eval().requires_grad_(False)

all_blurryrecons = None
# all_images = None
all_recons = None
all_predcaptions = []
all_clipvoxels = None

minibatch_size = 1
num_samples_per_image = 1
assert num_samples_per_image == 1

save_images = True 
os.makedirs(f"evals/{model_name}/images", exist_ok=True)

if utils.is_interactive(): plotting=True

# --- PATCH 2: Correction to visualize images avoiding errors
if hasattr(diffusion_engine, 'first_stage_model'):
    diffusion_engine.first_stage_model.to(device=device_1, dtype=torch.float32)

original_decode = diffusion_engine.decode_first_stage

def safe_decode(z, *args, **kwargs):
    with torch.cuda.amp.autocast(enabled=False):
        return original_decode(z.to(device_1, dtype=torch.float32), *args, **kwargs)

diffusion_engine.decode_first_stage = safe_decode
# --------------------------------------

# --- PATCH 3: FIX SGM ERROR
for module in diffusion_engine.modules():
    for attr_name in dir(module):
        if attr_name.startswith('__'): continue
        try:
            attr_val = getattr(module, attr_name)
            if isinstance(attr_val, torch.Tensor):
                setattr(module, attr_name, attr_val.to(device_1))
        except Exception:
            pass

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    # for batch in tqdm(range(0,len(np.unique(test_images_idx)),minibatch_size)): # uncomment this is for all images and comment the line right below
    for batch in tqdm(range(0,10)): # choose how many images you want to output

        uniq_imgs = np.unique(test_images_idx)[batch:batch+minibatch_size]
        voxel = None
        for uniq_img in uniq_imgs:
            locs = np.where(test_images_idx==uniq_img)[0]
            if len(locs)==1:
                locs = locs.repeat(3)
            elif len(locs)==2:
                locs = locs.repeat(2)[:3]
            assert len(locs)==3
            if voxel is None:
                voxel = test_voxels[None,locs] 
            else:
                voxel = torch.vstack((voxel, test_voxels[None,locs]))

        # Voxel on GPU 1
        voxel = voxel.to(device_1)

        for rep in range(3):
            voxel_ridge = model.ridge(voxel[:,[rep]],0)
            backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
            if rep==0:
                clip_voxels = clip_voxels0
                backbone = backbone0
                blurry_image_enc = blurry_image_enc0[0]
            else:
                clip_voxels += clip_voxels0
                backbone += backbone0
                blurry_image_enc += blurry_image_enc0[0]
        clip_voxels /= 3
        backbone /= 3
        blurry_image_enc /= 3

        if all_clipvoxels is None:
            all_clipvoxels = clip_voxels.cpu()
        else:
            all_clipvoxels = torch.vstack((all_clipvoxels, clip_voxels.cpu()))


        prior_out = model.diffusion_prior.p_sample_loop(backbone.shape, 
                        text_cond = dict(text_embed = backbone), 
                        cond_scale = 1., timesteps = 20)

        # --- CAPTIONING (GPU 0) ---
        prior_out_d0 = prior_out.to(device_0)
        pred_caption_emb = clip_convert(prior_out_d0)
        generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_predcaptions = np.hstack((all_predcaptions, generated_caption))
        print(generated_caption)

        # --- BLURRY RECONSTRUCTION (GPU 0) ---
        if blurry_recon:
            blurry_image_enc_d0 = blurry_image_enc.to(device_0)
            blurred_image = (autoenc.decode(blurry_image_enc_d0/0.18215).sample / 2 + 0.5).clamp(0,1)

        # --- UNCLIP RECON & COMBINED PLOTTING (GPU 1) ---
        with torch.cuda.device(device_1):
            for i, uniq_img in enumerate(uniq_imgs):

                # 1. unCLIP reconstruction
                samples = utils.unclip_recon(prior_out[[i]], 
                                 diffusion_engine,
                                 vector_suffix,
                                 init_latent=blurry_image_enc[[i]],
                                 num_samples=num_samples_per_image)
                if all_recons is None:
                    all_recons = samples.cpu()
                else:
                    all_recons = torch.vstack((all_recons, samples.cpu()))

                # 2. Blurry extraction
                im_blurry = None
                if blurry_recon:
                    im_blurry = torch.Tensor(blurred_image[i]).cpu()
                    if all_blurryrecons is None:
                        all_blurryrecons = im_blurry[None]
                    else:
                        all_blurryrecons = torch.vstack((all_blurryrecons, im_blurry[None]))

                # 3. Image composition (Original + Blurry + Recon)
                if save_images:
                    for s in range(num_samples_per_image):
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                        # Insert generated caption as main title of the image
                        fig.suptitle(generated_caption[i].capitalize(), fontsize=14, wrap=True)

                        # -- Original image from COCO --
                        coco_id = int(uniq_img)
                        try:
                            orig_img_tensor = torch.Tensor(images[coco_id]).float()

                            axes[0].imshow(transforms.ToPILImage()(orig_img_tensor))
                        except Exception as e:
                            axes[0].text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center')

                        axes[0].set_title(f"Original (Coco-ID: {coco_id})")
                        axes[0].axis('off')

                        # -- Blurry image --
                        if blurry_recon:
                            axes[1].imshow(transforms.ToPILImage()(im_blurry))
                        else:
                            axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center')
                        axes[1].set_title("Blurry")
                        axes[1].axis('off')

                        # -- Reconstructed image --
                        axes[2].imshow(transforms.ToPILImage()(samples[s].cpu()))
                        axes[2].set_title("Reconstructed")
                        axes[2].axis('off')


                        plt.tight_layout()
                        img_path_combined = f"evals/{model_name}/images/combined_batch{batch}_img{i}_sample{s}.png"
                        plt.savefig(img_path_combined, bbox_inches='tight', dpi=150)
                        plt.close()

        if save_images: 
            print(f"Generated batch {batch} and combined for {model_name} model")
            # break # Uncomment if you want to generate just 1 image

# resize outputs before saving
imsize = 256
all_recons = transforms.Resize((imsize,imsize))(all_recons).float()
if blurry_recon: 
    all_blurryrecons = transforms.Resize((imsize,imsize))(all_blurryrecons).float()

# saving
print(all_recons.shape)
# ! wget -O evals/all_images.pt https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/evals/all_images.pt
# torch.save(all_images, "evals/all_images.pt")
if blurry_recon:
    torch.save(all_blurryrecons,f"evals/{model_name}/{model_name}_all_blurryrecons.pt")
torch.save(all_recons,f"evals/{model_name}/{model_name}_all_recons.pt")
torch.save(all_predcaptions,f"evals/{model_name}/{model_name}_all_predcaptions.pt")
torch.save(all_clipvoxels,f"evals/{model_name}/{model_name}_all_clipvoxels.pt")
print(f"saved {model_name} outputs!")

if not utils.is_interactive():
    sys.exit(0)


# In[ ]:


# # ╔══════════════════════════════════════════════════════════════════════════╗
# # ║         CELLA NUOVA — ControlNet (native repo, SD1.5)                   ║
# # ╚══════════════════════════════════════════════════════════════════════════╝

# import os
# import sys
# from pathlib import Path

# # 1. Trova il percorso assoluto della cartella in cui si trova QUESTO script (ovvero 'src')
# src_path = Path(__file__).resolve().parent

# # 2. Calcola il percorso esatto della cartella ControlNet (che ora è dentro 'src')
# control_net_path = src_path / 'ControlNet'

# # Trasformiamo i percorsi in stringhe per sys.path
# str_src = str(src_path)
# str_control_net = str(control_net_path)

# # 3. Aggiungiamo 'src' al sys.path (per i tuoi import)
# if str_src not in sys.path:
#     sys.path.insert(0, str_src)

# # 4. Aggiungiamo 'ControlNet' al sys.path (per ingannare gli script originali della repo)
# if str_control_net not in sys.path:
#     sys.path.insert(0, str_control_net)

# import cv2
# import einops
# import numpy as np
# from PIL import Image

# import torch
# from pytorch_lightning import seed_everything

# # Import dalla repo originale — nessuna modifica ai file della repo
# from ControlNet.annotator.util import resize_image, HWC3
# from ControlNet.annotator.canny import CannyDetector
# from ControlNet.annotator.midas import MidasDetector
# from ControlNet.cldm.model import create_model, load_state_dict
# from ControlNet.cldm.ddim_hacked import DDIMSampler

# # ── Carica i modelli (una volta sola) ───────────────────────────────────────
# apply_canny = CannyDetector()
# apply_midas = MidasDetector()   # usa dpt_hybrid-midas-501f0c75.pt

# # Canny model
# canny_model = create_model('ControlNet/models/cldm_v15.yaml').cpu()
# # canny_model.load_state_dict(
# #     load_state_dict('ControlNet/models/control_sd15_canny.pth', location='cuda')
# # )
# canny_model.load_state_dict(
#     load_state_dict('ControlNet/models/control_sd15_canny.pth', location='cuda'),
#     strict=False
# )
# canny_model = canny_model.cuda()
# canny_sampler = DDIMSampler(canny_model)

# # Depth model
# depth_model = create_model('ControlNet/models/cldm_v15.yaml').cpu()
# # depth_model.load_state_dict(
# #     load_state_dict('ControlNet/models/control_sd15_depth.pth', location='cuda')
# # )
# depth_model.load_state_dict(
#     load_state_dict('ControlNet/models/control_sd15_depth.pth', location='cuda'),
#     strict=False
# )
# depth_model = depth_model.cuda()
# depth_sampler = DDIMSampler(depth_model)

# print("ControlNet (native) ready!")

# # ── Funzione di refinement — adattata da gradio_canny2image.py ──────────────
# def refine_with_controlnet(
#     recon_tensor,                    # CHW tensor [0,1] da MindEye2
#     caption: str,
#     use_caption: bool = True,
#     ddim_steps: int = 25,
#     strength: float = 1.0,          # control_scales — equivale a controlnet_conditioning_scale
#     guidance_scale: float = 7.5,    # unconditional_guidance_scale
#     image_resolution: int = 512,
#     low_threshold: int = 100,       # soglie Canny (interi 0-255, come in OpenCV)
#     high_threshold: int = 200,
#     guess_mode: bool = False,
#     seed: int = -1,
#     a_prompt: str = "best quality, extremely detailed",
#     n_prompt: str = "lowres, bad anatomy, worst quality, low quality",
# ) -> torch.Tensor:
# # refined = refine_with_controlnet(
# #     recon_tensor=recon_tensor,
# #     caption=caption,
# #     use_caption=use_caption,
# #     ddim_steps=40,            # era 25 — più step, più qualità
# #     strength=0.6,             # era 1.0 — meno controllo rigido della struttura
# #     guidance_scale=5.0,       # era 1.0 — anche senza caption, dai una direzione
# #     low_threshold=50,         # era 100 — cattura più bordi dalla recon
# #     high_threshold=150,       # era 200
# #     a_prompt="photorealistic, high quality photograph, sharp focus, 8k",
# #     n_prompt="painting, oil painting, artwork, illustration, sketch, "
# #              "blurry, foggy, hazy, smoke, low quality, worst quality, "
# #              "deformed, watermark, grainy, overexposed",
# # ) -> torch.Tensor:
#     """
#     Replica il pattern di gradio_canny2image.py e gradio_depth2image.py
#     adattato per ricevere un tensore CHW [0,1] invece di un numpy HWC.
#     """

#     # 1. Converti tensore → numpy HWC uint8 (formato atteso dalla repo)
#     img_np = (recon_tensor.cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
#     img_np = resize_image(HWC3(img_np), image_resolution)
#     H, W, C = img_np.shape

#     prompt = caption if use_caption else ""

#     if seed == -1:
#         seed = np.random.randint(0, 65535)
#     seed_everything(seed)

#     with torch.no_grad():

#         # ── Ramo Canny ───────────────────────────────────────────────────────
#         canny_map = apply_canny(img_np, low_threshold, high_threshold)
#         canny_map = HWC3(canny_map)
#         ctrl_canny = torch.from_numpy(canny_map.copy()).float().cuda() / 255.0
#         ctrl_canny = einops.rearrange(ctrl_canny[None], 'b h w c -> b c h w').clone()

#         cond_c = {
#             "c_concat":   [ctrl_canny],
#             "c_crossattn": [canny_model.get_learned_conditioning(
#                 [prompt + (', ' + a_prompt if use_caption else '')] * 1
#             )],
#         }
#         uncond_c = {
#             "c_concat":   None if guess_mode else [ctrl_canny],
#             "c_crossattn": [canny_model.get_learned_conditioning([n_prompt] * 1)],
#         }

#         canny_model.control_scales = (
#             [strength * (0.825 ** float(12 - i)) for i in range(13)]
#             if guess_mode else [strength] * 13
#         )

#         samples_c, _ = canny_sampler.sample(
#             ddim_steps, 1, (4, H // 8, W // 8), cond_c,
#             verbose=False, eta=0.0,
#             unconditional_guidance_scale=guidance_scale,
#             unconditional_conditioning=uncond_c,
#         )
#         out_canny = canny_model.decode_first_stage(samples_c)
#         out_canny = (einops.rearrange(out_canny, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

#         # ── Ramo Depth ───────────────────────────────────────────────────────
#         depth_map, _ = apply_midas(img_np)
#         depth_map = HWC3(depth_map)
#         ctrl_depth = torch.from_numpy(depth_map.copy()).float().cuda() / 255.0
#         ctrl_depth = einops.rearrange(ctrl_depth[None], 'b h w c -> b c h w').clone()

#         cond_d = {
#             "c_concat":    [ctrl_depth],
#             "c_crossattn": [depth_model.get_learned_conditioning(
#                 [prompt + (', ' + a_prompt if use_caption else '')] * 1
#             )],
#         }
#         uncond_d = {
#             "c_concat":    None if guess_mode else [ctrl_depth],
#             "c_crossattn": [depth_model.get_learned_conditioning([n_prompt] * 1)],
#         }

#         depth_model.control_scales = (
#             [strength * (0.825 ** float(12 - i)) for i in range(13)]
#             if guess_mode else [strength] * 13
#         )

#         samples_d, _ = depth_sampler.sample(
#             ddim_steps, 1, (4, H // 8, W // 8), cond_d,
#             verbose=False, eta=0.0,
#             unconditional_guidance_scale=guidance_scale,
#             unconditional_conditioning=uncond_d,
#         )
#         out_depth = depth_model.decode_first_stage(samples_d)
#         out_depth = (einops.rearrange(out_depth, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

#     # 3. Combina i due output con media pesata e restituisci tensore CHW [0,1]
#     blended = (out_canny.astype(np.float32) * 0.5 + out_depth.astype(np.float32) * 0.5).clip(0, 255).astype(np.uint8)
#     return transforms.ToTensor()(Image.fromarray(blended))  # CHW [0,1]


# In[ ]:


# # ╔══════════════════════════════════════════════════════════════════════════╗
# # ║              CELLA 12 — Loop di Refinement ControlNet                   ║
# # ╚══════════════════════════════════════════════════════════════════════════╝

# import gc

# gc.collect()
# torch.cuda.empty_cache()

# # ── 1. Carica i risultati della cella 9 ─────────────────────────────────────
# all_recons_fullres = torch.load(
#     f"evals/{model_name}/{model_name}_all_recons_fullres.pt"
# )
# all_predcaptions = torch.load(
#     f"evals/{model_name}/{model_name}_all_predcaptions.pt"
# )

# # Rileva automaticamente la modalità dal contenuto delle captions
# use_caption = any(c != "" for c in all_predcaptions)
# run_label   = "with_caption" if use_caption else "without_caption"
# print(f"Modalità rilevata: {run_label}")

# # ── 2. Loop di refinement ────────────────────────────────────────────────────
# all_refined = None
# os.makedirs(f"evals/{model_name}/images_refined_{run_label}", exist_ok=True)

# for idx in tqdm(range(len(all_recons_fullres))):
#     recon_tensor = all_recons_fullres[idx]
#     caption      = str(all_predcaptions[idx])   # stringa vuota se ablation

#     refined = refine_with_controlnet(
#         recon_tensor=recon_tensor,
#         caption=caption,
#         use_caption=use_caption,
#         ddim_steps=25,
#         strength=1.0,
#         guidance_scale=7.5 if use_caption else 1.0,
#         low_threshold=100,
#         high_threshold=200,
#     )

#     if all_refined is None:
#         all_refined = refined[None]
#     else:
#         all_refined = torch.vstack((all_refined, refined[None]))

#     # Figura comparativa: Originale | MindEye2 | ControlNet refined
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     title = caption.capitalize() if use_caption else "[no caption]"
#     fig.suptitle(title, fontsize=11)

#     try:
#         f_img = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
#         orig  = torch.Tensor(f_img['images'][int(np.unique(test_images_idx)[idx])]).float()
#         f_img.close()
#         axes[0].imshow(transforms.ToPILImage()(orig))
#     except Exception:
#         axes[0].text(0.5, 0.5, "N/A", ha='center', va='center')
#     axes[0].set_title("Originale"); axes[0].axis("off")

#     axes[1].imshow(transforms.ToPILImage()(recon_tensor.float()))
#     axes[1].set_title("MindEye2 Recon"); axes[1].axis("off")

#     axes[2].imshow(transforms.ToPILImage()(refined.float()))
#     axes[2].set_title(f"ControlNet ({run_label.replace('_', ' ')})")
#     axes[2].axis("off")

#     plt.tight_layout()
#     plt.savefig(
#         f"evals/{model_name}/images_refined_{run_label}/comparison_img{idx}.png",
#         bbox_inches="tight", dpi=150
#     )
#     plt.close()

# # ── 3. Salvataggio ───────────────────────────────────────────────────────────
# all_refined_256 = transforms.Resize((256, 256))(all_refined).float()
# torch.save(
#     all_refined_256,
#     f"evals/{model_name}/{model_name}_all_refined_{run_label}.pt"
# )
# print(f"Salvato: {all_refined_256.shape} → all_refined_{run_label}.pt")

# # # ── 1. Carica i risultati della cella 9 ─────────────────────────────────────
# # all_recons_fullres = torch.load(
# #     f"evals/{model_name}/{model_name}_all_recons_fullres.pt"
# # )  # shape: [N, 3, H, W], range [0,1]

# # all_predcaptions = torch.load(
# #     f"evals/{model_name}/{model_name}_all_predcaptions.pt"
# # )  # numpy array di N stringhe

# # print(f"Ricostruzioni caricate: {all_recons_fullres.shape}")
# # print(f"Esempio caption: '{all_predcaptions[0]}'")

# # # ── 2. Loop di refinement ────────────────────────────────────────────────────
# # all_refined_with_caption    = None
# # all_refined_without_caption = None

# # os.makedirs(f"evals/{model_name}/images_refined", exist_ok=True)

# # for idx in tqdm(range(len(all_recons_fullres))):
# #     recon_tensor = all_recons_fullres[idx]       # CHW [0,1]
# #     caption      = str(all_predcaptions[idx])

# #     # Variante A: con caption
# #     refined_cap = refine_with_controlnet(
# #         recon_tensor=recon_tensor,
# #         caption=caption,
# #         use_caption=USE_CAPTION,
# #         ddim_steps=25,
# #         strength=1.0,
# #         guidance_scale=7.5,
# #         low_threshold=100,
# #         high_threshold=200,
# #     )

# #     # # Variante B: senza caption (ablation)
# #     # refined_nocap = refine_with_controlnet(
# #     #     recon_tensor=recon_tensor,
# #     #     caption=caption,
# #     #     use_caption=False,
# #     #     ddim_steps=25,
# #     #     strength=1.0,
# #     #     guidance_scale=1.0,  # CFG basso senza testo
# #     #     low_threshold=100,
# #     #     high_threshold=200,
# #     # )

# #     # Accumula
# #     if all_refined_with_caption is None:
# #         all_refined_with_caption    = refined_cap[None]
# #         all_refined_without_caption = refined_nocap[None]
# #     else:
# #         all_refined_with_caption    = torch.vstack((all_refined_with_caption,    refined_cap[None]))
# #         all_refined_without_caption = torch.vstack((all_refined_without_caption, refined_nocap[None]))

# #     # ── Salva figura comparativa a 4 pannelli ────────────────────────────────
# #     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# #     fig.suptitle(caption.capitalize(), fontsize=11, wrap=True)

# #     panels = [
# #         (recon_tensor,   "MindEye2 Recon"),
# #         (refined_cap,    "ControlNet\n+ caption"),
# #         (refined_nocap,  "ControlNet\n- caption (ablation)"),
# #     ]

# #     # carica originale dal file hdf5 se ancora disponibile
# #     try:
# #         f_img = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
# #         orig = torch.Tensor(f_img['images'][int(np.unique(test_images_idx)[idx])]).float()
# #         f_img.close()
# #         axes[0].imshow(transforms.ToPILImage()(orig))
# #         axes[0].set_title("Originale")
# #     except Exception:
# #         axes[0].text(0.5, 0.5, "N/A", ha='center', va='center')
# #         axes[0].set_title("Originale")
# #     axes[0].axis("off")

# #     for ax, (tensor, title) in zip(axes[1:], panels):
# #         ax.imshow(transforms.ToPILImage()(tensor.float()))
# #         ax.set_title(title)
# #         ax.axis("off")

# #     plt.tight_layout()
# #     plt.savefig(
# #         f"evals/{model_name}/images_refined/comparison_img{idx}.png",
# #         bbox_inches="tight", dpi=150
# #     )
# #     plt.close()

# # print("Refinement completato!")

# # # ── 3. Salvataggio ───────────────────────────────────────────────────────────
# # imsize = 256

# # all_refined_with_caption_256 = transforms.Resize((imsize, imsize))(all_refined_with_caption).float()
# # all_refined_without_caption_256 = transforms.Resize((imsize, imsize))(all_refined_without_caption).float()

# # torch.save(
# #     all_refined_with_caption_256,
# #     f"evals/{model_name}/{model_name}_all_refined_with_caption.pt"
# # )
# # torch.save(
# #     all_refined_without_caption_256,
# #     f"evals/{model_name}/{model_name}_all_refined_without_caption.pt"
# # )

# # print(f"Salvati: {all_refined_with_caption_256.shape}")


# In[ ]:


# # ╔══════════════════════════════════════════════════════════════════════════╗
# # ║              CELLA NUOVA — ControlNet Refinement                        ║
# # ╚══════════════════════════════════════════════════════════════════════════╝

# import kornia
# import numpy as np
# from PIL import Image
# from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
# from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# print("ENTERING CLAUDE PART")
# # ── 0. Libera VRAM dai modelli MindEye2 (opzionale ma consigliato) ──────────
# # Se hai poca VRAM, decommenta queste righe per spostare tutto su CPU
# # model.to("cpu")
# # diffusion_engine.to("cpu")
# # clip_img_embedder.to("cpu")
# # clip_text_model.to("cpu")
# # autoenc.to("cpu")
# gc.collect()
# torch.cuda.empty_cache()

# # ── 1. Carica i risultati salvati dal loop MindEye2 ─────────────────────────
# all_recons_fullres = torch.load(
#     f"evals/{model_name}/{model_name}_all_recons_fullres.pt"
# )  # shape: [N, 3, 768, 768], range [0,1]

# all_predcaptions = torch.load(
#     f"evals/{model_name}/{model_name}_all_predcaptions.pt"
# )  # array di N stringhe

# print(f"Loaded {len(all_recons_fullres)} reconstructions")
# print(f"Sample caption: '{all_predcaptions[0]}'")

# # ── 2. Carica i modelli ControlNet ──────────────────────────────────────────
# print("Loading depth estimator...")
# depth_estimator = DPTForDepthEstimation.from_pretrained(
#     "Intel/dpt-hybrid-midas"
# ).to(device_0)
# depth_estimator.eval().requires_grad_(False)
# feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

# print("Loading ControlNet models...")
# controlnets = [
#     ControlNetModel.from_pretrained(
#         "diffusers/controlnet-depth-sdxl-1.0-small",  # 7x più leggero
#         torch_dtype=torch.float16,
#     ),
#     ControlNetModel.from_pretrained(
#         "diffusers/controlnet-canny-sdxl-1.0",
#         torch_dtype=torch.float16,
#     ),
# ]

# controlnet_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     controlnet=controlnets,
#     torch_dtype=torch.float16,
# )
# # CPU offload: sposta i sottomoduli su GPU solo quando servono
# controlnet_pipe.enable_model_cpu_offload()
# print("ControlNet ready!")

# # ── 3. Funzioni helper ───────────────────────────────────────────────────────
# @torch.no_grad()
# def get_depth_map(pil_img: Image.Image) -> Image.Image:
#     inputs = feature_extractor(images=pil_img, return_tensors="pt").to(device_0)
#     depth = depth_estimator(**inputs).predicted_depth          # [1, H, W]
#     depth = torch.nn.functional.interpolate(
#         depth.unsqueeze(1),
#         size=pil_img.size[::-1],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze().cpu().numpy()
#     depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255
#     return Image.fromarray(depth.astype(np.uint8)).convert("RGB")

# import kornia

# def get_canny_map(pil_img: Image.Image,
#                   low: float = 0.05,
#                   high: float = 0.15) -> Image.Image:
#     """
#     Kornia Canny: lavora su tensori, nessuna dipendenza da cv2.
#     low/high sono soglie normalizzate [0,1] invece degli interi di OpenCV.
#     """
#     # PIL → tensore float [0,1] BCHW su device_0
#     img_tensor = transforms.ToTensor()(pil_img.convert("RGB")).unsqueeze(0).to(device_0)

#     # Canny: restituisce (magnitude, edges_binary)
#     _, edges = kornia.filters.canny(
#         img_tensor,
#         low_threshold=low,
#         high_threshold=high,
#         kernel_size=(5, 5),
#         sigma=(1.0, 1.0),
#     )  # edges shape: [1, 1, H, W], valori 0/1

#     # Converti in RGB PIL
#     edges_np = (edges.squeeze().cpu().numpy() * 255).astype(np.uint8)
#     return Image.fromarray(np.stack([edges_np] * 3, axis=-1))

# def refine_with_controlnet(
#     recon_tensor,           # CHW tensor [0,1]
#     caption: str,
#     controlnet_scale: float = 0.8,
#     num_steps: int = 30,
# ) -> torch.Tensor:          # restituisce CHW tensor [0,1]
#     """
#     Prende un tensore CHW [0,1] e la caption testuale,
#     restituisce il tensore raffinato con ControlNet, stessa shape.
#     """
#     pil_orig = transforms.ToPILImage()(recon_tensor.float())
#     pil_1024 = pil_orig.resize((1024, 1024), Image.LANCZOS)

#     depth_map = get_depth_map(pil_1024)
#     canny_map  = get_canny_map(pil_1024)

#     refined_pil = controlnet_pipe(
#         prompt=caption,
#         negative_prompt="low quality, blurry, deformed, artifacts, watermark, hallucination",
#         image=[depth_map, canny_map],
#         num_inference_steps=num_steps,
#         controlnet_conditioning_scale=[controlnet_scale,          # depth
#                                        controlnet_scale * 0.8],   # canny
#         guidance_scale=7.5,
#         height=1024,
#         width=1024,
#     ).images[0]

#     # Riporta alla dimensione originale e converti in tensore
#     refined_pil = refined_pil.resize(pil_orig.size, Image.LANCZOS)
#     return transforms.ToTensor()(refined_pil)   # CHW [0,1]

# # ── 4. Loop di refinement ────────────────────────────────────────────────────
# all_refined = None
# os.makedirs(f"evals/{model_name}/images_refined", exist_ok=True)

# for idx in tqdm(range(len(all_recons_fullres))):
#     recon_tensor = all_recons_fullres[idx]          # CHW [0,1]
#     caption      = str(all_predcaptions[idx])

#     refined_tensor = refine_with_controlnet(
#         recon_tensor=recon_tensor,
#         caption=caption,
#         controlnet_scale=0.8,   # ← iperparametro principale da esplorare
#         num_steps=30,
#     )

#     # Accumula
#     if all_refined is None:
#         all_refined = refined_tensor[None]
#     else:
#         all_refined = torch.vstack((all_refined, refined_tensor[None]))

#     # Salva confronto: Recon MindEye2 | ControlNet Refined
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#     fig.suptitle(caption.capitalize(), fontsize=12, wrap=True)

#     axes[0].imshow(transforms.ToPILImage()(recon_tensor.float()))
#     axes[0].set_title("MindEye2 Recon")
#     axes[0].axis("off")

#     axes[1].imshow(transforms.ToPILImage()(refined_tensor.float()))
#     axes[1].set_title("ControlNet Refined")
#     axes[1].axis("off")

#     plt.tight_layout()
#     plt.savefig(f"evals/{model_name}/images_refined/refined_img{idx}.png",
#                 bbox_inches="tight", dpi=150)
#     plt.close()

# # ── 5. Salvataggio ───────────────────────────────────────────────────────────
# imsize = 256
# all_refined_256 = transforms.Resize((imsize, imsize))(all_refined).float()
# torch.save(all_refined_256, f"evals/{model_name}/{model_name}_all_refined.pt")
# print(f"Saved refined recons: {all_refined_256.shape}")


# # if not utils.is_interactive():
# #     sys.exit(0)

