import argparse
import os
import sys
import types

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from pytorch_lightning import seed_everything
import textwrap
from PIL import Image

from transformers import Blip2Processor, Blip2ForConditionalGeneration

sys.path.insert(0, 'ControlNet')

# Patch for xformers
if 'xformers' not in sys.modules:
    _mx = types.ModuleType('xformers')
    _mo = types.ModuleType('xformers.ops')
    _mx.ops = _mo
    sys.modules['xformers']     = _mx
    sys.modules['xformers.ops'] = _mo
    def _sdpa(q, k, v, *a, **kw):
        nd = q.dim() == 4
        if nd: q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)
        return o.transpose(1, 2) if nd else o
    _mo.memory_efficient_attention = _sdpa

# Same import from gradio_canny2image.py 
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


parser = argparse.ArgumentParser(description='ControlNet Canny — MindEye2 refinement with BLIP-2')
parser.add_argument('--model_name',     type=str,   default='final_subj01_pretrained_40sess_24bs')
parser.add_argument('--subj',           type=int,   default=1)
parser.add_argument('--seed',           type=int,   default=42)
parser.add_argument('--cache_dir',      type=str,   default='datasets')
parser.add_argument('--strength',       type=float, default=0.75,
    help='img2img: 0=same as recon, 1=pure generation')
parser.add_argument('--guidance_scale', type=float, default=7.5)
parser.add_argument('--ddim_steps',     type=int,   default=30)
args = parser.parse_args()

model_name     = args.model_name
strength       = args.strength
guidance_scale = args.guidance_scale
ddim_steps     = args.ddim_steps
cache_dir      = args.cache_dir

seed_everything(args.seed)
os.makedirs(f'evals/{model_name}', exist_ok=True)
print(f'strength={strength}  gs={guidance_scale}  steps={ddim_steps}')


apply_canny = CannyDetector()

model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
os.system('wget -O ./ControlNet/models/control_sd15_canny.pth https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth')
model.load_state_dict(load_state_dict('./ControlNet/models/control_sd15_canny.pth', location='cuda'), strict=False)
model = model.cuda()
model.eval()
ddim_sampler = DDIMSampler(model)

# Parameters
H = W            = 512
low_threshold    = 200   
high_threshold   = 250   
num_samples      = 1
eta              = 0.0

# Richer negative prompt to prevent hallucination
n_prompt = 'deformed, mutated, bad anatomy, bad proportions, unnatural eyes, unnatural body, fused digits, extra limbs, missing limbs, cloned face, fused bodies, asymmetric, impossible geometry, warped perspective, melted, broken, merged objects, structural failure, flat depth, out of frame, blurry, blurred edges, pixelated, low resolution, worst quality, jpeg artifacts, text, watermark, signature'

# Loading images
all_images       = torch.load('evals/all_images.pt').float()
all_recons       = torch.load(f'evals/{model_name}/{model_name}_all_recons.pt').float()

# Loading blip
print("Loading BLIP-2 model...")
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir, use_fast=False)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    torch_dtype=torch.float16,
    cache_dir=cache_dir
).cuda()
blip_model.eval()

print(f'all_recons: {all_recons.shape}')

# Function readapted from original file gradio_canny2image.py 
def process(input_tensor, prompt):
    with torch.no_grad():
        img = resize_image(
            HWC3((input_tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)), H
        )
        
        # Clean micro-deformation maintaining the exact shape
        img_for_canny = cv2.GaussianBlur(img, (7, 7), 0)
        
        detected_map = apply_canny(img_for_canny, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        # control tensor
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        
        # Adjusted to correct the image without distorting the image
        control = control * 0.7
        
        control = torch.stack([control] * num_samples, dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        cond = {
            'c_concat':    [control],
            'c_crossattn': [model.get_learned_conditioning([prompt] * num_samples)],
        }
        un_cond = {
            'c_concat':    [control],
            'c_crossattn': [model.get_learned_conditioning([n_prompt] * num_samples)],
        }

        init = torch.from_numpy(img).float().cuda() / 127.5 - 1.0
        init = einops.rearrange(init, 'h w c -> 1 c h w').clone()
        z0   = model.get_first_stage_encoding(model.encode_first_stage(init))

        ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=False)
        t_enc = int(strength * ddim_steps)
        z_enc = ddim_sampler.stochastic_encode(
            z0, torch.tensor([t_enc]).cuda()
        )

        samples = ddim_sampler.decode(
            z_enc, cond, t_enc,
            unconditional_guidance_scale = guidance_scale,
            unconditional_conditioning   = un_cond,
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5)
        x_samples = x_samples.cpu().numpy().clip(0, 255).astype(np.uint8)

    return torch.from_numpy(x_samples[0]).permute(2,0,1).float() / 255.0


# Loop and saving with BLIP-2
all_cn  = None
out_dir = f'evals/{model_name}/controlnet_canny'
os.makedirs(out_dir, exist_ok=True)

# Defining prompt to feed BLIP forcing it to describe details
blip_question = ("Question: List every object, person, and animal visible in this image. For each one describe: its color, its shape or appearance, and its position in the scene. Then describe the background environment. Answer:"
)


quality_tags = ", masterpiece, photorealistic, highly detailed, sharp focus, 8k resolution, realistic anatomy, natural proportions, symmetrical features, detailed eyes, realistic eyes, realistic fur, lifelike textures, correct solid geometry, intricate details, realistic materials, cinematic lighting, well-defined edges"
all_blip2captions = []
for idx in tqdm(range(len(all_recons))):
    recon = all_recons[idx].float()
    
    # Retrieve reconstructed image (brain-only) and generate detailed caption with BLIP-2
    orig_image_pil = transforms.ToPILImage()(all_images[idx].float())  
    recon_pil      = transforms.ToPILImage()(recon).resize((224, 224), Image.LANCZOS)
    blip_inputs    = blip_processor(recon_pil, text=blip_question, return_tensors="pt").to("cuda", torch.float16)
    
    with torch.no_grad():
        
        # Added 'min_new_tokens' and 'repetition_penalty' to force longer description
        generated_ids = blip_model.generate(
            **blip_inputs, 
            max_new_tokens=80, 
            min_new_tokens=10,
            num_beams=5,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        base_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if "Caption:" in base_caption:
            base_caption = base_caption.split("Caption:")[-1].strip()
        elif "Answer:" in base_caption:
            base_caption = base_caption.split("Answer:")[-1].strip()
        else:
            base_caption = base_caption.replace(blip_question, "").strip()
    
    all_blip2captions.append(base_caption)
    # combine caption and high quality tags
    enhanced_caption = base_caption + quality_tags
    
    # Feed to ControlNet
    refined = process(recon, enhanced_caption)

    if all_cn is None: all_cn = refined[None]
    else:              all_cn = torch.vstack((all_cn, refined[None]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    wrapped_caption = "\n".join(textwrap.wrap(base_caption, width=110))
    
    fig.suptitle(wrapped_caption, fontsize=10, y=1.05)
    axes[0].imshow(orig_image_pil)
    axes[0].set_title('Originale NSD'); axes[0].axis('off')
    axes[1].imshow(transforms.ToPILImage()(transforms.Resize((H, W))(recon)))
    axes[1].set_title('MindEye2 Recon'); axes[1].axis('off')
    axes[2].imshow(transforms.ToPILImage()(refined.float()))
    axes[2].set_title(f'ControlNet (s={strength})'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/comparison_{idx:04d}.png', bbox_inches='tight', dpi=150)
    plt.close()

imsize     = 256
all_cn_256 = transforms.Resize((imsize, imsize))(all_cn).float()
save_path  = f'evals/{model_name}/{model_name}_all_controlnet_canny.pt'
torch.save(all_cn_256, save_path)
print(f'Saved: {all_cn_256.shape}  ->  {save_path}')


caption_save_path = f'evals/{model_name}/{model_name}_all_blip2captions.pt'
torch.save(all_blip2captions, caption_save_path)
print(f'Saved: {len(all_blip2captions)} caption -> {caption_save_path}')