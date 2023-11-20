import yaml
import torch
from torch.optim import Adam
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode
import numpy as np
import clip
import os.path as osp
from NPHM.models.EnsembledDeepSDF import FastEnsembleDeepSDFMirrored
from NPHM import env_paths
import matplotlib.pyplot as plt

from render import render

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

with open('scripts/configs/fitting_nphm.yaml', 'r') as f:
    CFG = yaml.safe_load(f)
    
weight_dir_shape = env_paths.EXPERIMENT_DIR + '/{}/'.format(CFG['exp_name_shape'])
fname_shape = weight_dir_shape + 'configs.yaml'
with open(fname_shape, 'r') as f:
    CFG_shape = yaml.safe_load(f)
    

lm_inds = np.load(env_paths.ANCHOR_INDICES_PATH)
anchors = torch.from_numpy(np.load(env_paths.ANCHOR_MEAN_PATH)).float().unsqueeze(0).unsqueeze(0).to(device)

decoder_shape = FastEnsembleDeepSDFMirrored(
        lat_dim_glob=CFG_shape['decoder']['decoder_lat_dim_glob'],
        lat_dim_loc=CFG_shape['decoder']['decoder_lat_dim_loc'],
        hidden_dim=CFG_shape['decoder']['decoder_hidden_dim'],
        n_loc=CFG_shape['decoder']['decoder_nloc'],
        n_symm_pairs=CFG_shape['decoder']['decoder_nsymm_pairs'],
        anchors=anchors,
        n_layers=CFG_shape['decoder']['decoder_nlayers'],
        pos_mlp_dim=CFG_shape['decoder'].get('pos_mlp_dim', 256),
    )

decoder_shape = decoder_shape.to(device)

path = osp.join(weight_dir_shape, 'checkpoints/checkpoint_epoch_{}.tar'.format(CFG['checkpoint_shape']))
checkpoint = torch.load(path, map_location=device)
decoder_shape.load_state_dict(checkpoint['decoder_state_dict'], strict=True)

#from clip preprocessing
clip_tensor_preprocessor = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

def get_latent_mean_std():
    lat_mean = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_mean.npy'))
    lat_std = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_std.npy'))
    return lat_mean, lat_std

def forward(lat_rep, prompt, render_config):
    image = render(decoder_shape=decoder_shape,
           lat_rep=lat_rep,
           **render_config)
    
    image_c_first = image.permute(2, 0, 1)
    image_preprocessed = clip_tensor_preprocessor(image_c_first).unsqueeze(0)
    
    prompt_tokenized = clip.tokenize(prompt).to(device)
    
    return model(image_preprocessed, prompt_tokenized)[0]

def get_latent_from_text(prompt, n_updates=10):
    lat_mean, lat_std = get_latent_mean_std()
    lat_rep = (torch.randn(lat_mean.shape) * lat_std * 0.85 + lat_mean).detach().requires_grad_(True)
    
    optimizer = Adam(params=[lat_rep],
                 lr=0.001, 
                 maximize=True)
    
    res = 70
    render_config = {
        "pu": res,
        "pv": res,
        "camera_distance": 2.,
        "camera_angle": 0.,
        "ambient_coeff": 0.1,
        "diffuse_coeff": 0.6,
        "specular_coeff": 0.3,
        "shininess": 32.,
        "focal_length": 3.
    }

    scores = []
    latents = []
    for n in range(n_updates):
        optimizer.zero_grad()
        score = forward(lat_rep, prompt, render_config)
        scores.append(score.detach())
        latents.append(torch.clone(lat_rep))
        score.backward()
        print(f"update step {n} - score: {score}")
        optimizer.step()
        
    return lat_rep.detach(), scores, latents
    
    
    