import yaml
import torch
from torch.optim import Adam
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode
import numpy as np
import clip
import os
import os.path as osp
from src.NPHM.models.EnsembledDeepSDF import FastEnsembleDeepSDFMirrored
from src.NPHM import env_paths
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import uuid

from utils.render import render

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device="cpu")

with open('../NPHM/scripts/configs/fitting_nphm.yaml', 'r') as f:
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
    Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=None),
    CenterCrop(224),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

def get_latent_mean_std():
    lat_mean = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_mean.npy'))
    lat_std = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_std.npy'))
    return lat_mean, lat_std

def forward(lat_rep, prompt, camera_params, phong_params, light_params):
    image = render(decoder_shape, lat_rep, camera_params, phong_params, light_params)

    image_c_first = image.permute(2, 0, 1)
    image_preprocessed = clip_tensor_preprocessor(image_c_first).unsqueeze(0).cpu()

    prompt_tokenized = clip.tokenize(prompt).cpu()
    
    CLIP_score = model(image_preprocessed, prompt_tokenized)[0]
    
    lat_mean, lat_std = get_latent_mean_std()
    cov = lat_std * torch.eye(lat_mean.shape[0])
    delta = lat_rep - lat_mean
    prob = -delta.T @ torch.inverse(cov) @ delta
    
    score = CLIP_score + 0.2 * prob

    image_preprocessed = None
    prompt_tokenized = None
    return score, torch.clone(image)


def get_latent_from_text(prompt, hparams, init_lat=None):
    if init_lat is None:
        lat_mean, lat_std = get_latent_mean_std()
        lat_rep = (torch.randn(lat_mean.shape) * lat_std * 0.85 + lat_mean).detach().requires_grad_(True)
    else:
        lat_rep = init_lat.requires_grad_(True)

    optimizer = Adam(params=[lat_rep],
                     lr=hparams['optimizer_lr'],
                     maximize=True)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=hparams['lr_scheduler_factor'],
        patience=hparams['lr_scheduler_patience'],
        min_lr=hparams['lr_scheduler_min_lr']
    )

    camera_params = {
        "camera_distance": 0.21,
        "camera_angle": 45.,
        "focal_length": 2.57,
        "max_ray_length": (0.25 + 1) * 1.58 + 1.5,
        # Image
        "resolution_y": hparams['resolution'],
        "resolution_x": hparams['resolution']
    }
    phong_params = {
        "ambient_coeff": 0.51,
        "diffuse_coeff": 0.75,
        "specular_coeff": 0.64,
        "shininess": 0.5,
        # Colors
        "object_color": torch.tensor([0.53, 0.24, 0.64]),
        "background_color": torch.tensor([0.36, 0.77, 0.29])
    }

    light_params = {
        "amb_light_color": torch.tensor([0.9, 0.16, 0.55]),
        # light 1
        "light_intensity_1": 1.42,
        "light_color_1": torch.tensor([0.8, 0.97, 0.89]),
        "light_dir_1": torch.tensor([-0.6, -0.4, -0.67]),
        # light p
        "light_intensity_p": 0.62,
        "light_color_p": torch.tensor([0.8, 0.97, 0.89]),
        "light_pos_p": torch.tensor([1.19, -1.27, 2.24])
    }

    # Normal Mode
    #now = datetime.now()
    #writer = SummaryWriter(log_dir=f'../runs/identity-pipeline/train-time:{now.strftime("%Y-%m-%d-%H:%M:%S")}')

    # Hparam Optimization Mode
    trial_uuid = str(uuid.uuid4())
    writer = SummaryWriter(log_dir=f'../runs/identity_pipeline/hparamtuning-{trial_uuid}')

    scores = []
    latents = []
    images = []
    best_score = torch.tensor([0]).cpu()
    torch.cuda.empty_cache()
    optimizer.zero_grad()

    for iteration in tqdm(range(hparams['n_iterations'])):
    
        score, image = forward(lat_rep, prompt, camera_params, phong_params, light_params)

        scores.append(score.detach().cpu())
        latents.append(torch.clone(lat_rep).cpu())
        images.append(image.cpu())

        if score > best_score:
            best_score = score.detach().cpu()
            best_latent = torch.clone(lat_rep).cpu()

        score.backward()
        print(f"update step {iteration+1} - score: {score}")

        writer.add_scalar('CLIP score', score, iteration)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], iteration)
        if ((iteration == 0) or (iteration % 10 == 0)):
            writer.add_image(f'rendered image of {prompt}', image.detach().numpy(), iteration, dataformats='HWC')

        optimizer.step()
        lr_scheduler.step(score)

        score = None
        image = None
        optimizer.zero_grad()
        torch.cuda.empty_cache()


    stats = {
        "scores": scores,
        "latents": latents,
        "images": images
    }

    writer.add_hparams(hparams, {'Best score': best_score})
    writer.close()

    return best_latent.detach(), best_score.detach(), stats
    
    
    