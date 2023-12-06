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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

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
    image_preprocessed = clip_tensor_preprocessor(image_c_first).unsqueeze(0)

    prompt_tokenized = clip.tokenize(prompt).to(device)

    return model(image_preprocessed, prompt_tokenized)[0], torch.clone(image)


def get_latent_from_text(prompt, init_lat=None):
    if init_lat is None:
        lat_mean, lat_std = get_latent_mean_std()
        lat_rep = (torch.randn(lat_mean.shape) * lat_std * 0.85 + lat_mean).detach().requires_grad_(True)
    else:
        lat_rep = init_lat.requires_grad_(True)

    hparams = {
        'resolution': 50,
        'n_iterations': 10,
        'optimizer_lr': 0.03,
        'lr_scheduler_factor': 0.1,
        'lr_scheduler_patience': 2,
        'lr_scheduler_min_lr': 1e-5
    }

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
        "camera_distance": 0.25,
        "camera_angle": 45.,
        "focal_length": 1.58,
        "max_ray_length": (0.25 + 1) * 1.58 + 1.5,
        # Image
        "resolution_y": hparams['resolution'],
        "resolution_x": hparams['resolution']
    }
    phong_params = {
        "ambient_coeff": 0.45,
        "diffuse_coeff": 0.86,
        "specular_coeff": 0.59,
        "shininess": 1.,
        # Colors
        "object_color": torch.tensor([0.68, 0.38, 0.66]),
        "background_color": torch.tensor([0.29, 0.68, 0.35])
    }

    light_params = {
        "amb_light_color": torch.tensor([0.72, 0.09, 0.51]),
        # light 1
        "light_intensity_1": 1.53,
        "light_color_1": torch.tensor([0.85, 0.98, 0.84]),
        "light_dir_1": torch.tensor([-0.55, -0.30, -0.81]),
        # light p
        "light_intensity_p": 1.22,
        "light_color_p": torch.tensor([0.85, 0.98, 0.84]),
        "light_pos_p": torch.tensor([0., -3.28, 0.56])
    }

    now = datetime.now()
    writer = SummaryWriter(log_dir=f'runs/identity-pipeline/train-time:{now.strftime("%Y-%m-%d-%H:%M:%S")}')

    scores = []
    latents = []
    images = []
    for iteration in tqdm(range(hparams['n_iterations'])):
        optimizer.zero_grad()
        score, image = forward(lat_rep, prompt, camera_params, phong_params, light_params)

        scores.append(score.detach())
        latents.append(torch.clone(lat_rep))
        images.append(image)

        score.backward()
        print(f"update step {iteration+1} - score: {score}")
 
        optimizer.step()
        lr_scheduler.step(score)

        writer.add_scalar('CLIP score', score, iteration)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], iteration)
        writer.add_image('rendered_image', image.detach().numpy(), iteration, dataformats='HWC')

    stats = {
        "scores": scores,
        "latents": latents,
        "images": images
    }

    writer.add_hparams(hparams, {'Final score': score})
    writer.close()

    return lat_rep.detach(), stats
    
    
    