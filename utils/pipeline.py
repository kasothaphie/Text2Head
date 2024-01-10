import yaml
import torch
from torch.optim import Adam
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode
import numpy as np
import clip
import os
import os.path as osp
from NPHM.models.EnsembledDeepSDF import FastEnsembleDeepSDFMirrored
from NPHM import env_paths
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import uuid

from torch.profiler import profile, record_function, ProfilerActivity

from utils.render import render
from utils.similarity import CLIP_similarity, DINO_similarity


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
model_checkpoint = torch.load(path, map_location=device)
decoder_shape.load_state_dict(model_checkpoint['decoder_state_dict'], strict=True)

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
    delta = lat_rep.cpu() - lat_mean
    prob_score = -delta.T @ torch.inverse(cov) @ delta

    image_preprocessed = None
    prompt_tokenized = None
    return CLIP_score, prob_score, torch.clone(image)


def energy_level(lat_rep_1, lat_rep_2, prompt, hparams, steps=100):
    camera_params = {
        "camera_distance": 1.,
        "camera_angle": 45.,
        "focal_length": 10.,
        "max_ray_length": 3,
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
    
    with torch.no_grad():
        lat_reps = [torch.lerp(lat_rep_1, lat_rep_2, i) for i in torch.linspace(0., 1., steps)]
        forwards = [forward(lat_rep, prompt, camera_params, phong_params, light_params) for lat_rep in lat_reps]
        energy = [f[0] + hparams['lambda'] * f[1] for f in forwards]
    
    return energy
    
    

def get_latent_from_text(prompt, hparams, init_lat=None, CLIP_gt=None, DINO_gt=None):
    if init_lat is None:
        lat_mean, lat_std = get_latent_mean_std()
        lat_rep = (torch.randn(lat_mean.shape) * lat_std * 0.85 + lat_mean).detach()
    else:
        lat_mean, lat_std = get_latent_mean_std()
        lat_rep = init_lat
        
    lat_rep = lat_rep.to(device).requires_grad_(True)

    optimizer = Adam(params=[lat_rep],
                     lr=hparams['optimizer_lr'],
                     weight_decay=hparams['optimizer_weight_decay'],
                     maximize=True)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=hparams['lr_scheduler_factor'],
        patience=hparams['lr_scheduler_patience'],
        min_lr=hparams['lr_scheduler_min_lr']
    )

    camera_params = {
        "camera_distance": 1.,
        "camera_angle": 45.,
        "focal_length": 10.,
        "max_ray_length": 3,
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
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    writer = SummaryWriter(log_dir=f'../runs/identity-pipeline/train-time:{now}')

    # Hparam Optimization Mode
    #trial_uuid = str(uuid.uuid4())
    #writer = SummaryWriter(log_dir=f'../runs/identity_pipeline/hparamtuning-{trial_uuid}')

    scores = []
    latents = []
    images = []
    best_score = torch.tensor([0]).cpu()
    torch.cuda.empty_cache()
    optimizer.zero_grad()
    
    #prof = profile(
        #schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #    on_trace_ready=torch.profiler.tensorboard_trace_handler('../runs/profile/memory'),
    #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #    record_shapes=True,
    #    profile_memory=True,
    #    use_cuda=True,
    #    with_stack=True)

    #prof.start()
    for iteration in tqdm(range(hparams['n_iterations'])):
    #prof.step()
        #random_number = torch.rand(1).item()
        random_number = 0.7
        if random_number >= 0.5:
            camera_params["camera_angle"] = 45.
            light_params["light_dir_1"] = torch.tensor([-0.6, -0.4, -0.67])
            light_params["light_pos_p"] = torch.tensor([1.19, -1.27, 2.24])
        else:
            camera_params["camera_angle"] = -45.
            light_params["light_dir_1"] = torch.tensor([0.6, -0.4, -0.67])
            light_params["light_pos_p"] = torch.tensor([-1.19, -1.27, 2.24])

        with torch.no_grad():    
            _, _, mean_image = forward(lat_mean, prompt, camera_params, phong_params, light_params) # validation delta similarity
        CLIP_score, log_prob_score, image = forward(lat_rep, prompt, camera_params, phong_params, light_params)
        score = CLIP_score + hparams['lambda'] * log_prob_score

        scores.append(score.detach().cpu())
        latents.append(torch.clone(lat_rep).cpu())
        images.append(image.cpu())

        if score > best_score:
            best_score = score.detach().cpu()
            best_CLIP_score = CLIP_score.detach().cpu()
            best_prob_score = log_prob_score.detach().cpu()
            best_latent = torch.clone(lat_rep).cpu()
        
        if CLIP_gt != None:
            CLIP_gt_similarity, CLIP_delta_sim = CLIP_similarity(image, CLIP_gt, mean_image)
            writer.add_scalar('CLIP similarity to ground truth image', CLIP_gt_similarity, iteration)
            writer.add_scalar('CLIP delta similarity', CLIP_delta_sim, iteration)

        if DINO_gt != None:
            DINO_gt_similarity, DINO_delta_sim = DINO_similarity(image, DINO_gt, mean_image)
            writer.add_scalar('DINO similarity to ground truth image', DINO_gt_similarity, iteration)
            writer.add_scalar('DINO delta similarity', DINO_delta_sim, iteration)

        score.backward()
        ##print(f"update step {iteration+1} - score: {score.detach().numpy()}")

        writer.add_scalar('Score', score, iteration)
        writer.add_scalar('CLIP score', CLIP_score, iteration)
        writer.add_scalar('log prob score', log_prob_score, iteration)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], iteration)
        if ((iteration == 0) or (iteration % 10 == 0)):
            writer.add_image(f'rendered image of {prompt}', image.detach().numpy(), iteration, dataformats='HWC')

        optimizer.step()
        lr_scheduler.step(score)

        score = None
        image = None
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    #prof.stop()
    stats = {
        "scores": scores,
        "latents": latents,
        "images": images
    }

    writer.add_hparams(hparams, {'Best score': best_score})
    writer.close()

    return best_latent.detach(), best_CLIP_score, best_prob_score, stats
    
    
    