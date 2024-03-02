import yaml
import torch
import sys
import json
from torch.optim import Adam
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode, GaussianBlur
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
import numpy as np
import clip
#import open_clip
import os.path as osp
from NPHM.models.EnsembledDeepSDF import FastEnsembleDeepSDFMirrored
from NPHM import env_paths
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from nphm_tum import env_paths as mono_env_paths
from nphm_tum.models.neural3dmm import construct_n3dmm, load_checkpoint

from torch.profiler import profile, record_function, ProfilerActivity

from utils.render import render
from utils.similarity import CLIP_similarity, DINO_similarity
from utils.EMA import EMA


device = "cuda" if torch.cuda.is_available() else "cpu"
mode = "mono_nphm" # nphm, else mono_nphm
# --- Specify what you want to optimize! ---
opt_vars = ['geo', 'app'] # add 'exp' and/or 'app' (['geo', 'exp', 'app'])
grad_vars = ['geo'] # you can only skip exp and app

CLIP_model, CLIP_preprocess = clip.load("ViT-B/32", device="cpu")
#SigLIPmodel = open_clip.create_model("ViT-B-16-SigLIP", pretrained='webli', device="cpu")
#tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')

# TODO: Can we remove 'nphm' mode?
if mode == "nphm":
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
    
    def sdf(sdf_inputs, lat_rep):
        lat_rep_in = torch.reshape(lat_rep[0], (1, 1, -1))
        return decoder_shape(sdf_inputs, lat_rep_in, None)[0]
    
elif mode == "mono_nphm":
    weight_dir_shape = mono_env_paths.EXPERIMENT_DIR_REMOTE + '/'
    fname_shape = weight_dir_shape + 'configs.yaml'
    with open(fname_shape, 'r') as f:
        CFG = yaml.safe_load(f)

    # load participant IDs that were used for training
    fname_subject_index = f"{weight_dir_shape}/subject_train_index.json"
    with open(fname_subject_index, 'r') as f:
        subject_index = json.load(f)

    # load expression indices that were used for training
    fname_subject_index = f"{weight_dir_shape}/expression_train_index.json"
    with open(fname_subject_index, 'r') as f:
        expression_index = json.load(f)


    device = torch.device("cuda")
    modalities = ['geo', 'exp', 'app']
    n_lats = [len(subject_index), len(expression_index), len(subject_index)]

    neural_3dmm, latent_codes = construct_n3dmm(
        cfg=CFG,
        modalities=modalities,
        n_latents=n_lats,
        device=device,
        neutral_only=False,
        include_color_branch=True,
        skip_exp_grads= ('exp' not in grad_vars)
    )

    # load checkpoint from trained NPHM model, including the latent codes
    ckpt_path = osp.join(weight_dir_shape, 'checkpoints/checkpoint_epoch_2500.tar')
    load_checkpoint(ckpt_path, neural_3dmm, latent_codes)
        
    def sdf(sdf_inputs, lat_geo, lat_exp, lat_app):
        dict_in = {
            "queries":sdf_inputs
        }

        cond = {
            "geo": torch.reshape(lat_geo * geo_std, (1, 1, -1)),
            "exp": torch.reshape(lat_exp * exp_std, (1, 1, -1)),
            "app": torch.reshape(lat_app * app_std, (1, 1, -1))
        }
        dict_out = neural_3dmm(dict_in, cond)
        return dict_out["sdf"], dict_out["color"]

else:
    raise ValueError(f"unknown mode: {mode}")

#from clip preprocessing
clip_tensor_preprocessor = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=None),
    CenterCrop(224),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

if mode == "nphm":
    lat_mean = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_mean.npy'))
    lat_std = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_std.npy'))
elif mode == "mono_nphm":
    geo_mean = latent_codes.codebook['geo'].embedding.weight.mean(dim=0).detach()
    geo_std = latent_codes.codebook['geo'].embedding.weight.std(dim=0).detach()
    exp_mean = latent_codes.codebook['exp'].embedding.weight.mean(dim=0).detach()
    exp_std = latent_codes.codebook['exp'].embedding.weight.std(dim=0).detach()
    app_mean = latent_codes.codebook['app'].embedding.weight.mean(dim=0).detach()
    app_std = latent_codes.codebook['app'].embedding.weight.std(dim=0).detach()
else:
    raise ValueError(f"unknown mode: {mode}")

def loss_fn(clip_score, prob_geo_score, prob_exp_score, prob_app_score, hparams):
    if 'geo' in opt_vars:
        lambda_geo = hparams["lambda_geo"]
    else:
        lambda_geo = 0
    if 'exp' in opt_vars:
        lambda_exp = hparams["lambda_exp"]
    else:
        lambda_exp = 0
    if 'app' in opt_vars:
        lambda_app = hparams["lambda_app"]
    else:
        lambda_app = 0

    loss = clip_score
    #loss += lambda_geo * prob_geo_score
    #loss += lambda_exp * prob_exp_score
    #loss += lambda_app * prob_app_score
    #loss /= 1 + lambda_geo + lambda_exp + lambda_app

    return loss

def get_image_clip_embedding(lat_rep, camera_params, phong_params, light_params, with_app_grad, color):
    image = render(sdf, lat_rep, camera_params, phong_params, light_params, color, with_app_grad=with_app_grad)
    image_c_first = image.permute(2, 0, 1)
    image_preprocessed = clip_tensor_preprocessor(image_c_first).unsqueeze(0).cpu()
    CLIP_embedding_image = CLIP_model.encode_image(image_preprocessed) # [1, 512]
    #CLIP_embedding_image = SigLIPmodel.encode_image(image_preprocessed)
    normalized_CLIP_embedding_image = CLIP_embedding_image / CLIP_embedding_image.norm(dim=-1, keepdim=True)
    
    return normalized_CLIP_embedding_image, image

def get_text_clip_embedding(prompt):
    prompt_tokenized = clip.tokenize(prompt).cpu()
    #prompt_tokenized = tokenizer(prompt).cpu()
    text_embedded = CLIP_model.encode_text(prompt_tokenized)
    #text_embedded = SigLIPmodel.encode_text(prompt_tokenized)
    normalized_CLIP_embedding_text = text_embedded / text_embedded.norm(dim=-1, keepdim=True)
    
    return normalized_CLIP_embedding_text

def clip_score(image_embedding, text_embedding):
    return 100 * torch.matmul(image_embedding, text_embedding.T)

def std_factor_score(lat_rep):
    distance = lat_rep.cpu() - lat_mean.cpu()
    weighted_distance = distance / lat_std
    distance_abs = weighted_distance.abs()
    std_factor_score = distance_abs.mean()
    return std_factor_score

def penalty_score(lat_rep, a=0.065, b=2.5):
    cov = lat_std * torch.eye(lat_mean.shape[0])
    distance = lat_rep.cpu() - lat_mean.cpu()
    weighted_distance = torch.inverse(cov.cpu()) @ distance
    distance_abs = weighted_distance.abs()

    penalized_tensor = -a * torch.exp(b * distance_abs)
    penalty = penalized_tensor.mean()
    return penalty

def latent_norms(lat_rep):
    
    # --- Probability of latent geometry code ---
    #geo_rep = torch.reshape(lat_rep[0], (-1,)) # [1344] 
    #cov_geo = (1/(geo_std.cpu()**2)) * torch.eye(geo_mean.shape[0])
    #delta_geo = geo_rep.cpu() - geo_mean.cpu() 
    #prob_geo = -delta_geo.T @ cov_geo @ delta_geo
    
    #lat_geos = lat_rep[0].split([64, *([32] * 66)])
    #geo_stds = geo_std.split([64, *([32] * 66)])
    #lat_geo_prob_scores = torch.stack([torch.norm(sub_lat_geo) for sub_lat_geo in lat_geos])
    #lat_geo_prob_scores[0] = lat_geo_prob_scores[0] * 10
    #lat_std_means = torch.stack([torch.mean(sub_geo_std) for sub_geo_std in geo_stds])
    #lat_std_means = lat_std_means / torch.max(lat_std_means)
    #prob_geo = (lat_geo_prob_scores @ lat_std_means**2).cpu()
    
    #prob_geo = lat_geo_prob_scores[0].cpu() + lat_geo_prob_scores[1:].mean().cpu()
    prob_geo = -torch.norm(lat_rep[0]).cpu()
    prob_exp = -torch.norm(lat_rep[1]).cpu()
    prob_app = -torch.norm(lat_rep[2]).cpu()
    
    #prob_geo = -torch.norm(lat_rep[0] * torch.exp(geo_std / torch.max(geo_std))).cpu()
    #prob_geo = -torch.norm(lat_rep[0] * (geo_std / torch.max(geo_std))).cpu()
    # --- Probability of latent expression code ---
    #exp_rep = torch.reshape(lat_rep[1], (-1,)) # [200] 
    #cov_exp = torch.log(1/exp_std.cpu()) * torch.eye(exp_mean.shape[0])
    #delta_exp = exp_rep.cpu() - exp_mean.cpu() 
    #prob_exp = -delta_exp.T @ inv_exp_cov @ delta_exp
    # --- Probability of latent expression code ---
    #app_rep = torch.reshape(lat_rep[2], (-1,)) # [200] 
    #cov_app = app_std.cpu() * torch.eye(app_mean.shape[0])
    #delta_app = app_rep.cpu() - app_mean.cpu() 
    #prob_app = -delta_app.T @ inv_app_cov @ delta_app
    
    return prob_geo, prob_exp, prob_app

def forward(lat_rep, prompt, camera_params, phong_params, light_params, with_app_grad, color):
    # --- Render Image from current Lat Rep + Embedd ---
    image_embedding, image = get_image_clip_embedding(lat_rep, camera_params, phong_params, light_params, with_app_grad, color)
    # for debugging
    #plt.imshow(image.detach().numpy())
    #plt.axis('off')  # Turn off axes
    #plt.show()

    # --- Text Embedding ---
    text_embedded_normalized = get_text_clip_embedding(prompt)
    
    # --- CLIP Score ---
    CLIP_score = clip_score(image_embedding, text_embedded_normalized)
    
    # --- Log Prob Score ---
    norm_geo, norm_exp, norm_app = latent_norms(lat_rep)

    return CLIP_score, norm_geo, norm_exp, norm_app, torch.clone(image)

def energy_level(lat_rep_1, lat_rep_2, prompt, hparams, steps=100):
    with torch.no_grad():
        lat_rep_1 = lat_rep_1.cpu()
        lat_rep_2 = lat_rep_2.cpu()
        lat_reps = [torch.lerp(lat_rep_1, lat_rep_2, i) for i in torch.linspace(0., 1., steps)]
        forwards = [batch_forward(lat_rep.to(device), prompt, hparams) for lat_rep in lat_reps]
        energy = [loss_fn(f[0], f[1], hparams) for f in forwards]
    
    return energy, forwards

def get_augmented_latents(lat_rep, hparams):
    # --- Latent Representation Augmentation ---
    # Generate random values from a normal distribution with standard deviation alpha
    # TODO: This shift is not used for now b/c no positive impact --> to be verified!
    lat_geo_aug = lat_rep[0] + torch.randn_like(lat_rep[0], device=lat_rep[0].device) * hparams["alpha"]
    
    if 'exp' in opt_vars:
        lat_exp_aug = lat_rep[1] + torch.randn_like(lat_rep[1], device=lat_rep[1].device) * hparams["alpha"]
    else:
        lat_exp_aug = lat_rep[1]
        
    if 'app' in opt_vars:
        lat_app_aug = lat_rep[2] + torch.randn_like(lat_rep[2], device=lat_rep[2].device) * hparams["alpha"]
    else:
        lat_app_aug = lat_rep[2]

    lat_rep_aug = [lat_geo_aug, lat_exp_aug, lat_app_aug]
    return lat_rep_aug

def get_augmented_params_color(lat_rep, hparams):
    lat_rep_aug = get_augmented_latents(lat_rep, hparams)


    # --- Camera Parameters Augmentation ---
    camera_distance_factor = torch.randn(1).item() * 0.05 + 0.2 # originally: random value [0.2, 0.35]
    angle_rho = torch.randn(1) * 25
    angle_theta = torch.randn(1) * 10 # sample angle around 0, std 10
    camera_params_aug = {
        "camera_distance": camera_distance_factor * 2.57,
        "camera_angle_rho": angle_rho,
        "camera_angle_theta": angle_theta,
        "focal_length": 2.57,
        "max_ray_length": 3,
        # Image
        "resolution_y": hparams["resolution"],
        "resolution_x": hparams["resolution"]
    }
    # TODO: test whether rendering parameter augmentation has a positive impact on optimization with color
    #_, phong_params_aug, light_params_aug = get_optimal_params_color(hparams)

    # --- Phong Parameters Augmentation ---
    amb_coeff = torch.rand(1).item() * 0.28 + 0.1 #random value [0.1, 0.38]
    diffuse_coeff = torch.rand(1).item() * 0.2 + 0.8 #random value [0.8, 1.]
    specular_coeff = torch.rand(1).item() * 0.33 + 0.1 #random value [0.1, 0.43]
    background_color_0 = torch.rand(1).item() * 0.14 + 0.68 #random value [0.68, 0.82]
    background_color_1 = torch.rand(1).item() * 0.09 + 0.84 #random value [0.84, 0.93]
    background_color_2 = torch.rand(1).item() * 0.32 + 0.13 #random value [0.13, 0.45]
    phong_params_aug = {
        "ambient_coeff": amb_coeff,
        "diffuse_coeff": diffuse_coeff,
        "specular_coeff": specular_coeff,
        "shininess": 25,
        # Colors
        "object_color": None,
        "background_color": torch.tensor([background_color_0, background_color_1, background_color_2])
    }

    # --- Light Parameters Augmentation ---
    amb_color_0 = torch.rand(1).item() * 0.07 + 0.58 #random value [0.58, 0.65]
    amb_color_1 = torch.rand(1).item() * 0.19 + 0.53 #random value [0.53, 0.72]
    amb_color_2 = torch.rand(1).item() * 0.21 + 0.62 #random value [0.62, 0.83]
    light_intensity_1 = torch.rand(1).item() * 0.7 + 1.1 #random value [1.1, 1.8]
    light_intensity_p = torch.rand(1).item() * 0.3 + 0.2 #random value [0.2, 0.5]
    # switch light directions depending on view point
    if angle_rho >= 0:
        if angle_rho >= 15:
            light_dir_0 = torch.rand(1).item() * 0.11 - 0.95 #random value [-0.84, -0.95]
        else:
            light_dir_0 = torch.rand(1).item() * 0.1 - 0.05 #random value [-0.05, 0.05]
        light_pos_0 = 0.17
    else:
        if angle_rho <= -15:
            light_dir_0 = torch.rand(1).item() * 0.11 + 0.84 #random value [0.84, 0.95]
        else:
            light_dir_0 = torch.rand(1).item() * 0.1 - 0.05 #random value [-0.05, 0.05]
        light_pos_0 = -0.17
    light_dir_1 = torch.rand(1).item() * 0.33 - 0.5 #random value [-0.17, -0.5]
    light_dir_2 = torch.rand(1).item() * 0.2 - 0.84 #random value [-0.64, -0.84]
    light_params_aug = {
        "amb_light_color": torch.tensor([amb_color_0, amb_color_1, amb_color_2]),
        # light 1
        "light_intensity_1": light_intensity_1,
        "light_color_1": torch.tensor([1., 0.95, 0.9]),
        "light_dir_1": torch.tensor([light_dir_0, light_dir_1, light_dir_2]),
        # light p
        "light_intensity_p": light_intensity_p,
        "light_color_p": torch.tensor([1., 0.95, 0.9]),
        "light_pos_p": torch.tensor([light_pos_0, 2.77, -2.25])
    }

    return lat_rep_aug, camera_params_aug, phong_params_aug, light_params_aug

def get_augmented_params_no_color(lat_rep, hparams):
    lat_rep_aug = get_augmented_latents(lat_rep, hparams)

    # --- Camera Parameters Augmentation ---
    camera_distance_factor = torch.randn(1).item() * 0.07 + 0.21 # originally: random value [0.2, 0.25]
    angle_rho = torch.randn(1) * 25 
    angle_theta = torch.randn(1) * 10 # sample angle around 0, std 10
    camera_params_aug = {
        "camera_distance": camera_distance_factor * 2.57,
        "camera_angle_rho": angle_rho,
        "camera_angle_theta": angle_theta,
        "focal_length": 2.57,
        "max_ray_length": 3,
        # Image
        "resolution_y": hparams["resolution"],
        "resolution_x": hparams["resolution"]
    }

    #_, phong_params_aug, light_params_aug = get_optimal_params(hparams)


    # --- Phong Parameters Augmentation ---
    amb_coeff = torch.rand(1).item() * 0.06 + 0.47 #random value [0.47, 0.53]
    diffuse_coeff = torch.rand(1).item() * 0.1 + 0.7 #random value [0.7, 0.8]
    specular_coeff = torch.rand(1).item() * 0.08 + 0.59 #random value [0.59, 0.67]
    object_color_0 = torch.rand(1).item() * 0.1 + 0.53 #random value [0.53, 0.63]
    object_color_1 = torch.rand(1).item() * 0.13 + 0.19 #random value [0.19, 0.32]
    object_color_2 = torch.rand(1).item() * 0.06 + 0.63 #random value [0.63, 0.69]
    background_color_0 = torch.rand(1).item() * 0.05 + 0.34 #random value [0.34, 0.39]
    background_color_1 = torch.rand(1).item() * 0.08 + 0.7 #random value [0.7, 0.78]
    background_color_2 = torch.rand(1).item() * 0.04 + 0.27 #random value [0.27, 0.31]
    phong_params_aug = {
        "ambient_coeff": amb_coeff,
        "diffuse_coeff": diffuse_coeff,
        "specular_coeff": specular_coeff,
        "shininess": 0.5,
        # Colors
        "object_color": torch.tensor([object_color_0, object_color_1, object_color_2]),
        "background_color": torch.tensor([background_color_0, background_color_1, background_color_2])
    }

    # --- Light Parameters Augmentation ---
    amb_color_0 = torch.rand(1).item() * 0.1 + 0.8 #random value [0.8, 0.9]
    amb_color_1 = torch.rand(1).item() * 0.06 + 0.1 #random value [0.1, 0.16]
    amb_color_2 = torch.rand(1).item() * 0.05 + 0.54 #random value [0.54, 0.59]
    light_intensity_1 = torch.rand(1).item() * 0.3 + 1.3 #random value [1.3, 1.6]
    light_intensity_p = torch.rand(1).item() * 0.26 + 0.58 #random value [0.58, 0.84]
    # switch light directions depending on view point
    if angle_rho >= 0:
        if angle_rho >= 15:
            light_dir_0 = torch.rand(1).item() * 0.05 - 0.6 #random value [-0.6, -0.55]
        else:
            light_dir_0 = torch.rand(1).item() * 0.1 - 0.05 #random value [-0.05, 0.05]
        light_pos_0 = 1.19
    else:
        if angle_rho <= -15:
            light_dir_0 = torch.rand(1).item() * 0.05 + 0.55 #random value [0.55, 0.6]
        else:
            light_dir_0 = torch.rand(1).item() * 0.1 - 0.05 #random value [-0.05, 0.05]
        light_pos_0 = -1.19
    light_dir_1 = torch.rand(1).item() * 0.05 - 0.43 #random value [-0.43, -0.38]
    light_dir_2 = torch.rand(1).item() * 0.1 - 0.71 #random value [-0.71, -0.61]
    light_params_aug = {
        "amb_light_color": torch.tensor([amb_color_0, amb_color_1, amb_color_2]),
        # light 1
        "light_intensity_1": light_intensity_1,
        "light_color_1": torch.tensor([0.8, 0.97, 0.89]),
        "light_dir_1": torch.tensor([light_dir_0, light_dir_1, light_dir_2]),
        # light p
        "light_intensity_p": light_intensity_p,
        "light_color_p": torch.tensor([0.8, 0.97, 0.89]),
        "light_pos_p": torch.tensor([light_pos_0, -1.27, 2.24])
    }

    return lat_rep_aug, camera_params_aug, phong_params_aug, light_params_aug
        

def batch_forward(lat_rep_orig, prompt, hparams, with_app_grad):
    all_delta_CLIP_scores = []
    
    for i in range(hparams['batch_size']):

        if torch.rand(1) <= hparams['color_prob']:
            # --- Random Augmentation ---
            lat_rep, camera_params, phong_params, light_params = get_augmented_params_color(lat_rep_orig, hparams)
            # --- Forward Pass ---
            delta_CLIP_score, _, _, _, _ = forward(lat_rep, prompt, camera_params, phong_params, light_params, with_app_grad, color=True)
        else:
            # --- Random Augmentation ---
            lat_rep, camera_params, phong_params, light_params = get_augmented_params_no_color(lat_rep_orig, hparams)
            # --- Forward Pass ---
            delta_CLIP_score, _, _, _, _ = forward(lat_rep, prompt, camera_params, phong_params, light_params, with_app_grad, color=False)

        
        # --- Compute Scores ---
        all_delta_CLIP_scores.append(delta_CLIP_score)
    
    all_delta_CLIP_scores_tensor = torch.stack(all_delta_CLIP_scores)
    batch_delta_CLIP_score = torch.mean(all_delta_CLIP_scores_tensor)
    
    norm_geo, norm_exp, norm_app = latent_norms(lat_rep_orig)
    

    return batch_delta_CLIP_score, norm_geo, norm_exp, norm_app

# Optimal Params for rendering with color
def get_optimal_params_color(hparams):
    camera_params = {
            "camera_distance": 0.21 * 2.57,
            "camera_angle_rho": 15., #45.,
            "camera_angle_theta": 0.,
            "focal_length": 2.57,
            "max_ray_length": 3,
            # Image
            "resolution_y": 180,
            "resolution_x": 180
        }
    
    phong_params = {
            "ambient_coeff": 0.32,
            "diffuse_coeff": 0.85,
            "specular_coeff": 0.34,
            "shininess": 25,
            # Colors
            "object_color": torch.tensor([0.53, 0.24, 0.64]),
            "background_color": torch.tensor([0.75, 0.85, 0.31])
        }

    light_params = {
            "amb_light_color": torch.tensor([0.65, 0.66, 0.69]),
            # light 1
            "light_intensity_1": 1.69,
            "light_color_1": torch.tensor([1., 0.91, 0.88]),
            "light_dir_1": torch.tensor([0, -0.18, -0.8]),
            # light p
            "light_intensity_p": 0.52,
            "light_color_p": torch.tensor([1., 0.91, 0.88]),
            "light_pos_p": torch.tensor([0.17, 2.77, -2.25])
    }
    return camera_params, phong_params, light_params


# Optimial Params for Rendering without color
def get_optimal_params(hparams):
    camera_params = {
        "camera_distance": 0.21 * 2.57,
        "camera_angle_rho": 45.,
        "camera_angle_theta": 0.,
        "focal_length": 2.57,
        "max_ray_length": 3,
        # Image
        "resolution_y": 180,
        "resolution_x": 180
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
    
    return camera_params, phong_params, light_params

def initial_latent_sampling(prompt, _vars, hparams, n_samples=5):
    best_clip = -torch.inf
    best_latents = None
    for i in range(n_samples):
        if 'geo' in _vars:
            lat_geo = torch.randn_like(geo_mean) * 0.85
        else:
            lat_geo = geo_mean
        if 'exp' in _vars:
            lat_exp = torch.randn_like(exp_mean) * 0.85
        else:
            lat_exp = exp_mean
        if 'app' in _vars:
            lat_app = torch.randn_like(app_mean) * 0.85
        else:
            lat_app = app_mean
            
        with torch.no_grad():
            clip_score = forward([lat_geo, lat_exp, lat_app], prompt, *get_optimal_params_color(hparams), with_app_grad=False, color=True)[0]
        print(f"Sample {i}: {clip_score}")
        if clip_score > best_clip:
            best_latents = [lat_geo, lat_exp, lat_app]
            best_clip = clip_score
            
    print(f"Best Latent has {best_clip}")
    return best_latents


def get_latent_from_text(prompt, hparams, init_lat=None, CLIP_gt=None, DINO_gt=None):

    global geo_mean, geo_std, exp_mean, exp_std, app_mean, app_std

    print('######## SETTINGS ########')
    print('Optimization Mode: ', opt_vars)
    print('Models considered for backpropagating gradients:', grad_vars)
    print('###########################')

    if init_lat is None:
        best_sampled_latents = initial_latent_sampling(prompt, opt_vars, hparams)
        init_geo = best_sampled_latents[0]
        init_exp = best_sampled_latents[1]
        init_app = best_sampled_latents[2]
    else:
        init_geo = init_lat[0]
        init_exp = init_lat[1]
        init_app = init_lat[2]

    lat_geo = init_geo.clone().detach().to(device).requires_grad_(True)
    lat_exp = init_exp.clone().detach().to(device).requires_grad_(True)
    lat_app = init_app.clone().detach().to(device).requires_grad_(True)
    lat_mean = [torch.zeros_like(geo_mean), torch.zeros_like(exp_mean), torch.zeros_like(app_mean)]

    # --- Get Mean Image (required for CLIP and DINO validation) ---
    camera_params_opti, phong_params_opti, light_params_opti = get_optimal_params(hparams)
    camera_params_opti_c, phong_params_opti_c, light_params_opti_c = get_optimal_params_color(hparams)
    if (CLIP_gt != None) or (DINO_gt != None):
        with torch.no_grad():
            mean_image = render(sdf, lat_mean, camera_params_opti, phong_params_opti, light_params_opti, color=False)


    opt_params = list(filter(lambda x: x is not None,[
        lat_geo if 'geo' in opt_vars else None,
        lat_exp if 'exp' in opt_vars else None,
        #lat_app if 'app' in opt_vars else None,
    ]))
    
    optimizer_geo_exp = torch.optim.AdamW(params=opt_params,
                    lr=hparams['optimizer_lr'],
                    betas=(hparams['optimizer_beta1'], 0.999),
                    weight_decay=hparams['lambda'],
                    maximize=True)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_geo_exp,
        mode='max',
        factor=hparams['lr_scheduler_factor'],
        patience=hparams['lr_scheduler_patience'],
        min_lr=hparams['lr_scheduler_min_lr']
    )
    
    if 'app' in opt_vars:
        optimizer_app = torch.optim.AdamW(params=[lat_app],
                                          lr=hparams['optimizer_lr'],
                                          betas=(hparams['optimizer_beta1'], 0.999),
                                          weight_decay=hparams['lambda'],
                                          maximize=True)
        
        lr_scheduler_app = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_app,
            mode='max',
            factor=hparams['lr_scheduler_factor'],
            patience=hparams['lr_scheduler_patience'],
            min_lr=hparams['lr_scheduler_min_lr']
    )

    # Normal Mode
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    writer = SummaryWriter(log_dir=f'../runs/{hparams["exp_name"]}/train-time:{now}')

    best_score = torch.tensor([-torch.inf]).cpu()
    best_clip_score = torch.tensor([-torch.inf]).cpu()
    torch.cuda.empty_cache()
    optimizer_geo_exp.zero_grad()
    
    #prof = profile(
        #schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #    on_trace_ready=torch.profiler.tensorboard_trace_handler('../runs/profile/memory'),
    #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #    record_shapes=True,
    #    profile_memory=True,
    #    use_cuda=True,
    #    with_stack=True)
            
    lat_rep = [
        lat_geo if 'geo' in opt_vars else geo_mean,
        lat_exp if 'exp' in opt_vars else exp_mean,
        lat_app if 'app' in opt_vars else app_mean,
    ]
    
    step_geo = torch.zeros_like(geo_mean)
    step_exp = torch.zeros_like(exp_mean)
    step_app = torch.zeros_like(app_mean)
    #prof.start()
    for iteration in tqdm(range(hparams['n_iterations'])):
    #prof.step()
        hparams["iteration"] = iteration
        
        lat_geo_old = torch.clone(lat_rep[0]).detach().cpu()
        lat_exp_old = torch.clone(lat_rep[1]).detach().cpu()
        lat_app_old = torch.clone(lat_rep[2]).detach().cpu()
        prev_step_geo = step_geo
        prev_step_exp = step_exp
        prev_step_app = step_app
        
        if 'app' in opt_vars:
            lat_rep_aug, camera_params_aug, phong_params_aug, light_params_aug = get_augmented_params_color(lat_rep, hparams)
            CLIP_score_app, _, _, _, _ = forward(lat_rep_aug, prompt, camera_params_aug, phong_params_aug, light_params_aug, with_app_grad=True, color=True)
            optimizer_app.zero_grad()
            CLIP_score_app.backward()
            optimizer_app.step()
            lr_scheduler_app.step(CLIP_score_app)

        batch_CLIP_score, norm_geo, norm_exp, norm_app = batch_forward(lat_rep, prompt, hparams, with_app_grad=False)

        #batch_score = loss_fn(torch.clone(batch_CLIP_score), batch_log_prob_geo_score, batch_log_prob_exp_score, batch_log_prob_app_score, hparams)
        batch_score = torch.clone(batch_CLIP_score)
        

        if batch_score > best_score:
            best_score = batch_score.detach().cpu()
            best_latent_geo = torch.clone(lat_rep[0]).cpu()
            best_latent_exp = torch.clone(lat_rep[1]).cpu()
            best_latent_app = torch.clone(lat_rep[2]).cpu()
            
        if batch_CLIP_score > best_clip_score:
            best_clip_score = batch_CLIP_score.detach().cpu()
            #best_clip_latent = torch.clone(lat_rep).cpu()

        optimizer_geo_exp.zero_grad() 
        batch_score.backward()
        sys.stdout.flush()
        

        # --- Validation with CLIP / DINO Delta Score ---
        with torch.no_grad():
            CLIP_score_no_col, _, _, _, image_no_col = forward(lat_rep, prompt, camera_params_opti, phong_params_opti, light_params_opti, with_app_grad=False, color=False)
            CLIP_score_col, _, _, _, image_col = forward(lat_rep, prompt, camera_params_opti_c, phong_params_opti_c, light_params_opti_c, with_app_grad=False, color=True)
            writer.add_scalar('CLIP Score no color', CLIP_score_no_col, iteration)
            writer.add_scalar('CLIP Score color', CLIP_score_col, iteration)
        if (iteration == 0) or ((iteration+1) % 2 == 0):
            writer.add_image(f'rendered image of {prompt}', image_no_col.detach().numpy(), iteration, dataformats='HWC')
            writer.add_image(f'textured rendered image of {prompt}', image_col.detach().numpy(), iteration, dataformats='HWC')
            
        
        if CLIP_gt != None:
            CLIP_gt_similarity, CLIP_delta_sim = CLIP_similarity(image_no_col, CLIP_gt, mean_image)
            #writer.add_scalar('CLIP similarity to ground truth image', CLIP_gt_similarity, iteration)
            writer.add_scalar('CLIP delta similarity', CLIP_delta_sim, iteration)

        if DINO_gt != None:
            DINO_gt_similarity, DINO_delta_sim = DINO_similarity(image_no_col, DINO_gt, mean_image)
            #writer.add_scalar('DINO similarity to ground truth image', DINO_gt_similarity, iteration)
            writer.add_scalar('DINO delta similarity', DINO_delta_sim, iteration)
        

        writer.add_scalar('Batch Score', batch_score, iteration)
        writer.add_scalar('Batch CLIP Score', batch_CLIP_score, iteration)
        writer.add_scalar('Norm Geometry Code', norm_geo, iteration)
        writer.add_scalar('Norm Expression Code', norm_exp, iteration)
        writer.add_scalar('Norm Appearance Code', norm_app, iteration)
        writer.add_scalar('learning rate', optimizer_geo_exp.param_groups[0]['lr'], iteration)
        if 'geo' in opt_vars:
            lat_geo.grad = lat_geo.grad.nan_to_num(0.) # TODO: is this still needed?
            gradient_lat_geo = lat_geo.grad
            writer.add_scalar('Gradient norm of Score w.r.t. Geometry Latent', gradient_lat_geo.norm(), iteration)
            
            # Difference between lat_rep and previous lat_rep
            lat_geo_new = lat_rep[0].detach().cpu()
            lat_geo_diff = torch.abs(lat_geo_new - lat_geo_old)
            mean_diff = torch.mean(lat_geo_diff.abs())
            writer.add_scalar('Mean percentual diff geo', mean_diff, iteration)
            
            
            lat_geo_fig = plt.figure()
            lat_geo_plot = lat_geo_fig.add_subplot(1,2,1)
            lat_geo_plot.plot(lat_geo.detach().cpu().numpy())
            lat_geo_std_plot = lat_geo_fig.add_subplot(1,2,2)
            lat_geo_std_plot.plot((lat_geo.detach().cpu()) * geo_std.detach().cpu())
            writer.add_figure('Geometry Latent Code', lat_geo_fig, iteration)
            writer.add_histogram('Geometry Latent Code in normalized space', lat_geo.detach().cpu(), iteration)

        if 'exp' in opt_vars:
            gradient_lat_exp = lat_exp.grad
            writer.add_scalar('Gradient norm of Score w.r.t. Expression Latent', gradient_lat_exp.norm(), iteration)
            
            # Difference between lat_rep and previous lat_rep
            lat_exp_new = lat_rep[1].detach().cpu()
            lat_exp_diff = torch.abs(lat_exp_new - lat_exp_old)
            mean_diff = torch.mean(lat_exp_diff.abs())
            writer.add_scalar('Mean percentual diff exp', mean_diff, iteration)
            
            lat_exp_fig = plt.figure()
            lat_exp_plot = lat_exp_fig.add_subplot(1,2,1)
            lat_exp_plot.plot(lat_exp.detach().cpu().numpy())
            lat_exp_std_plot = lat_exp_fig.add_subplot(1,2,2)
            lat_exp_std_plot.plot((lat_exp.detach().cpu()) * exp_std.detach().cpu())
            writer.add_figure('Expression Latent Code', lat_exp_fig, iteration)
            writer.add_histogram('Expression Latent Code in normalized space', lat_exp.detach().cpu(), iteration)

        if 'app' in opt_vars:
            gradient_lat_app = lat_app.grad
            writer.add_scalar('Gradient norm of Score w.r.t. Appearance Latent', gradient_lat_app.norm(), iteration)
            
            # Difference between lat_rep and previous lat_rep
            lat_app_new = lat_rep[2].detach().cpu()
            lat_app_diff = torch.abs(lat_app_new - lat_app_old)
            mean_diff = torch.mean(lat_app_diff.abs())
            writer.add_scalar('Mean percentual diff app', mean_diff, iteration)
            
            lat_app_fig = plt.figure()
            lat_app_plot = lat_app_fig.add_subplot(1,2,1)
            lat_app_plot.plot(lat_app.detach().cpu().numpy())
            lat_app_std_plot = lat_app_fig.add_subplot(1,2,2)
            lat_app_std_plot.plot((lat_app.detach().cpu()) * app_std.detach().cpu())
            writer.add_figure('Appearence Latent Code', lat_app_fig, iteration)
            writer.add_histogram('Appearence Latent Code in normalized space', lat_app.detach().cpu(), iteration)
        
        
        optimizer_geo_exp.step()
        lr_scheduler.step(batch_score)
        

        if False:#'geo' in opt_vars:
            # Difference between lat_rep and previous lat_rep
            lat_geo_new = lat_rep[0].detach().cpu()
            lat_geo_diff = torch.abs(lat_geo_new - lat_geo_old)
            mean_diff = torch.mean(lat_geo_diff.abs())
            writer.add_scalar('Mean percentual diff geo wrt std dev', mean_diff, iteration)

            # Angle between step and previous step
            step_geo = lat_geo_new.squeeze(0).squeeze(0) - lat_geo_old.squeeze(0).squeeze(0)
            normalized_step_geo = step_geo / step_geo.norm(dim=-1, keepdim=True)
            normalized_prev_step_geo = prev_step_geo / prev_step_geo.norm(dim=-1, keepdim=True)
            cos_theta = torch.dot(normalized_step_geo, normalized_prev_step_geo)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angle_rad = torch.acos(cos_theta)
            angle_deg = torch.rad2deg(angle_rad)
            writer.add_scalar('Angle between steps geo', angle_deg, iteration)

        if False:#'exp' in opt_vars:
            # Difference between lat_rep and previous lat_rep
            lat_exp_new = lat_rep[1].detach().cpu()
            lat_exp_diff = torch.abs(lat_exp_new - lat_exp_old)
            mean_diff = torch.mean(lat_exp_diff.abs())
            writer.add_scalar('Mean percentual diff exp wrt std dev', mean_diff, iteration)

            # Angle between lat_rep and previous lat_rep
            step_exp = lat_exp_new.squeeze(0).squeeze(0) - lat_exp_old.squeeze(0).squeeze(0)
            normalized_step_exp = step_exp / step_exp.norm(dim=-1, keepdim=True)
            normalized_prev_step_exp = prev_step_exp / prev_step_exp.norm(dim=-1, keepdim=True)
            cos_theta = torch.dot(normalized_step_exp, normalized_prev_step_exp)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angle_rad = torch.acos(cos_theta)
            angle_deg = torch.rad2deg(angle_rad)
            writer.add_scalar('Angle between steps exp', angle_deg, iteration)


        optimizer_geo_exp.zero_grad()          

    #prof.stop()

    writer.add_hparams(hparams, {
        'Best score': best_score,
        'Best CLIP score': best_clip_score
        })
    writer.close()

    return [best_latent_geo.detach(), best_latent_exp.detach(), best_latent_app.detach()], best_score
    
    
    