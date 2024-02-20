import yaml
import torch
import sys
import json
from torch.optim import Adam
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode
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
opt_vars = ['geo'] # add 'exp' and/or 'app' (['geo', 'exp', 'app'])
grad_vars = ['geo']

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
            "geo": torch.reshape(lat_geo, (1, 1, -1)),
            "exp": torch.reshape(lat_exp, (1, 1, -1)),
            "app": torch.reshape(lat_app, (1, 1, -1))
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
    loss += lambda_geo * prob_geo_score
    loss += lambda_exp * prob_exp_score
    loss += lambda_app * prob_app_score
    loss /= 1 + lambda_geo + lambda_exp + lambda_app

    return loss

def get_image_clip_embedding(lat_rep, camera_params, phong_params, light_params, color):
    image = render(sdf, lat_rep, camera_params, phong_params, light_params, color, model_grads=grad_vars)
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

def log_prop_score(lat_rep):
    
    # --- Probability of latent geometry code ---
    geo_rep = torch.reshape(lat_rep[0], (-1,)) # [1344] 
    cov_geo = geo_std.cpu() * torch.eye(geo_mean.shape[0])
    delta_geo = geo_rep.cpu() - geo_mean.cpu() 
    prob_geo = -delta_geo.T @ torch.inverse(cov_geo.cpu()) @ delta_geo
    # --- Probability of latent expression code ---
    exp_rep = torch.reshape(lat_rep[1], (-1,)) # [200] 
    cov_exp = exp_std.cpu() * torch.eye(exp_mean.shape[0])
    delta_exp = exp_rep.cpu() - exp_mean.cpu() 
    prob_exp = -delta_exp.T @ torch.inverse(cov_exp.cpu()) @ delta_exp
    # --- Probability of latent expression code ---
    app_rep = torch.reshape(lat_rep[2], (-1,)) # [200] 
    cov_app = app_std.cpu() * torch.eye(app_mean.shape[0])
    delta_app = app_rep.cpu() - app_mean.cpu() 
    prob_app = -delta_app.T @ torch.inverse(cov_app.cpu()) @ delta_app
    
    return prob_geo, prob_exp, prob_app

def forward(lat_rep, prompt, camera_params, phong_params, light_params, color):
    # --- Render Image from current Lat Rep + Embedd ---
    image_embedding, image = get_image_clip_embedding(lat_rep, camera_params, phong_params, light_params, color)
    # for debugging
    #plt.imshow(image.detach().numpy())
    #plt.axis('off')  # Turn off axes
    #plt.show()

    # --- Text Embedding ---
    text_embedded_normalized = get_text_clip_embedding(prompt)
    
    # --- CLIP Score ---
    CLIP_score = clip_score(image_embedding, text_embedded_normalized)
    
    # --- Log Prob Score ---
    prob_geo, prob_exp, prob_app = log_prop_score(lat_rep)

    return CLIP_score, prob_geo, prob_exp, prob_app, torch.clone(image)

def energy_level(lat_rep_1, lat_rep_2, prompt, hparams, steps=100):
    with torch.no_grad():
        lat_rep_1 = lat_rep_1.cpu()
        lat_rep_2 = lat_rep_2.cpu()
        lat_reps = [torch.lerp(lat_rep_1, lat_rep_2, i) for i in torch.linspace(0., 1., steps)]
        forwards = [batch_forward(lat_rep.to(device), prompt, hparams) for lat_rep in lat_reps]
        energy = [loss_fn(f[0], f[1], hparams) for f in forwards]
    
    return energy, forwards

def get_augmented_params_color(lat_rep, hparams):
    # --- Latent Representation Augmentation ---
    # Generate random values from a normal distribution with standard deviation alpha
    if 'geo' in opt_vars:
        # TODO: This shift is not used for now b/c no positive impact --> to be verified!
        random_multipliers_geo = torch.randn(geo_std.shape) * hparams["alpha"]
        shift_geo = geo_std.cpu() * random_multipliers_geo
        shift_geo = shift_geo.to(device)
        lat_geo_aug = lat_rep[0] + shift_geo
        #lat_geo_aug = lat_rep[0]
    else:
        lat_geo_aug = lat_rep[0]
    
    if 'exp' in opt_vars:
        random_multipliers_exp = torch.randn(exp_std.shape) * hparams["alpha"]
        shift_exp = exp_std.cpu() * random_multipliers_exp
        shift_exp = shift_exp.to(device)
        lat_exp_aug = lat_rep[1] + shift_exp
    else:
        lat_exp_aug = lat_rep[1]
        
    if 'app' in opt_vars:
        random_multipliers_app = torch.randn(app_std.shape) * hparams["alpha"]
        shift_app = app_std.cpu() * random_multipliers_app
        shift_app = shift_app.to(device)
        lat_app_aug = lat_rep[2] + shift_app
    else:
        lat_app_aug = lat_rep[2]

    lat_rep_aug = [lat_geo_aug, lat_exp_aug, lat_app_aug]

    # --- Camera Parameters Augmentation ---
    camera_distance_factor = torch.randn(1).item() * 0.05 + 0.2 # originally: random value [0.2, 0.35]
    angle_rho = float(torch.randint(-50, 90, (1,)).item()) # random int [-45, 90]
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
    # --- Latent Representation Augmentation ---
    # Generate random values from a normal distribution with standard deviation alpha
    if 'geo' in opt_vars:
        random_multipliers_geo = torch.randn(geo_std.shape) * hparams["alpha"]
        shift_geo = geo_std.cpu() * random_multipliers_geo
        shift_geo = shift_geo.to(device)
        lat_geo_aug = lat_rep[0] + shift_geo
        lat_geo_aug = lat_rep[0]
    else:
        lat_geo_aug = lat_rep[0]
    
    if 'exp' in opt_vars:
        random_multipliers_exp = torch.randn(exp_std.shape) * hparams["alpha"]
        shift_exp = exp_std.cpu() * random_multipliers_exp
        shift_exp = shift_exp.to(device)
        lat_exp_aug = lat_rep[1] + shift_exp
    else:
        lat_exp_aug = lat_rep[1]
        
    if 'app' in opt_vars:
        random_multipliers_app = torch.randn(app_std.shape) * hparams["alpha"]
        shift_app = app_std.cpu() * random_multipliers_app
        shift_app = shift_app.to(device)
        lat_app_aug = lat_rep[2] + shift_app
    else:
        lat_app_aug = lat_rep[2]

    lat_rep_aug = [lat_geo_aug, lat_exp_aug, lat_app_aug]

    # --- Camera Parameters Augmentation ---
    camera_distance_factor = torch.randn(1).item() * 0.07 + 0.21 # originally: random value [0.2, 0.25]
    angle_rho = float(torch.randint(-50, 90, (1,)).item()) # random int [-45, 90]
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
    
    
def batch_forward_img_mean(lat_rep_orig, prompt, hparams):
    image_embeddings = []
    for i in range(hparams['batch_size']):
        if torch.rand(1) <= hparams['color_prob']:
            # --- Random Augmentation ---
            lat_rep, camera_params, phong_params, light_params = get_augmented_params_color(lat_rep_orig, hparams)
            # --- Forward Pass ---
            image_embedding, _ = get_image_clip_embedding(lat_rep, camera_params, phong_params, light_params, color=True)
        else:
            # --- Random Augmentation ---
            lat_rep, camera_params, phong_params, light_params = get_augmented_params_no_color(lat_rep_orig, hparams)
            # --- Forward Pass ---
            image_embedding, _ = get_image_clip_embedding(lat_rep, camera_params, phong_params, light_params, color=False)
        
        image_embeddings.append(image_embedding)
        
    mean_embedding = torch.concat(image_embeddings, axis=0).mean(axis=0, keepdim=True)
    
    # --- Text Embedding ---
    text_embedded_normalized = get_text_clip_embedding(prompt)
    
    # --- Delta CLIP Score ---
    delta_CLIP_score = clip_score(mean_embedding, text_embedded_normalized)
    
    # --- Log Prob Score ---
    prob_geo, prob_exp, prob_app = log_prop_score(lat_rep)
    
    return delta_CLIP_score, prob_geo, prob_exp, prob_app
        
        

def batch_forward(lat_rep_orig, prompt, hparams):
    all_delta_CLIP_scores = []
    all_log_probs_geo = []
    all_log_probs_exp = []
    all_log_probs_app = []
    
    for i in range(hparams['batch_size']):

        if torch.rand(1) <= hparams['color_prob']:
            # --- Random Augmentation ---
            lat_rep, camera_params, phong_params, light_params = get_augmented_params_color(lat_rep_orig, hparams)
            # --- Forward Pass ---
            delta_CLIP_score, log_prob_geo, log_prob_exp, log_prop_app, _ = forward(lat_rep, prompt, camera_params, phong_params, light_params, color=True)
        else:
            # --- Random Augmentation ---
            lat_rep, camera_params, phong_params, light_params = get_augmented_params_no_color(lat_rep_orig, hparams)
            # --- Forward Pass ---
            delta_CLIP_score, log_prob_geo, log_prob_exp, log_prop_app, _ = forward(lat_rep, prompt, camera_params, phong_params, light_params, color=False)

        
        # --- Compute Scores ---
        all_delta_CLIP_scores.append(delta_CLIP_score)
        all_log_probs_geo.append(log_prob_geo)
        all_log_probs_exp.append(log_prob_exp)
        all_log_probs_app.append(log_prop_app)
    
    all_delta_CLIP_scores_tensor = torch.stack(all_delta_CLIP_scores)
    all_log_probs_geo_tensor = torch.stack(all_log_probs_geo)
    all_log_probs_exp_tensor = torch.stack(all_log_probs_exp)
    all_log_probs_app_tensor = torch.stack(all_log_probs_app)
    batch_delta_CLIP_score = torch.mean(all_delta_CLIP_scores_tensor)
    batch_log_prob_geo = torch.mean(all_log_probs_geo_tensor)
    batch_log_prob_exp = torch.mean(all_log_probs_exp_tensor)
    batch_log_prob_app = torch.mean(all_log_probs_app_tensor)

    return batch_delta_CLIP_score, batch_log_prob_geo, batch_log_prob_exp, batch_log_prob_app

# Optimal Params for rendering with color
def get_optimal_params_color(hparams):
    camera_params = {
            "camera_distance": 0.21 * 2.57,
            "camera_angle_rho": 15., #45.,
            "camera_angle_theta": 0.,
            "focal_length": 2.57,
            "max_ray_length": 3,
            # Image
            "resolution_y": hparams['resolution'],
            "resolution_x": hparams['resolution']
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
        "resolution_y": hparams["resolution"],
        "resolution_x": hparams["resolution"]
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
            lat_geo = torch.randn_like(geo_std) * geo_std * 0.85 + geo_mean
        else:
            lat_geo = geo_mean
        if 'exp' in _vars:
            lat_exp = torch.randn_like(exp_std) * exp_std * 0.85 + exp_mean
        else:
            lat_exp = exp_mean
        if 'app' in _vars:
            lat_app = torch.randn_like(app_std) * app_std * 0.85 + app_mean
        else:
            lat_app = app_mean
            
        clip_score = forward([lat_geo, lat_exp, lat_app], prompt, *get_optimal_params_color(hparams), True)[0]
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
    lat_mean = [geo_mean, exp_mean, app_mean]

    # --- Get Mean Image (required for CLIP and DINO validation) ---
    camera_params_opti, phong_params_opti, light_params_opti = get_optimal_params(hparams)
    camera_params_opti_c, phong_params_opti_c, light_params_opti_c = get_optimal_params_color(hparams)
    if (CLIP_gt != None) or (DINO_gt != None):
        with torch.no_grad():
            mean_image = render(sdf, lat_mean, camera_params_opti, phong_params_opti, light_params_opti, color=False)


    opt_params = list(filter(lambda x: x is not None,[
        lat_geo if 'geo' in opt_vars else None,
        lat_exp if 'exp' in opt_vars else None,
        lat_app if 'app' in opt_vars else None,
    ]))
    
    optimizer = Adam(params=opt_params,
                    lr=hparams['optimizer_lr'],
                    betas=(hparams['optimizer_beta1'], 0.999),
                    weight_decay=0,
                    maximize=True)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
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
    optimizer.zero_grad()
    
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
    
    #prof.start()
    for iteration in tqdm(range(hparams['n_iterations'])):
    #prof.step()
        
        lat_geo_old = lat_rep[0].detach().cpu()
        lat_exp_old = lat_rep[1].detach().cpu()

        if hparams["mean_mode"] == "mean_embed":
            batch_CLIP_score, batch_log_prob_geo_score, batch_log_prob_exp_score, batch_log_prob_app_score = batch_forward_img_mean(lat_rep, prompt, hparams)
        elif hparams["mean_mode"] == "mean_clip":
            batch_CLIP_score, batch_log_prob_geo_score, batch_log_prob_exp_score, batch_log_prob_app_score = batch_forward(lat_rep, prompt, hparams)
        else:
            raise ValueError(f"unknown mode: {mode}")

        batch_score = loss_fn(torch.clone(batch_CLIP_score), batch_log_prob_geo_score, batch_log_prob_exp_score, batch_log_prob_app_score, hparams)

        if batch_score > best_score:
            best_score = batch_score.detach().cpu()
            best_latent_geo = torch.clone(lat_rep[0]).cpu()
            best_latent_exp = torch.clone(lat_rep[1]).cpu()
            best_latent_app = torch.clone(lat_rep[2]).cpu()
            
        if batch_CLIP_score > best_clip_score:
            best_clip_score = batch_CLIP_score.detach().cpu()
            #best_clip_latent = torch.clone(lat_rep).cpu()

        optimizer.zero_grad() 
        batch_score.backward()
        sys.stdout.flush()

        # --- Validation with CLIP / DINO Delta Score ---
        with torch.no_grad():
            CLIP_score_no_col, _, _, _, image_no_col = forward(lat_rep, prompt, camera_params_opti, phong_params_opti, light_params_opti, color=False)
            CLIP_score_col, _, _, _, image_col = forward(lat_rep, prompt, camera_params_opti_c, phong_params_opti_c, light_params_opti_c, color=True)
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
        
        #clip_grad_norm_([lat_rep], hparams['grad_norm'])

        writer.add_scalar('Batch Score', batch_score, iteration)
        writer.add_scalar('Batch CLIP Score', batch_CLIP_score, iteration)
        writer.add_scalar('Batch Log Prob Geometry Score', batch_log_prob_geo_score, iteration)
        writer.add_scalar('Batch Log Prob Expression Score', batch_log_prob_exp_score, iteration)
        writer.add_scalar('Batch Log Prob Appearance Score', batch_log_prob_app_score, iteration)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], iteration)
        if 'geo' in opt_vars:
            lat_geo.grad = lat_geo.grad.nan_to_num(0.) # TODO: is this still needed?
            gradient_lat_geo = lat_geo.grad
            writer.add_scalar('Gradient norm of Score w.r.t. Geometry Latent', gradient_lat_geo.norm(), iteration)
            
            # Difference between lat_rep and previous lat_rep
            lat_geo_new = lat_rep[0].detach().cpu()
            lat_geo_diff = torch.abs(lat_geo_new - lat_geo_old) / geo_std.cpu()
            mean_diff = torch.mean(lat_geo_diff.abs())
            writer.add_scalar('Mean percentual diff geo wrt std dev', mean_diff, iteration)

        if 'exp' in opt_vars:
            gradient_lat_exp = lat_exp.grad
            writer.add_scalar('Gradient norm of Score w.r.t. Expression Latent', gradient_lat_exp.norm(), iteration)

        if 'app' in opt_vars:
            gradient_lat_app = lat_app.grad
            writer.add_scalar('Gradient norm of Score w.r.t. Appearance Latent', gradient_lat_app.norm(), iteration)

        optimizer.step()
        lr_scheduler.step(batch_score)

        if 'geo' in opt_vars:
            # Difference between lat_rep and previous lat_rep
            lat_geo_new = lat_rep[0].detach().cpu()
            lat_geo_diff = torch.abs(lat_geo_new - lat_geo_old) / geo_std.cpu()
            mean_diff = torch.mean(lat_geo_diff.abs())
            writer.add_scalar('Mean percentual diff geo wrt std dev', mean_diff, iteration)

            # Angle between lat_rep and previous lat_rep
            lat_geo_new = lat_geo_new.squeeze(0).squeeze(0)
            lat_geo_old = lat_geo_old.squeeze(0).squeeze(0)
            normalized_lat_geo_new = lat_geo_new / lat_geo_new.norm(dim=-1, keepdim=True)
            normalized_lat_geo_old = lat_geo_old / lat_geo_old.norm(dim=-1, keepdim=True)
            cos_theta = torch.dot(normalized_lat_geo_new, normalized_lat_geo_old)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angle_rad = torch.acos(cos_theta)
            angle_deg = torch.rad2deg(angle_rad)
            writer.add_scalar('Angle between lat reps geo', angle_deg, iteration)

        if 'exp' in opt_vars:
            # Difference between lat_rep and previous lat_rep
            lat_exp_new = lat_rep[1].detach().cpu()
            lat_exp_diff = torch.abs(lat_exp_new - lat_exp_old) / exp_std.cpu()
            mean_diff = torch.mean(lat_exp_diff.abs())
            writer.add_scalar('Mean percentual diff exp wrt std dev', mean_diff, iteration)

            # Angle between lat_rep and previous lat_rep
            lat_exp_new = lat_exp_new.squeeze(0).squeeze(0)
            lat_exp_old = lat_exp_old.squeeze(0).squeeze(0)
            normalized_lat_exp_new = lat_exp_new / lat_exp_new.norm(dim=-1, keepdim=True)
            normalized_lat_exp_old = lat_exp_old / lat_exp_old.norm(dim=-1, keepdim=True)
            cos_theta = torch.dot(normalized_lat_exp_new, normalized_lat_exp_old)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            angle_rad = torch.acos(cos_theta)
            angle_deg = torch.rad2deg(angle_rad)
            writer.add_scalar('Angle between lat reps exp', angle_deg, iteration)


        optimizer.zero_grad()          

    #prof.stop()

    writer.add_hparams(hparams, {
        'Best score': best_score,
        'Best CLIP score': best_clip_score
        })
    writer.close()

    return [best_latent_geo.detach(), best_latent_exp.detach(), best_latent_app.detach()], best_score
    
    
    