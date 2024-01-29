import yaml
import torch
import sys
from torch.optim import Adam
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
import numpy as np
import clip
#import open_clip
import os
import os.path as osp
from NPHM.models.EnsembledDeepSDF import FastEnsembleDeepSDFMirrored
from NPHM import env_paths
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import uuid
import random

from torch.profiler import profile, record_function, ProfilerActivity

from utils.render import render
from utils.similarity import CLIP_similarity, DINO_similarity
from utils.EMA import EMA


device = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_model, CLIP_preprocess = clip.load("ViT-B/32", device="cpu")
#SigLIPmodel = open_clip.create_model("ViT-B-16-SigLIP", pretrained='webli', device="cpu")
#tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')

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

lat_mean = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_mean.npy'))
lat_std = torch.from_numpy(np.load(env_paths.ASSETS + 'nphm_lat_std.npy'))

def loss_fn(clip_score, prob_score, hparams):
    return (clip_score + hparams["lambda"] * prob_score) / (1 + hparams["lambda"])

def get_image_clip_embedding(lat_rep, camera_params, phong_params, light_params):
    lat_rep = lat_rep.to(device)
    image = render(decoder_shape, lat_rep, camera_params, phong_params, light_params)
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
    cov = lat_std * torch.eye(lat_mean.shape[0])
    delta = lat_rep.cpu() - lat_mean.cpu()
    prob_score = -delta.T @ torch.inverse(cov.cpu()) @ delta
    
    return prob_score

def forward(lat_rep, prompt, camera_params, phong_params, light_params):
    # --- Render Image from current Lat Rep + Embedd ---
    image_embedding, image = get_image_clip_embedding(lat_rep, camera_params, phong_params, light_params)
    ##
    delta_images_normalized = image_embedding
    '''
    # --- Render Image from Lat Mean WITH SAME PARAMS AS LAT REP + Embedd ---
    #mean_image_embedding, _ = get_image_clip_embedding(lat_mean, camera_params, phong_params, light_params)
    
    # --- Difference between both embeddings ---
    #delta_images = image_embedding - mean_image_embedding
    delta_images = image_embedding
    if delta_images.norm() >= 1e-9:
        delta_images_normalized = delta_images / delta_images.norm(dim=-1, keepdim=True)
    else:
        delta_images_normalized = delta_images
    '''
    # --- Text Embedding ---
    text_embedded_normalized = get_text_clip_embedding(prompt)
    
    # --- Delta CLIP Score ---
    delta_CLIP_score = clip_score(delta_images_normalized, text_embedded_normalized)
    
    # --- Log Prob Score ---
    prob_score = log_prop_score(lat_rep)

    return delta_CLIP_score, prob_score, torch.clone(image)


def energy_level(lat_rep_1, lat_rep_2, prompt, hparams, steps=100):
    with torch.no_grad():
        lat_rep_1 = lat_rep_1.cpu()
        lat_rep_2 = lat_rep_2.cpu()
        lat_reps = [torch.lerp(lat_rep_1, lat_rep_2, i) for i in torch.linspace(0., 1., steps)]
        forwards = [batch_forward(lat_rep.to(device), prompt, hparams) for lat_rep in lat_reps]
        energy = [loss_fn(f[0], f[1], hparams) for f in forwards]
    
    return energy, forwards

def get_augmented_params(lat_rep, hparams):
    # --- Latent Representation Augmentation ---
    # Generate random values from a normal distribution with standard deviation a
    random_multipliers = torch.randn(lat_std.shape) * hparams["alpha"]
    shift = lat_std * random_multipliers
    shift = shift.to(device)
    lat_rep_aug = lat_rep + shift

    # --- Camera Parameters Augmentation ---
    camera_distance_factor = torch.rand(1).item() * 0.05 + 0.2 #random value [0.2, 0.25]
    focal_length = torch.rand(1).item() * 0.4 + 2.5 #random value [2.5, 3.3]
    angle = float(torch.randint(-50, 90, (1,)).item()) # random int [-45, 90]
    camera_params_aug = {
        "camera_distance": camera_distance_factor * focal_length,
        "camera_angle": angle,
        "focal_length": focal_length,
        "max_ray_length": 3,
        # Image
        "resolution_y": hparams["resolution"],
        "resolution_x": hparams["resolution"]
    }

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
    if angle >= 0:
        light_dir_0 = torch.rand(1).item() * 0.05 - 0.6 #random value [-0.6, -0.55]
        light_pos_0 = 1.19
    else:
        light_dir_0 = torch.rand(1).item() * 0.05 + 0.55 #random value [0.55, 0.6]
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
    
def batch_forward(lat_rep_orig, prompt, hparams):
    all_delta_CLIP_scores = []
    all_log_probs = []
    
    # --- Params for Determined Augmentation (not necessary for default = Random Augmentation) ---
    angles = [45., -30., 90., 
              -50., 35., -20., 
              55., -40., 10.]
    shift = hparams['alpha'] * lat_std
    shift = shift.to(device)
    lat_reps = [lat_rep_orig, (lat_rep_orig + shift), (lat_rep_orig - shift),
               (lat_rep_orig + 0.5 * shift), (lat_rep_orig - 0.5 * shift), (lat_rep_orig + 0.3 *shift), (lat_rep_orig - 0.3 * shift)]
    
    for i in range(hparams['batch_size']):
        # --- Random Augmentation ---
        lat_rep, camera_params, phong_params, light_params = get_augmented_params(lat_rep_orig, hparams)
        
        # --- Determined Augmentation ----
        #lat_rep = lat_reps[i]
        #lat_rep = lat_rep_orig
        #lat_rep, _, _, _ = get_augmented_params(lat_rep_orig, hparams)
        #camera_params, phong_params, light_params = get_optimal_params(hparams)
        #_, camera_params, phong_params, light_params = get_augmented_params(lat_rep_orig, hparams)
        #camera_params['camera_angle'] = angles[i]
        #if angles[i] >= 0:
            #light_params['light_dir_1'] = torch.tensor([-0.6, -0.4, -0.67])
            #light_params['light_pos_p'] = torch.tensor([1.19, -1.27, 2.24])
        #else:
            #light_params['light_dir_1'] = torch.tensor([0.6, -0.4, -0.67])
            #light_params['light_pos_p'] = torch.tensor([-1.19, -1.27, 2.24])
        
        # --- Compute Scores ---
        delta_CLIP_score, log_prob, _ = forward(lat_rep, prompt, camera_params, phong_params, light_params)
        all_delta_CLIP_scores.append(delta_CLIP_score)
        all_log_probs.append(log_prob)
    
    all_delta_CLIP_scores_tensor = torch.stack(all_delta_CLIP_scores)
    all_log_probs_tensor = torch.stack(all_log_probs)
    batch_delta_CLIP_score = torch.mean(all_delta_CLIP_scores_tensor)
    batch_log_prob = torch.mean(all_log_probs_tensor)

    return batch_delta_CLIP_score, batch_log_prob    

def get_optimal_params(hparams):
    camera_params = {
        "camera_distance": 0.21 * 2.57,
        "camera_angle": 45.,
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


def get_latent_from_text(prompt, hparams, init_lat=None, CLIP_gt=None, DINO_gt=None):
    global lat_std, lat_mean
    
    if init_lat is None:
        lat_rep = (torch.randn_like(lat_std) * lat_std * 0.85 + lat_mean).detach()
    else:
        lat_rep = init_lat
        
    lat_rep = lat_rep.to(device).requires_grad_(True)
    lat_mean = lat_mean.to(device)

    # --- Get Mean Image (required for CLIP and DINO validation) ---
    camera_params_opti, phong_params_opti, light_params_opti = get_optimal_params(hparams)
    with torch.no_grad():
        mean_image = render(decoder_shape, lat_mean, camera_params_opti, phong_params_opti, light_params_opti)

    if hparams['optimizer'] == 'Adam':
        optimizer = Adam(params=[lat_rep],
                     lr=hparams['optimizer_lr'],
                     betas=(0.9, 0.999),
                     weight_decay=0,
                     maximize=True)
    elif hparams['optimizer'] == 'EMA':
        optimizer = EMA(params=[lat_rep],
                     lr=hparams['optimizer_lr'],
                     beta=hparams['EMA_beta'],
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
    writer = SummaryWriter(log_dir=f'../runs/a/train-time:{now}')

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

    #prof.start()
    for iteration in tqdm(range(hparams['n_iterations'])):
    #prof.step()
        lat_rep_old = lat_rep.detach().cpu()
        batch_delta_CLIP_score, batch_log_prob_score = batch_forward(lat_rep, prompt, hparams)
        sys.stdout.flush()
        batch_score = loss_fn(batch_delta_CLIP_score, batch_log_prob_score, hparams)

        if batch_score > best_score:
            best_score = batch_score.detach().cpu()
            best_latent = torch.clone(lat_rep).cpu()
            
        if batch_delta_CLIP_score > best_clip_score:
            best_clip_score = batch_delta_CLIP_score.detach().cpu()
            best_clip_latent = torch.clone(lat_rep).cpu()

        batch_score.backward()
        
        # Manually modify the gradient to set NaN values to zero
        lat_rep.grad = lat_rep.grad.nan_to_num(0.)

        # --- Validation with CLIP / DINO Delta Score ---
        if (CLIP_gt != None) or (DINO_gt != None):
            image = render(decoder_shape, lat_rep, camera_params_opti, phong_params_opti, light_params_opti)
            writer.add_image(f'rendered image of {prompt}', image.detach().numpy(), iteration, dataformats='HWC')
        
        if CLIP_gt != None:
            CLIP_gt_similarity, CLIP_delta_sim = CLIP_similarity(image, CLIP_gt, mean_image)
            writer.add_scalar('CLIP similarity to ground truth image', CLIP_gt_similarity, iteration)
            writer.add_scalar('CLIP delta similarity', CLIP_delta_sim, iteration)

        if DINO_gt != None:
            DINO_gt_similarity, DINO_delta_sim = DINO_similarity(image, DINO_gt, mean_image)
            writer.add_scalar('DINO similarity to ground truth image', DINO_gt_similarity, iteration)
            writer.add_scalar('DINO delta similarity', DINO_delta_sim, iteration)
        
        #clip_grad_norm_([lat_rep], hparams['grad_norm'])
        gradient_lat_rep = lat_rep.grad

        writer.add_scalar('Batch Score', batch_score, iteration)
        writer.add_scalar('Batch CLIP Score', batch_delta_CLIP_score, iteration)
        writer.add_scalar('Batch Log Prob Score', batch_log_prob_score, iteration)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], iteration)
        writer.add_scalar('Gradient norm of Score w.r.t. Latent', gradient_lat_rep.norm(), iteration)

        optimizer.step()
        lr_scheduler.step(batch_score)

        # Difference between lat_rep and previous lat_rep
        lat_rep_diff = torch.abs(lat_rep.detach().cpu() - lat_rep_old) / lat_std.cpu()
        mean_diff = torch.mean(lat_rep_diff.abs())
        writer.add_scalar('Mean percentual diff wrt std dev', mean_diff, iteration)

        optimizer.zero_grad()          

    #prof.stop()
    lat_rep_end = lat_rep.detach()

    writer.add_hparams(hparams, {
        'Best score': best_score,
        'Best CLIP score': best_clip_score
        })
    writer.close()

    return best_latent.detach(), best_clip_latent.detach(), lat_rep_end, best_score
    
    
    