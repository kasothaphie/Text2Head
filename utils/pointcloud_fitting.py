import torch
import numpy as np
import os.path as osp
import json
import yaml
from torch.optim import Adam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.checkpoint import checkpoint
from pytorch3d.loss import chamfer_distance
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils.render import acc_sphere_trace, rotate
from nphm_tum import env_paths as mono_env_paths
from nphm_tum.models.neural3dmm import construct_n3dmm, load_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

class PointCloudGenerator(torch.nn.Module):
    def __init__(self, model, camera_params):
        super(PointCloudGenerator, self).__init__()
        self.model = model
        self.camera_params = camera_params

    def forward(self, lat_rep):

        def sdf(positions, chunk_size=10000):
        
            def get_sdf(nphm_input, lat_rep_in):
                #distance, color = model(nphm_input.to(device), *lat_rep_in)
                distance, color = checkpoint(self.model, *[nphm_input.to(device), *lat_rep_in])
                distance = distance.to("cpu")
                color = color.to("cpu")
                return distance.squeeze(), color.squeeze()
                
            nphm_input = torch.reshape(positions, (1, -1, 3))
            
            if nphm_input.shape[1] > chunk_size:
                chunked = torch.chunk(nphm_input, chunks=nphm_input.shape[1] // chunk_size, dim=1)
                distances, colors = zip(*[get_sdf(chunk, lat_rep) for chunk in chunked])
                return torch.cat(distances, dim=0), torch.cat(colors, dim=0)
            else:
                #distance = model(nphm_input.to(device), lat_rep_in.to(device).requires_grad_(True), None)[0].to("cpu")
                distance, color = get_sdf(nphm_input, lat_rep)
                return distance, color

        point_clouds = []   
        for angles in [-30., 30.]:
        
            camera_params = self.camera_params
            camera_params['camera_angle_rho'] = angles

            pu = camera_params["resolution_x"]
            pv = camera_params["resolution_y"]

            # Normalize the xy value of the current pixel [-0.5, 0.5]
            u_norms = ((torch.arange(pu) + 0.5) / pu - 0.5) * pu / pv
            v_norms = 0.5 - (torch.arange(pv) + 0.5) / pv

            # Calculate the ray directions for all pixels
            directions_unn = torch.cat(
                torch.meshgrid(u_norms, v_norms, torch.tensor(-camera_params["focal_length"]), indexing='ij'), dim=-1)
            directions_unn = directions_unn.reshape(
                (pu * pv, 3))  # [pu, pv, 3] --> [pu*pv, 3] (u1, v1, f)(u1, v2, f)...(u2, v1, f)...

            camera = torch.tensor([0, 0, 1])

            # rotate direction vectors and camera position
            directions, camera_position = rotate(camera_params, camera, directions_unn)

            # start close to head model to get useful sdf scores
            first_step_length = camera_params['focal_length'] + camera_params['camera_distance'] - 1
            N = directions.shape[0]
            starting_positions = camera_position.unsqueeze(dim=0).repeat(N, 1) + first_step_length * directions

            hits, hit_mask, _ = acc_sphere_trace(sdf, starting_positions, directions, camera_params['max_ray_length'], scale=np.sqrt(2.), eps=0.001)

            gt_points = hits[hit_mask]
            point_clouds.append(gt_points)

        return torch.cat(point_clouds)

opt_vars = ['geo'] # add 'exp' and/or 'app' (['geo', 'exp', 'app'])
grad_vars = ['geo']

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

geo_mean = latent_codes.codebook['geo'].embedding.weight.mean(dim=0).detach()
geo_std = latent_codes.codebook['geo'].embedding.weight.std(dim=0).detach()
exp_mean = latent_codes.codebook['exp'].embedding.weight.mean(dim=0).detach()
exp_std = latent_codes.codebook['exp'].embedding.weight.std(dim=0).detach()
app_mean = latent_codes.codebook['app'].embedding.weight.mean(dim=0).detach()
app_std = latent_codes.codebook['app'].embedding.weight.std(dim=0).detach()

def get_latent_from_points(lat_rep_gt, hparams, camera_params):

    model = PointCloudGenerator(sdf, camera_params)
    model.to(device)

    lat_rep_gt = [tensor.to(device) for tensor in lat_rep_gt]
    gt_point_cloud = model(lat_rep_gt)

    x = gt_point_cloud[:, 0]
    y = gt_point_cloud[:, 1]
    z = gt_point_cloud[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='.')
    ax.view_init(elev=0, azim=0, roll=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    plt.show()

    lat_geo = geo_mean.clone().detach().to(device).requires_grad_(True) 

    optimizer = Adam(params=[lat_geo],
                    lr=hparams['optimizer_lr'],
                    betas=(0.9, 0.999),
                    weight_decay=0,
                    maximize=False) 
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=hparams['lr_scheduler_factor'],
        patience=hparams['lr_scheduler_patience'],
        min_lr=hparams['lr_scheduler_min_lr']
    )
    
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    writer = SummaryWriter(log_dir=f'../runs/pointcloud/train-time:{now}')

    for iteration in range(hparams['n_iterations']):    

        lat_rep = [lat_geo, exp_mean.to(device), app_mean.to(device)]
        lat_geo_old = lat_rep[0].detach().cpu()
        point_cloud = model(lat_rep)
        print(point_cloud.shape)

        loss, _ = chamfer_distance(point_cloud.unsqueeze(0), gt_point_cloud.unsqueeze(0))
        writer.add_scalar('Loss * 1e4', loss*1e4, iteration)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], iteration)

        optimizer.zero_grad()
        loss.backward()

        gradient_lat_geo = lat_geo.grad
        writer.add_scalar('Gradient norm of Score w.r.t. Geometry Latent', gradient_lat_geo.norm(), iteration)
        
        optimizer.step()
        lr_scheduler.step(loss)

        # Difference between lat_rep and previous lat_rep
        lat_geo_diff = torch.abs(lat_geo.detach().cpu() - lat_geo_old) / geo_std.cpu()
        mean_diff = torch.mean(lat_geo_diff.abs())
        writer.add_scalar('Mean percentual diff wrt std dev', mean_diff, iteration)

        # Print the loss for monitoring
        print(f'Epoch [{iteration+1}], Loss: {loss.item()}')
    
    x = point_cloud[:, 0].detach()
    y = point_cloud[:, 1].detach()
    z = point_cloud[:, 2].detach()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='.')
    ax.view_init(elev=0, azim=0, roll=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    plt.show()


    return lat_rep
