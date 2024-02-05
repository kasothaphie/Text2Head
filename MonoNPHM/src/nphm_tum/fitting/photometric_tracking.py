import os

import cv2
import trimesh
import pyvista
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import mediapy
#import wandb

from pytorch3d.transforms import so3_exp_map, so3_log_map

import matplotlib.pyplot as plt
from torchvision.transforms.functional import gaussian_blur

from nphm_tum import env_paths
from nphm_tum.fitting.renderer import RendererMonoNPHM  # volume rendering (NeuS-stlye)
from nphm_tum.fitting.data_loading import prepare_data
from nphm_tum.models.reconstruction import get_logits, get_vertex_color
from nphm_tum.utils.reconstruction import create_grid_points_from_bounds, mesh_from_logits
from nphm_tum.models.iterative_root_finding import search
from nphm_tum.models.diff_operators import jac
from nphm_tum.utils.render_utils import project_points_torch

from dreifus.matrix import Pose


# corresponding indices between MonoNPHM's anchor points and FLAME/iBUG68 landmarks
# The first column holds indices into the anchors, the second into FLAME landmarks,
# e.g. the chin is the 60th anchor and 8th landmark
ANCHOR_iBUG68_pairs_65 = np.array([
    [0, 0],  # left upmost jaw
    [1, 16],  # right upmost jaw
    [38, 2],  # jaw
    [39, 14],  # jaw
    [2, 4],  # jaw
    [3, 12],  # jaw
    [4, 6],  # jaw
    [5, 10],  # jaw
    [60, 8],  # chin
    [10, 31],  # nose
    [11, 35],  # nose
    [62, 30],  # nose tip
    [61, 27],  # nose top
    [6, 17],  # l eyebrow outer
    [7, 26],  # r eyebrow outer
    [12, 36],  # l eye outer,
    [13, 45],  # r eye outer,
    [14, 39],  # l eye inner,
    [15, 42],  # r eye inner,
    [16, 48],  # l mouth corner
    [17, 54],  # r mouth corner
    [18, 50],  # l mouth top
    [19, 52],  # r mouth top
    [20, 58],  # l mouth bottom
    [21, 56],  # r mouth bottom,
    [44, 49],  # mouth
    [45, 53],  # mouth
    [46, 59],  # mouth
    [47, 55],  # mouth
    [48, 38],  # eye upper l
    [49, 43],  # eye upper r
    [50, 41],  # eye lower l
    [51, 46],  # eye lower r
    [52, 21],  # eyebrow inner l
    [53, 22],  # eyebrow inner r
])


class ImageSpaceLoss(nn.Module):
    '''
    Implements loss functions in image space.
    It has the following components:
     (1) RGB loss
     (2) Siolhouette Loss
     (3) Normal Loss
    '''
    def __init__(self,
                 rgb_weight,
                 silhouette_weight,
                 lambdas_reg,
                 reg_weight_expr,
                 disable_mouth = True,
                 normal_weight : float = 0.0):
        super().__init__()

        self.rgb_weight = rgb_weight

        self.normal_weight = normal_weight

        self.silhouette_weight = silhouette_weight
        self.lambdas_reg = lambdas_reg
        self.reg_weight_expr = reg_weight_expr
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.disable_mouth = disable_mouth


    # compute photometric loss
    def get_rgb_loss(self, rgb_values, rgb_gt, object_mask, hair_region, mouth_interior):

        mouth_interior = ~mouth_interior.squeeze()[object_mask]
        rgb_loss = torch.abs(rgb_values[object_mask] - rgb_gt[0, object_mask, :]) / float(object_mask.shape[0])

        # decide how much to weight the color loss inside the mouth region
        # this can be important if there are dark shadows when the mouth is open
        if self.disable_mouth:
            rgb_loss[mouth_interior] /= 25
        else:
            rgb_loss[mouth_interior] /= 2

        # weight the RGB loss loess in the hair region
        rgb_loss[hair_region[object_mask]] /= 10


        return rgb_loss.sum()

    def get_normal_loss(self, normals_values, normals_gt, object_mask):

        valid_normals = torch.abs(normals_gt.norm(dim=-1) - 1) < 0.02
        object_mask = object_mask & valid_normals.squeeze()
        normal_loss = torch.abs(normals_values[object_mask] - normals_gt[0, object_mask, :]) / float(object_mask.shape[0])

        normal_loss = normal_loss.sum()
        return normal_loss


    # compute silhouette loss
    def get_jaccard_distance_loss(self, y_pred, y_true, smooth=5):
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

        The jaccard distance loss is usefull for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
        gradient.

        Ref: https://en.wikipedia.org/wiki/Jaccard_index

        @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
        @author: wassname
        """
        y_pred = y_pred.squeeze().clamp(0, 1)

        intersection = torch.sum(torch.abs(y_true * y_pred))
        sum_ = torch.sum(torch.abs(y_true) + torch.abs(y_pred))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth


    def forward(self, model_outputs, ground_truth, wandb_dict=None):

        rgb_gt = ground_truth['rgb']
        normals_gt = ground_truth['normals']
        network_object_mask = model_outputs['network_object_mask']
        facer_mask = model_outputs['object_mask']

        rgb_loss_mask = ~((facer_mask == 3) | (facer_mask == 0))
        foreground_mask = rgb_loss_mask.clone()
        hair_region = facer_mask == 14

        # TODO the rgb loss in the hair region that is close the face should be valued stronger
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, rgb_loss_mask, hair_region=hair_region,
                                     mouth_interior=ground_truth['mm'] == 0)
        normal_loss = self.get_normal_loss(model_outputs['nphm_space_normals'], normals_gt, rgb_loss_mask)

        # compute silhouette loss
        if model_outputs['weights_sum'] is not None:
            # compute the silhouette loss separately for the hair and all the remainder
            # the hair mask loss is weighted less, intuition is that we care mour about properly fitted cheecks etc.
            # but we don't need to fit each hair strand perfectly
            n_hair = hair_region.sum()
            nn_hair = (~hair_region).sum() > 0
            if n_hair > 0:
                mask_loss_hair_region = self.get_jaccard_distance_loss(model_outputs['weights_sum'][hair_region],
                                                                       foreground_mask[hair_region])
            else:
                mask_loss_hair_region = 0
            if nn_hair > 0:
                mask_loss = self.get_jaccard_distance_loss(model_outputs['weights_sum'][~hair_region], foreground_mask[~hair_region])
            else:
                mask_loss = 0
            mask_loss = (nn_hair * mask_loss + n_hair * mask_loss_hair_region / 20) / (nn_hair + n_hair)
        else:
            mask_loss = self.get_mask_lossOG(model_outputs['sdf_output'], network_object_mask, foreground_mask)

        # TODO why is regularization loss aggregated in here? move out side or into its own function
        reg_loss = 0
        for k in model_outputs['reg_loss'].keys():
            reg_loss += self.lambdas_reg[k] * model_outputs['reg_loss'][k]
            wandb_dict[k] = model_outputs['reg_loss'][k].item()
            wandb_dict[k + '_scaled'] = model_outputs['reg_loss'][k].item() * self.lambdas_reg[k]
        reg_loss_expr = model_outputs['reg_loss_expr']

        # TODO move outside
        loss = self.rgb_weight * rgb_loss + \
               self.silhouette_weight * mask_loss + \
               reg_loss + \
               self.reg_weight_expr * reg_loss_expr + \
               self.normal_weight * normal_loss


        # log each loss separately
        # once scaled by its weight and once log the raw values
        if wandb_dict is not None:
            wandb_dict['loss_rgb'] = rgb_loss.item()
            wandb_dict['loss_normals'] = normal_loss.item()
            wandb_dict['loss_mask'] = mask_loss.item()
            wandb_dict['reg_expr'] = reg_loss_expr.item()
            wandb_dict['loss_rgb_scaled'] = rgb_loss.item() * self.rgb_weight
            wandb_dict['loss_mask_scaled'] = mask_loss.item() * self.silhouette_weight
            wandb_dict['reg_expr_scaled'] = reg_loss_expr.item() * self.reg_weight_expr

        return_dict = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'normal_loss': normal_loss,
            'mask_loss': mask_loss,
            'reg_loss_expr': reg_loss_expr,
        }
        # add regularization loss for ID to return dict
        return_dict.update(model_outputs['reg_loss'])
        return return_dict


class LandmarkLoss(nn.Module):
    '''
    Implements landmark loss.
    '''
    def __init__(self, n_expr, pose_codebook, scale_param):
        super().__init__()

        if False: # old scaling
            anchor_loss_scale = np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, #10, 10, 10, 10, 10, 10,
                 100,
                 10, 10, 10, 10, 0, 0, 10, 10, 10, 10,
                 100, 100, 100, 100, 100, 100,
                 100, 100, 100, 100,
                 100, 100, 100, 100,
                 10, 10]
            )

        # all anchors are equally weighted, but anchors on the jaw line are excluded for the loss
        anchor_loss_scale = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0,  # 10, 10, 10, 10, 10, 10,
             100,
             100, 100, 100, 100, 0, 0, 100, 100, 100, 100,
             100, 100, 100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100, 100, 100,
             100, 100]
        )
        anchor_loss_scale = anchor_loss_scale.astype(float) / 2
        self.anchor_loss_scale = torch.from_numpy(anchor_loss_scale).cuda()
        # chache anchors in canonical and posed space for visualization later
        self.anchors_posed = [None for _ in range(n_expr)]
        self.anchors_canonical = None
        self.valid_anchors_posed = [None for _ in range(n_expr)]

        # store a reference to the pose and scale of the head
        # TODO remove this and pass it to the forward call --> that is more explicit and not so hidden
        self.pose_codebook = pose_codebook
        self.scale_param = scale_param


    def predict_canonical_anchors(self, idr):
        can_anchors = idr.implicit_network.GTA.id_model.get_anchors(idr.implicit_network.latent_code_id['geo'])
        self.anchors_canonical = can_anchors.detach()
        # small hack to adjust anchor positions to match landmarks slightly better, or to open mouth and eye more or less
        # eyes
        can_anchors[:, 18, 1] += 0.005
        can_anchors[:, 19, 1] += 0.005
        can_anchors[:, 20, 1] -= 0.01
        can_anchors[:, 21, 1] -= 0.01
        # mouth
        #can_anchors[:, 48, 1] -= 0.05
        #can_anchors[:, 49, 1] -= 0.05
        #can_anchors[:, 50, 1] += 0.05
        #can_anchors[:, 51, 1] += 0.05
        return can_anchors


    def forward(self, idr, cur_timestep, data):

        rnd_view_idx = 0

        can_anchors = self.predict_canonical_anchors(idr)


        index_anchors = torch.from_numpy(ANCHOR_iBUG68_pairs_65[:, 0]).cuda()
        index_lms = torch.from_numpy(ANCHOR_iBUG68_pairs_65[:, 1]).cuda()

        _cond = {k: idr.implicit_network.latent_code_id[k].clone() for k in
                 idr.implicit_network.latent_code_id.keys()}
        _cond['exp'] = idr.implicit_network.latent_codes_expr(
            torch.tensor([cur_timestep], device='cuda')).unsqueeze(0)

        # there is likely some bug in the implementation of iterative root finding when using multiple intialization per anchor point
        multi_corresp = False # therefore set this to false
        # find anchors in posed space
        p_corresp_posed, search_result = search(can_anchors,
                                                _cond,
                                                idr.implicit_network.GTA.ex_model,
                                                can_anchors.clone().unsqueeze(1).repeat(1, can_anchors.shape[1], 1, 1),
                                                multi_corresp=multi_corresp)

        #if multi_corresp:
        #    print('NUM valid corresps: ', search_result['valid_ids'].any(dim=-1).sum())
        #else:
        #    print('NUM valid corresps: ', search_result['valid_ids'].sum())

        p_corresp_posed = p_corresp_posed.detach()

        # handle multi correspondence
        num_inits = None
        if len(p_corresp_posed.shape) == 4:
            batch_size, num_points_root, num_inits, _ = p_corresp_posed.shape
            p_corresp_posed = p_corresp_posed.reshape(1, -1, 3)

        search_result['valid_ids'] = search_result['valid_ids'].reshape(-1)

        # since iterative root finding is performed in torch.no_grad()-block, manually add gradients needed for backprop.
        # see SNARF paper/code  for reference: https://github.com/xuchen-ethz/snarf
        out = idr.implicit_network.GTA.ex_model({'queries': p_corresp_posed, 'cond': _cond, 'anchors': can_anchors})
        preds_can = p_corresp_posed + out['offsets']
        grad_inv = jac(idr.implicit_network.GTA.ex_model, {'queries': p_corresp_posed,
                                                           'cond': _cond,  # idr.implicit_network.latent_code_id,
                                                           'anchors': can_anchors}).inverse()
        correction = preds_can - preds_can.detach()
        correction = torch.einsum("bnij,bnj->bni", -grad_inv.detach(), correction)
        # trick for implicit diff with autodiff:
        x_posed = p_corresp_posed + correction
        anchors_posed = x_posed.detach().cpu().squeeze().numpy()

        # cache results, used when optimization is done
        self.anchors_posed[cur_timestep] = anchors_posed
        self.valid_anchors_posed[cur_timestep] = search_result['valid_ids'].detach()

        # before we project into image space, apply world2cam transformation
        _pose_params = self.pose_codebook(torch.tensor([cur_timestep], device='cuda'))
        x_posed = ((x_posed - _pose_params[:, 3:6]) / self.scale_param) @ so3_exp_map(_pose_params[:, :3]).squeeze()

        if num_inits is not None:
            index_anchors = index_anchors.repeat(num_inits)


        # project 3d posed anchors into image space using camera intrinsics
        # (note: in a moncular setting the extrinsics should is an identity matrix)
        points2d = project_points_torch(x_posed.squeeze()[index_anchors, :],
                                        torch.from_numpy(data['intrinsics'][cur_timestep][rnd_view_idx]).to(
                                            x_posed.device).float(),
                                        torch.from_numpy(data['w2c'][cur_timestep][rnd_view_idx]).to(
                                            x_posed.device).float(),
                                        )

        points2d_gt = torch.from_numpy(data['landmarks_2d'][cur_timestep][rnd_view_idx]).float().to(
                x_posed.device)[index_lms,
                          :]

        if num_inits is not None and points2d_gt is not None:
            points2d_gt = points2d_gt.repeat(num_inits, 1)

        # scale from pixels to [0-1] range --> the loss needs to be invariatn w.r.t. rendering size
        points2d[:, 0] = points2d[:, 0] / data['width'][cur_timestep][rnd_view_idx]
        points2d[:, 1] = points2d[:, 1] / data['height'][cur_timestep][rnd_view_idx]
        points2d_gt[:, 0] = points2d_gt[:, 0] / data['width'][cur_timestep][rnd_view_idx]
        points2d_gt[:, 1] = (points2d_gt[:, 1] / data['height'][cur_timestep][rnd_view_idx])

        # compute the loss only for anchors where iterative root finding was actually successful
        valid_anchors = search_result['valid_ids'].squeeze()[index_anchors]
        points2d = points2d[valid_anchors, :]
        points2d_gt = points2d_gt.squeeze()[valid_anchors, :]
        # apply weighting to specific landmarks
        anchor_loss_scale = self.anchor_loss_scale.clone()
        if num_inits is not None:
            anchor_loss_scale = anchor_loss_scale.repeat(num_inits)
        anchor_loss_scale_valid = anchor_loss_scale[valid_anchors].unsqueeze(-1)
        loss_lms = (anchor_loss_scale_valid * (points2d[:, :2] - points2d_gt[:, :2])).square().mean()

        return loss_lms


class SimpleDataloader():

    def __init__(self, seq_name, timesteps, cfg, intrinsics_provided = True):
        self.cfg = cfg

        # generate ray-based input tensors for each frame and store in dict of lists
        self.data = {}
        for timestep in timesteps:
            _data = prepare_data(timestep=timestep, seq_name=seq_name, intrinsics_provided=intrinsics_provided, downsample_factor=1/3)
            for data_key in _data.keys():
                if data_key not in self.data:
                    self.data[data_key] = [_data[data_key]]
                else:
                    self.data[data_key].append(_data[data_key])

        # load landmarks and compute how much expression changed
        all_detected_lms = np.load(f'{env_paths.DATA_TRACKING}/{seq_name}/pipnet/test.npy')
        self.change = np.nanmean(np.square((all_detected_lms[1:, ...] - all_detected_lms[:-1, ...])), axis=(1, 2))
        self.mean_change = np.mean(self.change)

        # need two loops: outer loops over timesteps inner loops over cameras
        # however, for a very long time I always used exactly one camera only
        self.full_gt_rgb = [[rgb[:, :].unsqueeze(0).float() for rgb in timestep_rgbs] for timestep_rgbs in self.data['rgb']]
        self.full_gt_mouth_interiors = [[mm.unsqueeze(0).float() for mm in cur_mm] for cur_mm in self.data['mouth_interior_mask']]
        self.full_gt_mask = [[mask[:].unsqueeze(0).unsqueeze(-1) for mask in timestep_masks] for timestep_masks in
                        self.data['segmentation_mask']]
        self.full_gt_mask_hq = [[mask[:].unsqueeze(0).unsqueeze(-1) for mask in timestep_masks] for timestep_masks in
                             self.data['segmentation_mask_hq']]
        self.full_gt_view_dir = [[view_dirs[:, :].unsqueeze(0).float() for view_dirs in timestep_view_dirss] for
                            timestep_view_dirss in self.data['view_dir']]
        self.full_gt_view_dir_hq = [[view_dirs[:, :].unsqueeze(0).float() for view_dirs in timestep_view_dirss] for
                                 timestep_view_dirss in self.data['view_dir_hq']]
        self.full_gt_cam_pos = [[cam_pos.unsqueeze(0).float() for cam_pos in timestep_cam_poss] for timestep_cam_poss in
                           self.data['cam_pos']]

        self.full_gt_normal_map = [[n[:, :].unsqueeze(0).float() for n in timestep_n] for timestep_n in self.data['normal_map']]
        self.full_w2c = [[torch.from_numpy(n[:, :]).unsqueeze(0).float() for n in timestep_n] for timestep_n in self.data['w2c']]

        self.full_in_dict = {
            'ray_dirs': self.full_gt_view_dir,
            'ray_dirs_hq': self.full_gt_view_dir_hq,
            'cam_loc': self.full_gt_cam_pos,
            'object_mask': self.full_gt_mask,
            'object_mask_hq': self.full_gt_mask_hq,
            'rgb': self.full_gt_rgb,
            'mouth_interior': self.full_gt_mouth_interiors,

            'normal_map': self.full_gt_normal_map,
            'w2c': self.full_w2c,
        }

    def subsample_ray_batch(self, timestep, camera_idx=0):
        # subsample rays, add batch_dim, push to GPU
        selected_rays = torch.randint(0, self.full_gt_rgb[timestep][camera_idx].shape[1], [self.cfg['opt']['rays_per_batch']])

        gt_rgb = self.full_gt_rgb[timestep][camera_idx][:, selected_rays, :].cuda()
        gt_mm = self.full_gt_mouth_interiors[timestep][camera_idx][:, selected_rays].cuda().float()

        gt_mask = self.full_gt_mask[timestep][camera_idx][:, selected_rays].unsqueeze(-1).cuda()
        gt_view_dir = self.full_gt_view_dir[timestep][camera_idx][:, selected_rays, :].cuda().float()
        gt_cam_pos = self.data['cam_pos'][timestep][camera_idx].unsqueeze(0).cuda().float()
        w2c = torch.from_numpy(self.data['w2c'][timestep][camera_idx]).cuda().float()

        gt_normals = self.full_gt_normal_map[timestep][camera_idx][:, selected_rays, :].cuda()

        in_dict = {
            'ray_dirs': gt_view_dir,
            'cam_loc': gt_cam_pos,
            'object_mask': gt_mask,
            'rgb': gt_rgb,
            'mm': gt_mm,

            'normals': gt_normals,
            'w2c': w2c,
        }
        return in_dict


class SimpleLogger():
    def __init__(self, optimizers):

        self.optimizers = optimizers

        return

    def init_step(self, timestep):
        self.wandb_dict = {}
        self.wandb_dict['opt_step'] = timestep

    def log_learning_rates(self):
        if 'id' in self.optimizers:
            for param_group in self.optimizers['id'].param_groups:
                self.wandb_dict['lr_opt'] = param_group['lr']
        for param_group in self.optimizers['expr'].param_groups:
            self.wandb_dict['lr_opt_expr'] = param_group['lr']
        for param_group in self.optimizers['sh'].param_groups:
            self.wandb_dict['lr_opt_sh'] = param_group['lr']
        for param_group in self.optimizers['pose'].param_groups:
            self.wandb_dict['lr_opt_pose'] = param_group['lr']
        for param_group in self.optimizers['scale'].param_groups:
            self.wandb_dict['lr_opt_scale'] = param_group['lr']

    def log_losses(self, lm_loss, lm_loss_scale,
                         reg_smooth, smoothness_scale,
                         reg_smooth_pose, smoothness_scale_pose,
                         loss_total):
        self.wandb_dict['loss_backwarp'] = lm_loss
        self.wandb_dict['loss_backwarp_scaled'] = lm_loss * lm_loss_scale
        self.wandb_dict['smooth_expr'] = reg_smooth
        self.wandb_dict['smooth_pose'] = reg_smooth_pose
        self.wandb_dict['smooth_expr_scaled'] = reg_smooth * smoothness_scale
        self.wandb_dict['smooth_pose_scaled'] = reg_smooth_pose * smoothness_scale_pose
        self.wandb_dict['loss'] = loss_total
        return


class SimpleOptimizerBundle():
    def __init__(self, fix_id, stage2, parameters):
        # eye-balled learning rates
        if stage2:
            lr_id = 0.0002*4
            lr_expr = 0.0001*2
            lr_sh = 0.0001*2
            lr_pose = 0.00003*2
            lr_scale = 0.0003*2
        else:
            lr_id = 0.0005 #0.001 #0.0005
            lr_expr = 0.0002
            lr_sh = 0.005 #0.002 ###0.05
            lr_pose = 0.02 #0.02 #0.005
            lr_scale = 0.005

        # optimizer for static components
        if not fix_id or stage2:
            opt = torch.optim.Adam(params=[parameters['id']['geo'], parameters['id']['app'],
                                           parameters['colorA'], parameters['colorb']] , lr=lr_id)
        # dynamic components
        opt_expr = torch.optim.SparseAdam(params=parameters['expr'].parameters(), lr=lr_expr)
        params_pose = torch.cat(parameters['pose'], dim=-1)
        pose_codebook = torch.nn.Embedding(num_embeddings=params_pose.shape[0],
                                           embedding_dim=params_pose.shape[1],
                                           sparse=True)
        pose_codebook.weight = torch.nn.Parameter(params_pose)
        self.pose_codebook = pose_codebook
        optim_pose = torch.optim.SparseAdam(params=pose_codebook.parameters(), lr=lr_pose)

        # also static components
        if fix_id and not stage2:
            parameters['scale'].requires_grad = False
            parameters['sh'].requires_grad = False
        opt_sh = torch.optim.Adam(params=[parameters['sh']], lr=lr_sh)
        optim_scale = torch.optim.Adam(params=[parameters['scale']], lr=lr_scale)

        self.optimizers = {
            'pose': optim_pose,
            'scale': optim_scale,
            'sh': opt_sh,
            'expr': opt_expr
        }
        if not fix_id or stage2:
            self.optimizers['id'] = opt


class SimpleSchdeuler():
    '''
    A poor implementation of a purely empirical schudling for the following things:
    (1) Learning rates of the seprate optimizers
    (2) Variance Scheudling for the NeuS rendering
    (3) Scheduling for the landmark loss --> Landmarks should be important as coarse guidance in the beginning but are less relevant later on when the details matter more
    (4) Smoothness terms, only relevant for tracking a sequence, also I never really figured out how to make the tracking perfectly smooth
    '''
    def __init__(self, fix_id, stage2, timesteps, n_timesteps, simple_data_loader, cfg, lr_scale, optimizers):
        self.fix_id = fix_id
        self.stage2 = stage2
        self.n_timesteps = n_timesteps
        self.timesteps = timesteps
        self.simple_data_loader = simple_data_loader
        self.cfg = cfg
        self.lr_scale = lr_scale
        self.optimizers = optimizers
        epoch_mult = 1
        if self.fix_id:
            if self.stage2:
                epoch_mult = 0.25
            else:
                # for stage 1 we adopt the number of epochs based on how much the landmark detections moved in 2D compared
                # to the average movement of the sequence
                if self.n_timesteps == 1 and self.timesteps[0] > 0:
                    if self.simple_data_loader.change[self.timesteps[0] - 1] > 2 * self.simple_data_loader.mean_change:
                        epoch_mult = 0.5
                    elif self.simple_data_loader.change[
                        self.timesteps[0] - 1] > 0.5 * self.simple_data_loader.mean_change:
                        epoch_mult = 0.25
                    else:
                        epoch_mult = 0.1

        n_steps = int(self.cfg['opt']['n_epochs'] * epoch_mult * self.n_timesteps)
        #n_steps = int(10 * epoch_mult * self.n_timesteps)
        print(f'Using step multiplier of {epoch_mult}, resulting in {n_steps} steps of optmiization')
        self.n_steps = n_steps
        self.epoch_mult = epoch_mult

    def step(self, epoch):
        # eye-balled LR schadule
        if not self.stage2:
            if epoch == int(0.15 * self.n_steps):
                for param_group in self.optimizers['pose'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['scale'].param_groups:
                    param_group["lr"] *= 1 / 2
            #TODO everything should be relative here, DONT USE ABSOLUTE VALUES HIDDEN IN HERE
            if epoch == int(0.5 * self.n_steps):
                if 'id' in self.optimizers:
                    for param_group in self.optimizers['id'].param_groups:
                        param_group["lr"] = 0.0005 * self.lr_scale  # 0.001
                for param_group in self.optimizers['expr'].param_groups:
                    param_group["lr"] = 0.0005 * self.lr_scale  # 0.001
                # for param_group in opt.param_groups:
                #   param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['sh'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['pose'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['scale'].param_groups:
                    param_group["lr"] *= 1 / 2

            if epoch == int(0.7 * self.n_steps):
                if 'id' in self.optimizers:
                    for param_group in self.optimizers['id'].param_groups:
                        param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['expr'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['pose'].param_groups:
                    param_group["lr"] *= 1 / 3
                for param_group in self.optimizers['scale'].param_groups:
                    param_group["lr"] *= 1 / 3
                # if loss_function.geo3d_weight > 0 and loss_function.rgb3d_weight == 0:
                #    loss_function.rgb3d_weight = 30
                #    loss_function.geo3d_weight /= 2
            if epoch == int(0.9*self.n_steps):
                if 'id' in self.optimizers:
                    for param_group in self.optimizers['id'].param_groups:
                        param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['expr'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['sh'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['pose'].param_groups:
                    param_group["lr"] *= 1 / 3
                for param_group in self.optimizers['scale'].param_groups:
                    param_group["lr"] *= 1 / 3
        else:
            if epoch == self.n_steps // 2:
                if 'id' in self.optimizers:
                    for param_group in self.optimizers['id'].param_groups:
                        param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['expr'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['sh'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['pose'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['scale'].param_groups:
                    param_group["lr"] *= 1 / 2
            if epoch == int(self.n_steps * 3 / 4):
                if 'id' in self.optimizers:
                    for param_group in self.optimizers['id'].param_groups:
                        param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['expr'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['sh'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['pose'].param_groups:
                    param_group["lr"] *= 1 / 2
                for param_group in self.optimizers['scale'].param_groups:
                    param_group["lr"] *= 1 / 2

    def get_neus_variance(self, epoch):
        if self.fix_id or self.stage2:
            variance = min(0.5 + ((epoch / self.n_timesteps) // (50 * self.epoch_mult) * 0.15), 0.8) #TODO also improve this schedule for second stage video tracing
        else:  # start more blurry for inital identity optimization in stage 1
            variance = 0.3 + 0.5 * min((epoch / (self.n_steps * 0.85)), 1)
            if np.random.rand() < 0.25 and epoch < self.n_steps*0.85:
                rnd_var = np.random.rand() * 0.3
                variance = variance - rnd_var
                variance = max(variance, 0.3)


        return variance

    def get_lm_loss_weight(self, step):
        if self.stage2:
            lm_loss_weight = 5
        else:
            if step < 0.1 * self.n_steps:
                lm_loss_weight = 100
            elif step < 0.2 * self.n_steps:
                lm_loss_weight = 50
            elif step < 0.4 * self.n_steps:
                lm_loss_weight = 20

            else:
                lm_loss_weight = 10
        return lm_loss_weight

    def get_smoothness_loss_weight(self, step):
        '''
        Loss schedule for smoothness constrained that is used in stage 2.
        '''
        if step > self.n_steps // 2:
            scale_lat_expr = 5000
        else:
            scale_lat_expr = 10000
        if step > self.n_steps // 2:
            scale_pose = 250000
        else:
            scale_pose = 500000
        return scale_lat_expr, scale_pose


class TrackerMonoNPHM():

    def __init__(self,
                 net,
                 cfg,
                 seq_name,
                 timesteps,
                 num_views,
                 exp_dirs,
                 fix_id=False,
                 colorA=None,
                 colorb=None,
                 params_pose=None,
                 lr_scale=1,
                 sh_coeffs=None,
                 stage2=False,
                 progress_interval = 300, # set this to be negative to disable progress/debugging renerings
                 intrinsics_provided = True,
                 disable_mouth : bool = True,
                 lambda_normals : float = 0.0
                 ):

        lr_scale_pose = lr_scale
        # only relevant for tracking; TODO: why though?
        if fix_id:
            lr_scale_pose /= 10

        self.lr_scale = lr_scale
        self.lr_scale_pose = lr_scale_pose

        self.net = net
        self.exp_dirs = exp_dirs
        self.timesteps = timesteps
        self.n_timesteps = len(timesteps)
        self.fix_id = fix_id
        self.stage2 = stage2
        self.seq_name = seq_name
        self.cfg = cfg
        self.intrinsics_provided = intrinsics_provided

        self.progress_dirs = []
        for exp_dir in self.exp_dirs:
            self.progress_dirs.append(f'{exp_dir}/progress/')
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(self.progress_dirs[-1], exist_ok=True)

        self.renderer = RendererMonoNPHM(net, cfg['ray_tracer'], sh_coeffs=sh_coeffs).cuda()

        self.simple_data_loader = SimpleDataloader(seq_name, timesteps, cfg, intrinsics_provided=self.intrinsics_provided)

        self.parameters = self.set_up_parameters(colorA, colorb, params_pose)

        # set up optimizers
        self.optimizer_bundle = SimpleOptimizerBundle(fix_id, stage2, parameters=self.parameters)

        self.progress_images = {e: {i: [] for i in range(num_views)} for e in range(len(timesteps))}

        self.loss_function = ImageSpaceLoss(rgb_weight=cfg['opt']['lambda_rgb'],
                                            silhouette_weight=cfg['opt']['lambda_mask'],
                                            lambdas_reg=cfg['opt']['lambdas_reg'],
                                            reg_weight_expr=cfg['opt']['lambda_reg_expr'],
                                            disable_mouth=disable_mouth,
                                            normal_weight=lambda_normals,
                                       )


        self.landmark_loss_function = LandmarkLoss(n_expr=len(timesteps),
                                                   pose_codebook=self.optimizer_bundle.pose_codebook,
                                                   scale_param=self.parameters['scale']
                                                )


        self.simple_logger = SimpleLogger(self.optimizer_bundle.optimizers)

        self.simple_scheduler = SimpleSchdeuler(fix_id, stage2, self.timesteps,
                                                self.n_timesteps, self.simple_data_loader, self.cfg,
                                                lr_scale, self.optimizer_bundle.optimizers)

        self.progress_interval = self.simple_scheduler.n_timesteps * progress_interval


    def set_up_parameters(self, colorA, colorb, params_pose):
        if self.fix_id and not self.stage2:
            params = []
        else:
            self.renderer.implicit_network.latent_code_id['geo'].requires_grad = True
            self.renderer.implicit_network.latent_code_id['app'].requires_grad = True
            params = [self.renderer.implicit_network.latent_code_id['geo'], self.renderer.implicit_network.latent_code_id['app']]
        if colorA is not None:
            colorA = torch.from_numpy(colorA).cuda()
            colorb = torch.from_numpy(colorb).cuda()
        else:
            colorA = torch.eye(3, device='cuda')
            colorb = torch.zeros([1, 3], device='cuda')
            colorA.requires_grad = True
            colorb.requires_grad = True
            params.append(colorA)
            params.append(colorb)

        #TODO clean this up big time
        # init pose with previous estimates
        if params_pose is not None and params_pose[0] is not None:
            #TODO fix dirty hack
            if len(params_pose) == 3:
                params_pose = [params_pose]
            params_pose_individual = []
            # numpy to torch and 3x3 rotm to rodrigues parameterization
            for timestep in range(len(params_pose)):
                _params_pose = [
                    so3_log_map(torch.from_numpy(params_pose[timestep][0])).float().cuda(),
                    torch.from_numpy(params_pose[timestep][1]).squeeze().unsqueeze(0).float().cuda(),
                    torch.from_numpy(params_pose[timestep][2]).float().cuda(),
                ]
                if len(_params_pose[0].shape) < 2:
                    _params_pose[0] = _params_pose[0].unsqueeze(0)
                params_pose_individual.append(_params_pose)

            params_pose_rot = torch.cat([params_pose_individual[i][0] for i in range(self.n_timesteps)], dim=0)
            params_pose_trans = torch.cat([params_pose_individual[i][1] for i in range(self.n_timesteps)], dim=0)
            scale_param = params_pose_individual[0][2]
            scale_param.requires_grad = False  # only optimized for first frame
            params_pose = [params_pose_rot, params_pose_trans]
        else:
            # init head pose correction with identity
            params_pose = []
            rot_params = so3_log_map(torch.eye(3).unsqueeze(0)).repeat(self.n_timesteps, 1).float().cuda()
            trans_params = torch.zeros([self.n_timesteps, 3]).float().cuda()
            scale_param = torch.ones([1]).float().cuda()
            rot_params.requires_grad = True
            trans_params.requires_grad = True
            scale_param.requires_grad = False
            params_pose.append(rot_params)
            params_pose.append(trans_params)

        # avoid completely empty parameter list
        params.append(torch.zeros([1], device='cuda', requires_grad=True))

        parameters = {
            'id': self.renderer.implicit_network.latent_code_id,
            'expr': self.renderer.implicit_network.latent_codes_expr,
            'pose': params_pose,
            'scale': scale_param,
            'sh': self.renderer.sh_coeffs,
            'colorA': colorA,
            'colorb': colorb,
        }
        return parameters


    def compute_smoothness_loss(self, timestep):
        n_terms_smooth = 0
        if timestep > 0:
            reg_loss_smooth1 = (torch.norm(
                self.renderer.implicit_network.latent_codes_expr(torch.tensor([timestep], device='cuda')) -
                self.renderer.implicit_network.latent_codes_expr(torch.tensor([timestep - 1], device='cuda')),
                dim=-1) ** 2).mean()
            reg_loss_smooth1_pose = (torch.norm(
                self.optimizer_bundle.pose_codebook(torch.tensor([timestep], device='cuda')) -
                self.optimizer_bundle.pose_codebook(torch.tensor([timestep - 1], device='cuda')),
                dim=-1) ** 2).mean()

            n_terms_smooth += 1
        else:
            reg_loss_smooth1 = torch.zeros([1], device='cuda')
            reg_loss_smooth1_pose = torch.zeros([1], device='cuda')
        if timestep < self.n_timesteps - 1:
            reg_loss_smooth2 = (torch.norm(
                self.renderer.implicit_network.latent_codes_expr(torch.tensor([timestep], device='cuda')) -
                self.renderer.implicit_network.latent_codes_expr(torch.tensor([timestep + 1], device='cuda')),
                dim=-1) ** 2).mean()
            reg_loss_smooth2_pose = (torch.norm(
                self.optimizer_bundle.pose_codebook(torch.tensor([timestep], device='cuda')) -
                self.optimizer_bundle.pose_codebook(torch.tensor([timestep + 1], device='cuda')),
                dim=-1) ** 2).mean()
            n_terms_smooth += 1
        else:
            reg_loss_smooth2 = torch.zeros([1], device='cuda')

            reg_loss_smooth2_pose = torch.zeros([1], device='cuda')

        reg_smooth = torch.zeros([1], device='cuda')
        reg_smooth_pose = torch.zeros([1], device='cuda')
        if n_terms_smooth > 0:
            reg_smooth = (reg_loss_smooth1 + reg_loss_smooth2) / n_terms_smooth
            reg_smooth_pose = (reg_loss_smooth1_pose + reg_loss_smooth2_pose) / n_terms_smooth

        return reg_smooth, reg_smooth_pose


    def run_tracking(self):


        for opt_step in range(self.simple_scheduler.n_steps):

            self.simple_logger.init_step(opt_step)

            for opt_key, opt in self.optimizer_bundle.optimizers.items():
                opt.zero_grad()

            # eye-balled LR schadule
            self.simple_scheduler.step(opt_step)
            self.simple_logger.log_learning_rates()

            cur_timestep = np.random.randint(0, self.n_timesteps)  # sample random frame
            in_dict = self.simple_data_loader.subsample_ray_batch(cur_timestep) # get batch of data

            # fixed variance schedule used in the NeuS formulation (i.e. it is not optimized for)
            variance = self.simple_scheduler.get_neus_variance(opt_step)

            _pose_params = self.optimizer_bundle.pose_codebook(torch.tensor([cur_timestep], device='cuda'))

            # diff. implicit-surface rendering
            out_dict = self.renderer(in_dict,
                                     cur_timestep,
                                     skip_render=False,
                                     neus_variance=variance,
                                     pose_params=[_pose_params[:, :3],
                                     _pose_params[:, 3:6],
                                     self.parameters['scale']],
                                     )

            # obtain latent regularization
            reg_loss = self.renderer.implicit_network.GTA.id_model.get_reg_loss(self.renderer.implicit_network.latent_code_id)
            reg_loss_expr = (torch.norm(
                self.renderer.implicit_network.latent_codes_expr(torch.tensor([cur_timestep], device='cuda')),
                dim=-1) ** 2).mean()
            out_dict['reg_loss'] = reg_loss
            out_dict['reg_loss_expr'] = reg_loss_expr

            # apply affine color correction
            out_dict['rgb_values'] = out_dict['rgb_values'] @ self.parameters['colorA'] + self.parameters['colorb']

            # further scheduling; the idea is that in the beginning we only have a coarse estimate and large loss,
            # therefore we need to regularize the latent codes a lot, in order for the optimization to not derail.
            if opt_step > self.simple_scheduler.n_steps * 0.6:
                out_dict['reg_loss']['reg_loc_geo'] /= 8
                out_dict['reg_loss']['reg_loc_app'] /= 8
                out_dict['reg_loss']['reg_global_geo'] /= 8
                out_dict['reg_loss']['reg_global_app'] /= 8
                out_dict['reg_loss']['symm_dist_geo'] /= 20
                out_dict['reg_loss']['symm_dist_app'] /= 20
                out_dict['reg_loss_expr'] /= 8 #4
            elif opt_step > self.simple_scheduler.n_steps // 3:
                out_dict['reg_loss']['reg_loc_geo'] /= 4
                out_dict['reg_loss']['reg_loc_app'] /= 4
                out_dict['reg_loss']['reg_global_geo'] /= 4
                out_dict['reg_loss']['reg_global_app'] /= 4
                out_dict['reg_loss']['symm_dist_geo'] /= 5
                out_dict['reg_loss']['symm_dist_app'] /= 5
                out_dict['reg_loss_expr'] /= 3 #2

            # compute loss functions in image space
            loss_dict = self.loss_function(out_dict, in_dict, wandb_dict=self.simple_logger.wandb_dict)

            loss_landmarks = self.landmark_loss_function(self.renderer, cur_timestep, self.simple_data_loader.data)

            # schedule for landmark loss
            lm_loss_scale = self.simple_scheduler.get_lm_loss_weight(opt_step)
            loss_dict['loss'] += loss_landmarks * lm_loss_scale

            # smoothness constraints
            if self.stage2:
                reg_smooth, reg_smooth_pose = self.compute_smoothness_loss(cur_timestep)
            else:
                reg_smooth, reg_smooth_pose = torch.zeros([1], device='cuda'), torch.zeros([1], device='cuda')

            loss_dict['smoothness_expr'] = reg_smooth
            loss_dict['smoothness_pose'] = reg_smooth_pose

            # apply smoothness loss terms
            smoothness_scale_expr, smoothness_scale_pose = self.simple_scheduler.get_smoothness_loss_weight(opt_step)
            loss_dict['loss'] += reg_smooth.squeeze() * smoothness_scale_expr
            loss_dict['loss'] += reg_smooth_pose.squeeze() * smoothness_scale_pose

            # log remaining losses
            self.simple_logger.log_losses(loss_landmarks.item(), lm_loss_scale,
                                     reg_smooth.item(), smoothness_scale_expr,
                                     reg_smooth_pose.item(), smoothness_scale_pose,
                                     loss_dict['loss'].item())

            loss_dict['loss'].backward()

            torch.nn.utils.clip_grad_norm_([self.net.latent_code_id['geo']], 0.1)
            torch.nn.utils.clip_grad_norm_([self.net.latent_code_id['app']], 0.1)
            # cannot do clippting for expr codes, since latent expression codes are implemented as sparse Embedding,
            #   which doesn't clipping of gradients
            # torch.nn.utils.clip_grad_norm_(net.latent_codes_expr., 0.1)

            for opt_key, opt in self.optimizer_bundle.optimizers.items():
                opt.step()

            print_str = f'Opt. Step: {opt_step}, '
            for k in loss_dict.keys():
                print_str += f'{k}: {loss_dict[k].item():3.5f}, '

            print_str += f' LM loss: {loss_landmarks.item()}'

            print(print_str)

            # progress can be logged using wandb
            # wandb.log(simple_logger.wandb_dict)

            # regularly render images for debugging/analysis reasons
            if self.progress_interval > 0 and opt_step % int(self.progress_interval * self.simple_scheduler.epoch_mult) == 0 and opt_step > 0:
                for cur_timestep in range(self.n_timesteps):
                    if self.landmark_loss_function.anchors_posed[cur_timestep] is None:
                        continue # safety check, since we want to render the 2D anchors, they need to be available from a previous optimization step
                    valid_anchors = self.landmark_loss_function.valid_anchors_posed[cur_timestep]
                    x_posed = torch.from_numpy(self.landmark_loss_function.anchors_posed[cur_timestep]).cuda()
                    I_composed = self.render_progress(cur_timestep, valid_anchors, x_posed, variance=variance)
                    I_composed = Image.fromarray(I_composed)
                    I_composed.save(f'{self.progress_dirs[cur_timestep]}/step{opt_step:04d}.png')

            opt_step += 1

        torch.cuda.empty_cache()

        if False: # Optionally you can perform a higher resolution + higher quality ( more samples along a ray) rendering once the optimizatin is finished
            for cur_timestep in range(self.n_timesteps):

                I_composed = self.render_progress(cur_timestep, None, None, variance=0.8, is_final=True)
                I_composed = Image.fromarray(I_composed)
                I_composed.save(f'{self.exp_dirs[cur_timestep]}/final.png')

        self.reconstruct_meshes()

        # save results
        params_pose, colorA, colorb = self.save_result()

        return {
            'lat_rep_id': self.net.latent_code_id,
            'lat_rep_expr': self.net.latent_codes_expr.weight.data,
            'colorA': colorA,
            'colorb': colorb,
            'params_pose': params_pose,
            'sh_coeffs': self.renderer.sh_coeffs.detach().cpu().numpy(),
        }

    def reconstruct_meshes(self):
        # prepare grid for marching cubes
        grid_points = create_grid_points_from_bounds(self.cfg['reconstruction']['min'],
                                                     self.cfg['reconstruction']['max'],
                                                     self.cfg['reconstruction']['res'])
        grid_points = torch.from_numpy(grid_points).cuda().float()
        grid_points = torch.reshape(grid_points, (1, len(grid_points), 3)).cuda()

        for e, timestep in enumerate(self.timesteps):
            condition = self.renderer.implicit_network.latent_code_id
            condition.update(
                {'exp': self.renderer.implicit_network.latent_codes_expr(torch.tensor([e], device='cuda')).unsqueeze(
                    0)})

            original_nneigh = self.renderer.implicit_network.GTA.id_model.num_neighbors
            self.renderer.implicit_network.GTA.id_model.num_neighbors = 2*original_nneigh
            # perform marching cubes
            logits = get_logits(self.renderer.implicit_network.GTA, condition, grid_points, nbatch_points=40000)
            mesh = mesh_from_logits(logits.copy(),
                                    self.cfg['reconstruction']['min'],
                                    self.cfg['reconstruction']['max'],
                                    self.cfg['reconstruction']['res'])

            # color mesh by querying color MLP with vertex positions
            vertex_color = get_vertex_color(self.renderer.implicit_network.GTA,
                                            encoding=condition,
                                            vertices=torch.from_numpy(mesh.vertices).float().unsqueeze(0).cuda(),
                                            nbatch_points=40000,
                                            )
            vertex_color = ((vertex_color / 255) - 0.5) * 2
            #if self.parameters['colorA'] is not None:
            #    vertex_color = vertex_color @ self.parameters['colorA'].detach().cpu().numpy() + self.parameters[
            #        'colorb'].detach().cpu().numpy()
            vertex_color = ((vertex_color + 1) / 2 * 255).astype(np.uint8)
            mesh.visual.vertex_colors = vertex_color
            mesh.export(self.exp_dirs[e] + 'mesh.ply'.format(self.seq_name, timestep))
            self.renderer.implicit_network.GTA.id_model.num_neighbors = original_nneigh
            torch.cuda.empty_cache()

    def save_result(self):
        for e, timestep in enumerate(self.timesteps):
            # save reconstructed latent codes:
            np.save(f'{self.exp_dirs[e]}/z_geo.npy', self.net.latent_code_id['geo'].detach().cpu().numpy())
            np.save(f'{self.exp_dirs[e]}/z_app.npy', self.net.latent_code_id['app'].detach().cpu().numpy())
            if self.parameters['colorA'] is not None:
                np.save(f'{self.exp_dirs[e]}/colorA.npy', self.parameters['colorA'].detach().cpu().numpy())
                np.save(f'{self.exp_dirs[e]}/colorb.npy', self.parameters['colorb'].detach().cpu().numpy())
                np.save(f'{self.exp_dirs[e]}/z_exp.npy', self.net.latent_codes_expr.weight.data.detach().cpu().numpy())

        colorA, colorb = self.parameters['colorA'].detach().cpu().numpy(), self.parameters['colorb'].detach().cpu().numpy()

        for e, timestep in enumerate(self.timesteps):
            _pose_params = self.optimizer_bundle.pose_codebook(torch.tensor([e], device='cuda'))

            np.save(f'{self.exp_dirs[e]}/scale.npy', self.parameters['scale'].detach().cpu().numpy())
            np.save(f'{self.exp_dirs[e]}/trans.npy', _pose_params[:, 3:6].detach().cpu().numpy())
            np.save(f'{self.exp_dirs[e]}/rot.npy', so3_exp_map(_pose_params[:, :3]).detach().cpu().numpy())

            np.save(f'{self.exp_dirs[e]}/sh_coeffs.npy', self.renderer.sh_coeffs.detach().cpu().numpy())

            np.save(f'{self.exp_dirs[e]}/intrinsics.npy', self.simple_data_loader.data['intrinsics'][e][0])
            np.save(f'{self.exp_dirs[e]}/w2c.npy', self.simple_data_loader.data['w2c'][e][0])



        params_pose = [so3_exp_map(_pose_params[:, :3]).detach().cpu().numpy(),
                       _pose_params[:, 3:6].detach().cpu().numpy(),
                       self.parameters['scale'].detach().cpu().numpy()]

        for e, timestep in enumerate(self.timesteps):
            if self.landmark_loss_function.anchors_posed[e] is not None:
                np.save(f'{self.exp_dirs[e]}/anchors.npy', self.landmark_loss_function.anchors_posed[e])

        return params_pose, colorA, colorb



    def render_image(self,
                     timestep,
                     in_dict,
                     n_batch_points=2000,
                     w=None,
                     h=None,
                     variance=None,
                     pose_params=None,
                     use_sh=False,
                     color_correction=None,
                     is_hq=False):
        self.renderer.eval()
        torch.cuda.empty_cache()
        if use_sh:
            n_batch_points = 500 if not is_hq else 300  # bc. of SH cannot handle large batch sizes
        if is_hq:
            ray_dir_split = torch.split(in_dict['ray_dirs_hq'], n_batch_points, dim=1)
            object_mask_split = torch.split(in_dict['object_mask_hq'], n_batch_points, dim=1)
        else:
            ray_dir_split = torch.split(in_dict['ray_dirs'], n_batch_points, dim=1)
            object_mask_split = torch.split(in_dict['object_mask'], n_batch_points, dim=1)
        color_list = []
        list_weights_sum = []
        list_depths = []
        list_normals = []
        for chunk_i, (ray_dirs, object_masks) in enumerate(zip(ray_dir_split, object_mask_split)):
            cur_in_dict = {'ray_dirs': ray_dirs,
                           'object_mask': object_masks,
                           'cam_loc': in_dict['cam_loc'],
                           'w2c': in_dict['w2c'].squeeze(),
                           }
            if use_sh:
                out_dict = self.renderer(cur_in_dict, timestep, compute_non_convergent=True, neus_variance=variance,
                               pose_params=[pose_params[0], pose_params[1], pose_params[2]],
                               use_SH=use_sh, num_samples=64 if is_hq else 32)  # , debug_plot=chunk_i==0)
            else:
                with torch.no_grad():
                    out_dict = self.renderer(cur_in_dict, timestep, compute_non_convergent=True, neus_variance=variance,
                                   pose_params=[pose_params[0], pose_params[1], pose_params[2]], use_SH=use_sh)

            torch.cuda.empty_cache()

            color = out_dict['rgb_values'].detach()
            color = color.squeeze()
            color_list.append(color.squeeze(0).detach().cpu())
            if out_dict['weights_sum'] is not None:
                list_weights_sum.append(out_dict['weights_sum'].squeeze(0).detach().cpu())
                list_depths.append(out_dict['weighted_depth'].squeeze(0).detach().cpu())
            if out_dict['nphm_space_normals'] is not None:
                list_normals.append(out_dict['nphm_space_normals'].squeeze(0).detach().cpu())


            del out_dict, color
            torch.cuda.empty_cache()
        color = np.concatenate(color_list, axis=0)
        if len(list_weights_sum) > 0:
            _weights_sum = np.concatenate(list_weights_sum, axis=0)
        if len(list_depths) > 0:
            depths = np.concatenate(list_depths, axis=0)
        if len(list_normals) > 0:
            normals = np.concatenate(list_normals, axis=0)
        color = np.reshape(color, [h, w, 3])
        if len(list_weights_sum) > 0:
            weights_sum = np.reshape(_weights_sum, [h, w]).astype(np.float32)
            weights_img = ((np.tile(np.reshape(_weights_sum, [h, w]).astype(np.float32)[:, :, np.newaxis],
                                    [1, 1, 3]) / 1) * 255).astype(np.uint8)
        else:
            weights_img = None
            weights_sum = None

        if len(list_depths) > 0:
            #print('DEPTH:', depths.min(), depths.max())
            min_depth = depths.min()

            depths = (np.tile(np.reshape(depths, [h, w]).astype(np.float32)[:, :, np.newaxis], [1, 1, 3]) - min_depth)
            max_depth = depths.max()
            depths = depths / max_depth
            depths = (depths * 255).astype(np.uint8)
        else:
            depths = None
        if len(list_normals) > 0:
            normals = np.reshape(normals, [h, w, 3]).astype(np.float32)

            normals = (normals + 1) / 2 * 255
            normals = normals.astype(np.uint8)

        # apply color correction
        if color_correction is not None:
            h, w, _ = color.shape
            color = color.reshape(-1, 3)
            color = color @ color_correction['A'] + color_correction['b']
            color = color.reshape(h, w, 3)
        color = (color + 1) / 2 * 255

        color = np.clip(color, 0, 255)
        color = color.astype(np.uint8)

        I = Image.fromarray(color)
        self.renderer.train()
        return I, weights_sum, weights_img, depths, normals

    #TODO: clean this up
    #TODO: use loss function to compute error plots!
    def render_progress(self,
                        timestep,
                        valid_anchors,
                        anchors_posed,
                        variance, is_final=False):
        #variance = 0.6
        _cond = {k: self.renderer.implicit_network.latent_code_id[k].clone() for k in
                 self.renderer.implicit_network.latent_code_id.keys()}
        _cond['exp'] = self.renderer.implicit_network.latent_codes_expr(
            torch.tensor([timestep], device='cuda')).unsqueeze(0)

        _pose_params = self.optimizer_bundle.pose_codebook(torch.tensor([timestep], device='cuda'))



        index_anchors = torch.from_numpy(ANCHOR_iBUG68_pairs_65[:, 0]).cuda()
        index_lms = torch.from_numpy(ANCHOR_iBUG68_pairs_65[:, 1]).cuda()

        intrinsics_key = 'intrinsics'
        w2c_key = 'w2c'
        width_key = 'width'
        height_key = 'height'
        if is_final:
            intrinsics_key = 'intrinsics_hq'
            width_key = 'width_hq'
            height_key = 'height_hq'




        _full_in_dict = {k: v[timestep][0].cuda() for k, v in self.simple_data_loader.full_in_dict.items()}

        # render image
        I, weights_sum, weights_img, depths, normals = self.render_image(
                                                           timestep,
                                                           _full_in_dict,
                                                           w=self.simple_data_loader.data[width_key][timestep][0],
                                                           h=self.simple_data_loader.data[height_key][timestep][0],
                                                           n_batch_points=15000,
                                                           variance=variance,
                                                           pose_params=[_pose_params[:, :3],
                                                                        _pose_params[:, 3:6],
                                                                        self.parameters['scale']],
                                                           use_sh=True,
                                                           color_correction={
                                                               'A': self.parameters['colorA'].detach().cpu().numpy(),
                                                               'b': self.parameters['colorb'].detach().cpu().numpy()},
                                                          is_hq=is_final)
        if is_final:
            return np.array(I)

        anchors_posed = ((anchors_posed - _pose_params[:, 3:6]) / self.parameters['scale']) @ so3_exp_map(
            _pose_params[:, :3]).squeeze()
        # project posed anchors into screen space
        points2d = project_points_torch(anchors_posed.squeeze()[index_anchors, :],
                                        torch.from_numpy(
                                            self.simple_data_loader.data[intrinsics_key][timestep][0]).to(
                                            anchors_posed.device).float(),  # view 0
                                        torch.from_numpy(self.simple_data_loader.data[w2c_key][timestep][0]).to(
                                            anchors_posed.device).float(),
                                        # self.simple_data_loader.data['width'][timestep][0]
                                        )
        points2d_gt = \
        torch.from_numpy(self.simple_data_loader.data['landmarks_2d'][timestep][0]).float().to(anchors_posed.device)[
            index_lms]
        points2d_px = points2d.detach().clone()
        points2d_gt_px = points2d_gt.detach().clone()


        gt_im = np.array(self.simple_data_loader.data['rgb'][timestep][0])
        gt_im = np.reshape(gt_im, (self.simple_data_loader.data['height'][timestep][0],
                                   self.simple_data_loader.data['width'][timestep][0], 3))
        gt_im = ((gt_im + 1) / 2 * 255).astype(np.uint8)
        _gt_im = gt_im.copy()

        gt_normals = np.array(self.simple_data_loader.data['normal_map'][timestep][0])
        gt_normals = np.reshape(gt_normals, (self.simple_data_loader.data['height'][timestep][0],
                                   self.simple_data_loader.data['width'][timestep][0], 3))
        gt_normals = ((gt_normals + 1) / 2 * 255).astype(np.uint8)

        # draw landmark detections
        if points2d_gt is not None and valid_anchors is not None:
            for anchor_idx in range(points2d_px.shape[0]):
                color_anchor = (100, 0, 0)
                color_lm = (0, 0, 255)
                color_line = (100, 225, 100)
                # invalid anchors are drawn with darker color
                if not valid_anchors[anchor_idx]:
                    color_anchor = (255 // 2, 0, 0)
                    color_lm = (0, 0, 255 // 2)
                    color_line = (100 // 2, 225 // 2, 100 // 2)
                # draw line connecting predicted and gt landmakrs
                _gt_im = cv2.line(_gt_im, (
                    int(points2d_px[anchor_idx][0].item()),
                    int(points2d_px[anchor_idx][1].item())), (
                                     int(points2d_gt_px[anchor_idx][0].item()),
                                     int(points2d_gt_px[anchor_idx][1].item())), color=color_line,
                                 thickness=1)
                # draw predicted landmarks
                _gt_im = cv2.circle(_gt_im, (
                    int(points2d_px[anchor_idx][0].item()),
                    int(points2d_px[anchor_idx][1].item())),
                                   radius=1, color=color_anchor, thickness=-1)
                # draw gt landmarks
                _gt_im = cv2.circle(_gt_im, (
                    int(points2d_gt_px[anchor_idx][0].item()),
                    int(points2d_gt_px[anchor_idx][1].item())),
                                   radius=1, color=color_lm, thickness=-1)

        # ---------------------------------------------------
        ############## Error Plots #########################
        # RGB error image, mask error image, and predicte depth
        # ---------------------------------------------------
        _I = (np.array(I) / 255 - 0.5) * 2
        gt_I = (gt_im / 255 - 0.5) * 2

        _N = (np.array(normals) / 255 - 0.5)*2
        gt_N = (gt_normals / 255 - 0.5) * 2

        facer_mask = _full_in_dict['object_mask'].detach().cpu().squeeze().reshape(_I.shape[0], _I.shape[1]).numpy()
        rgb_loss_mask = np.logical_not((facer_mask == 3) | (facer_mask == 0))
        foreground_mask = rgb_loss_mask.copy()

        rgb_error_img = np.mean(np.abs(_I - gt_I), axis=-1)
        rgb_error_img[~rgb_loss_mask] = 0
        rgb_error_img[facer_mask==14] /= 10

        normal_error_img = np.mean(np.abs(_N - gt_N), axis=-1)
        valid_normals = np.abs(np.linalg.norm(gt_N, axis=-1) - 1) < 0.02 #torch.abs(gt_N.norm(dim=-1) - 1) < 0.02
        normal_error_img[~(rgb_loss_mask & valid_normals)] = 0

        mm_mask = np.array(self.simple_data_loader.full_in_dict['mouth_interior'][timestep][0]) != 0
        mm_mask = np.reshape(mm_mask[0, :], (rgb_error_img.shape[0], rgb_error_img.shape[1]))
        if self.loss_function.disable_mouth:
            rgb_error_img[mm_mask] /= 25
        else:
            rgb_error_img[mm_mask] /= 2
        O_mask_torch = torch.from_numpy(np.array(foreground_mask)).float()
        if weights_sum is None:
            O_mask_pred = torch.zeros_like(O_mask_torch)
        else:
            O_mask_pred = torch.from_numpy(weights_sum).float().clamp(0, 1)

        O_mask_pred[torch.isnan(O_mask_pred)] = 0
        mask_error_img = F.binary_cross_entropy(O_mask_pred, O_mask_torch, reduction='none')

        cmap = plt.get_cmap('turbo')
        rgb_error_img = cmap(np.clip(rgb_error_img * self.loss_function.rgb_weight / 30 + 0.05, 0, 1),
                             bytes=True)[..., :3]
        mask_error_img = cmap(np.clip(mask_error_img * self.loss_function.silhouette_weight / 30 + 0.05, 0, 1),
                              bytes=True)[..., :3]
        normal_error_img = cmap(np.clip(normal_error_img * self.loss_function.rgb_weight / 30 + 0.05, 0, 1),
                             bytes=True)[..., :3]

        cmap_depth = plt.get_cmap('viridis')

        if depths is None:
            depths = np.zeros_like(mask_error_img)
        else:
            depths = cmap_depth(depths[..., 0] / 255, bytes=True)[..., :3]

        if normals is None:
            depths = np.zeros_like(mask_error_img)

        I_err_composed = np.concatenate([rgb_error_img, mask_error_img], axis=1)

        #I_err_composed = np.concatenate([I_err_composed, depths], axis=1)
        I_err_composed = np.concatenate([I_err_composed, normal_error_img], axis=1)

        # cat images for convenience
        composed_image = np.concatenate([_gt_im,
                                         np.array(I),
                                         normals, #weights_img if weights_img is not None else np.zeros_like(_gt_im),
                                         ], axis=1)

        I_composed = np.concatenate([composed_image, I_err_composed], axis=0)
        return I_composed




